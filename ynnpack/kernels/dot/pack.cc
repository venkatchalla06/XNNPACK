#include "ynnpack/kernels/dot/pack.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/kernels/transpose/interleave.h"
#include "ynnpack/kernels/transpose/transpose.h"

namespace ynn {

packer::packer(bool transpose, size_t elem_size_bits, size_t tile_m,
               size_t tile_n)
    : elem_size_bits(elem_size_bits), tile_m(tile_m), tile_n(tile_n) {
  // This operation is fusing 3 separate transposes (with padding as needed):
  //
  // 1. (optional) the caller might want to transpose the input prior to the
  // subsequent transposes (`transpose` = true). We assume this transpose is of
  // `tile_m` x elements at a time.
  // 2. A small transpose of `tile_m` x elements at a time
  // 3. A larger transpose of `tile_m * tile_m` x elements at a time
  //
  // If each transpose has an element size that is a multiple of the previous
  // transpose's element size, those transposes partially "cancel out",
  // resulting in a different transpose instead.
  if (transpose) {
    // We have all 3 transposes.
    // We're transposing columns of the input to rows of the output, but doing
    // `tile_m` of them at a time. In this case, (2) and (3) are equivalent to
    // "bitcasting" the input to have elements that are `tile_m` x larger (and
    // `tile_m` x fewer of them), and then transposing that. (3) is handled by a
    // loop over calls to this kernel below.
    // Since we're tiling this ourselves, we don't need the added overhead of a
    // tiled transpose.
    transpose_fn = get_transpose_kernel(elem_size_bits * tile_m);
    assert(transpose_fn);
  } else {
    // We only have (2) and (3).
    if (tile_m == 1) {
      // (2) is a no-op, we only need to do (3).
      transpose_blocks_fn = get_tiled_transpose(elem_size_bits * tile_n);
      assert(transpose_blocks_fn);
    } else {
      // We need to do (2).
      // We're interleaving rows of the input to produce rows of the output.
      interleave_fn = get_interleave_kernel(elem_size_bits, tile_m);
      assert(interleave_fn);
    }
  }
}

void packer::pack(size_t m, size_t n, size_t input_stride, const void* input,
                  size_t output_stride, size_t output_block_stride,
                  void* output) {
  if (transpose_blocks_fn) {
    assert(tile_m == 1);
    assert(m == 1 || output_stride == tile_n * elem_size_bits / 8);
    transpose_blocks_fn(ceil_div(n, tile_n), m, n * elem_size_bits / 8,
                        input_stride, input, output_block_stride, output);
  } else if (transpose_fn) {
    while (n > 0) {
      const size_t n_i = std::min(n, tile_n);
      transpose_fn(ceil_div(m, tile_m), n_i, m * elem_size_bits / 8,
                   input_stride, input, output_stride, output);
      size_t elem_count = std::max<size_t>(8 / elem_size_bits, 1);
      input = offset_bytes(input, input_stride * (tile_n / elem_count));
      output = offset_bytes(output, output_block_stride);
      n = sub_sat(n, tile_n);
    }
  } else if (interleave_fn) {
    while (n > 0) {
      const size_t n_i = std::min(n, tile_n);
      // In each row, we have a range that we produce via
      // interleaving, which handles padding of rows, but we also have
      // padding in the columns, which the interleave kernel does not
      // handle. Here we compute the size produced by interleaving
      // (including the padded rows), and then the size of the padding
      // in each row, which we set to 0.
      const size_t row_size = n_i * tile_m * elem_size_bits / 8;
      const size_t padding_size = (tile_n - n_i) * tile_m * elem_size_bits / 8;
      const void* input_i = input;
      void* output_i = output;
      for (size_t i = 0; i < m; i += tile_m) {
        const size_t m_i = std::min(m - i, tile_m);
        interleave_fn(tile_m, m_i, n_i, input_stride, input_i, output_i);
        memset(offset_bytes(output_i, row_size), 0, padding_size);
        input_i = offset_bytes(input_i, input_stride * tile_m);
        output_i = offset_bytes(output_i, output_stride);
      }
      input = offset_bytes(input, tile_n * elem_size_bits / 8);
      output = offset_bytes(output, output_block_stride);
      n = sub_sat(n, tile_n);
    }
  } else {
    YNN_UNREACHABLE;
  }
}

// Packs a 2D row-major matrix A of shape [m, k] into 2D row-major tiles of
// shape [block_m, tile_k], zero-padding partial tiles at the m and k
// boundaries.
//
// Before (`input`, shape [m, k], row stride `input_stride` bytes):
//   input(r, c) is at byte offset: r * input_stride + c * elem_size
//
//   Example (m = 3, k = 5, block_m = 2, tile_k = 3):
//     [ a00 a01 a02 | a03 a04 ]
//     [ a10 a11 a12 | a13 a14 ]
//     --------------+----------
//     [ a20 a21 a22 | a23 a24 ]
//
// After (`output`, [output_m_blocks, output_k_blocks] tiles of
// [block_m, tile_k]):
//   Each tile (mo, ko) is contiguous row-major of shape [block_m, tile_k],
//   where r = mo * block_m + mi and c = ko * tile_k + ki:
//   output(mo, ko, mi, ki) is at byte offset:
//     mo * output_mo_stride + ko * output_ko_stride +
//     (mi * tile_k + ki) * elem_size
//
//   Tile (mo=0, ko=0) [offset: 0 * output_mo_stride + 0 * output_ko_stride]:
//     [ a00 a01 a02 ]
//     [ a10 a11 a12 ]
//   Tile (mo=0, ko=1) [offset: 0 * output_mo_stride + 1 * output_ko_stride]:
//     [ a03 a04  0  ]  <-- k padded to tile_k
//     [ a13 a14  0  ]
//   Tile (mo=1, ko=0) [offset: 1 * output_mo_stride + 0 * output_ko_stride]:
//     [ a20 a21 a22 ]
//     [  0   0   0  ]  <-- m padded to block_m
//   Tile (mo=1, ko=1) [offset: 1 * output_mo_stride + 1 * output_ko_stride]:
//     [ a23 a24  0  ]
//     [  0   0   0  ]
void pack_a(size_t m, size_t k, size_t block_m, size_t tile_k, size_t elem_size,
            size_t input_stride, const void* input, size_t output_ko_stride,
            size_t output_mo_stride, void* output, size_t output_m_blocks,
            size_t output_k_blocks) {
  const size_t tile_k_bytes = tile_k * elem_size;
  const size_t k_bytes = k * elem_size;
  if (output_k_blocks == 0) {
    output_k_blocks = ceil_div(k, tile_k);
  }
  if (output_m_blocks == 0) {
    output_m_blocks = ceil_div(m, block_m);
  }
  const auto* in_bytes = static_cast<const uint8_t*>(input);
  auto* out_bytes = static_cast<uint8_t*>(output);

  for (size_t block_m_i = 0; block_m_i < output_m_blocks; ++block_m_i) {
    const size_t m_begin = block_m_i * block_m;
    const size_t m_avail = std::min(block_m, sub_sat(m, m_begin));
    for (size_t block_k_i = 0; block_k_i < output_k_blocks; ++block_k_i) {
      const size_t k_offset_bytes = block_k_i * tile_k_bytes;
      const size_t k_avail_bytes =
          (k_offset_bytes < k_bytes)
              ? std::min(tile_k_bytes, k_bytes - k_offset_bytes)
              : 0;
      const size_t k_pad_bytes = tile_k_bytes - k_avail_bytes;

      uint8_t* out = out_bytes + block_k_i * output_ko_stride +
                     block_m_i * output_mo_stride;
      const uint8_t* in = in_bytes + m_begin * input_stride + k_offset_bytes;
      for (size_t m_i = 0; m_i < m_avail; ++m_i) {
        memcpy(out, in, k_avail_bytes);
        if (k_pad_bytes > 0) {
          memset(out + k_avail_bytes, 0, k_pad_bytes);
        }
        out += tile_k_bytes;
        in += input_stride;
      }
      if (m_avail < block_m) {
        memset(out, 0, (block_m - m_avail) * tile_k_bytes);
      }
    }
  }
}

}  // namespace ynn
