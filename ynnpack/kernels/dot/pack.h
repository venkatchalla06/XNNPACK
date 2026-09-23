#ifndef XNNPACK_YNNPACK_KERNELS_DOT_PACK_H_
#define XNNPACK_YNNPACK_KERNELS_DOT_PACK_H_

#include <cassert>
#include <cstddef>
#include <cstring>

#include "ynnpack/kernels/transpose/interleave.h"
#include "ynnpack/kernels/transpose/transpose.h"

namespace ynn {

// Packing is the following reshape + transpose operation:
//
// packed(mi, ni, mo, no) = input(mo * tile_m + mi, no * tile_n + ni)
//
// Where the input is padded with zeros where it is not aligned to a multiple of
// tile_m, tile_n.
class packer {
 public:
  // Prepare to run a packing operation for the given packing parameters.
  // If `transpose` is true, the input is transposed prior to the above reshape
  // and transpose operation.
  // `tile_m` must be a power of 2.
  packer(bool transpose, size_t elem_size_bits, size_t tile_m, size_t tile_n);

  // Run the packing operation for input and output buffers. The input has an
  // un-transposed shape of `m` x `n`. The output will be rounded up to a
  // multiple of the tile size.
  void pack(size_t m, size_t n, size_t input_stride, const void* input,
            size_t output_stride, size_t output_block_stride, void* output);

 protected:
  size_t elem_size_bits;
  size_t tile_m;
  size_t tile_n;
  interleave_kernel_fn interleave_fn = nullptr;
  transpose_kernel_fn transpose_fn = nullptr;
  ynn::transpose_fn transpose_blocks_fn = nullptr;
};

// Pack matrix A of shape `m` x `k` into 2D tiles of shape `block_m` x `tile_k`:
//
// packed(ki, mi, ko, mo) = input((mo * block_m + mi) * input_stride +
//                                (ko * tile_k + ki) * elem_size)
//
// Where the input is padded with zeros where `mo * block_m + mi >= m` or
// `ko * tile_k + ki >= k`. If `output_m_blocks` or `output_k_blocks` are 0,
// they default to `ceil_div(m, block_m)` and `ceil_div(k, tile_k)`.
void pack_a(size_t m, size_t k, size_t block_m, size_t tile_k, size_t elem_size,
            size_t input_stride, const void* input, size_t output_ko_stride,
            size_t output_mo_stride, void* output, size_t output_m_blocks = 0,
            size_t output_k_blocks = 0);

class packer_a {
 public:
  packer_a(size_t elem_size_bits, size_t block_m, size_t tile_k)
      : elem_size_(elem_size_bits / 8), block_m_(block_m), tile_k_(tile_k) {
    assert(elem_size_bits % 8 == 0);
  }

  void pack(size_t m, size_t k, size_t input_stride, const void* input,
            size_t output_ko_stride, size_t output_mo_stride, void* output,
            size_t num_block_m = 0, size_t num_block_k = 0) const {
    pack_a(m, k, block_m_, tile_k_, elem_size_, input_stride, input,
           output_ko_stride, output_mo_stride, output, num_block_m,
           num_block_k);
  }

 private:
  size_t elem_size_;
  size_t block_m_;
  size_t tile_k_;
};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_DOT_PACK_H_
