"""CuTe DSL kernels kept in a module so source inspection is reliable."""

from __future__ import annotations

import cutlass.cute as cute
import cuda.bindings.driver as cuda


@cute.kernel
def vector_add_kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    """Per-thread vector add: c[i] = a[i] + b[i]."""

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    frg_a = cute.make_rmem_tensor(cute.size(a, mode=[0]), a.element_type)
    frg_b = cute.make_rmem_tensor(cute.size(b, mode=[0]), b.element_type)
    frg_c = cute.make_rmem_tensor(cute.size(c, mode=[0]), c.element_type)
    cute.autovec_copy(a[None, tidx, bidx], frg_a)
    cute.autovec_copy(b[None, tidx, bidx], frg_b)
    frg_c.store(frg_a.load() + frg_b.load())
    cute.autovec_copy(frg_c, c[None, tidx, bidx])


@cute.jit
def launch_vector_add(
    stream: cuda.CUstream,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
):
    """Launch vector add with one logical element per thread."""

    vector_add_kernel(a, b, c).launch(
        grid=[a.shape[-1], 1, 1],
        block=[a.shape[-2], 1, 1],
        stream=stream,
    )

