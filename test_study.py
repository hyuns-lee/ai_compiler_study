# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import pytest
import torch
from typing import Callable, Dict, Tuple, Union
from transformer_engine.pytorch.attention import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)
import triton
import triton.language as tl
import inspect
# print(inspect.getfile(RotaryPositionEmbedding))

def apply_rotary_pos_emb_thd(
    t: torch.Tensor, cu_seqlens: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    return torch.cat(
        [
            apply_rotary_pos_emb(x.unsqueeze(1), freqs[: x.size(0)])
            for x in torch.split(t, seqlens)
        ]
    ).squeeze(1)


def get_tol(dtype: torch.dtype) -> Dict:
    if dtype == torch.bfloat16:
        return dict(atol=1e-2, rtol=1e-2)
    elif dtype == torch.float16:
        return dict(atol=1e-3, rtol=1e-3)
    return dict(atol=1e-5, rtol=1.3e-6)


# Gradient is a broadcasted scalar
def _overlapping_grad(output: torch.Tensor) -> torch.Tensor:
    return output.sum() * 2

# Gradient is a full tensor
def _non_overlapping_grad(output: torch.Tensor) -> torch.Tensor:
    t = torch.ones_like(output)
    return torch.sum(output * t)

@triton.jit
def rope_fwd2(in_ptr, freq_ptr, out_ptr,
            dim, f_dim,
            stride_h, stride_d,    # stride direction
            BLOCK_SIZE: tl.constexpr):
    # f_dim = dim * roraty_percent
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    in_ptr += row * stride_d
    freq_ptr += row * stride_d
    
    _sin = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _cos = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    t = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    #t_rotate = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    #t_pass = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, f_dim, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        freq = tl.load(freq_ptr + cols, mask=cols < f_dim, other=0.).to(tl.float32)
        _sin, _cos = tl.sin(freq), tl.cos(freq)
        t = tl.load(in_ptr + cols, mask=cols < f_dim, other=0.).to(tl.float32)
        t_rotate = tl.where(cols < f_dim//2, -tl.load(in_ptr + f_dim//2 + cols), tl.load(in_ptr - f_dim//2 + cols))
        # Compute and store the result
        tl.store(out_ptr + cols, t*_cos + t_rotate*_sin, mask = cols < f_dim)
    
    for off in range(0, dim-f_dim, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        t_pass = tl.load(in_ptr + f_dim + cols, mask=cols < dim-f_dim, other=0.).to(tl.float32)
        tl.store(out_ptr + f_dim + cols, t_pass, mask = cols < dim)

@triton.jit
def rope_fwd(t_ptr, emb_ptr, out_ptr, seq_length, batch_size, head_num, hidden_size, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0)

    t_start = pid * hidden_size
    emb_start = pid//(batch_size*head_num) * BLOCK_SIZE

    emboffset = emb_start + tl.arange(0, BLOCK_SIZE//2)
    maskemb = emboffset < seq_length*BLOCK_SIZE
    emb = tl.load(emb_ptr + emboffset, mask=maskemb)
    cos_emb = tl.cos(emb)
    sin_emb = tl.sin(emb)

    t_left_offset = t_start + tl.arange(0, BLOCK_SIZE//2)
    t_right_offset = t_start + tl.arange(0, BLOCK_SIZE//2) + BLOCK_SIZE//2

    maskl = t_left_offset < seq_length*batch_size*head_num*hidden_size
    t_left = tl.load(t_ptr + t_left_offset, mask=maskl)
    maskr = t_right_offset < seq_length*batch_size*head_num*hidden_size
    t_right = tl.load(t_ptr + t_right_offset, mask=maskr)

    output = t_left * cos_emb - t_right * sin_emb
    tl.store(out_ptr+t_left_offset, output, mask = maskl)

    output =  t_left * sin_emb + t_right * cos_emb
    tl.store(out_ptr+t_right_offset, output, mask = maskr)

@triton.jit
def rope_bwd(t_ptr, emb_ptr, out_ptr, seq_length, batch_size, head_num, hidden_size, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0)

    t_start = pid * hidden_size
    emb_start = pid//(batch_size*head_num) * BLOCK_SIZE

    emboffset = emb_start + tl.arange(0, BLOCK_SIZE//2)
    maskemb = emboffset < seq_length*BLOCK_SIZE
    emb = tl.load(emb_ptr + emboffset, mask=maskemb)
    cos_emb = tl.cos(emb)
    sin_emb = tl.sin(emb)

    t_left_offset = t_start + tl.arange(0, BLOCK_SIZE//2)
    t_right_offset = t_start + tl.arange(0, BLOCK_SIZE//2) + BLOCK_SIZE//2

    maskl = t_left_offset < seq_length*batch_size*head_num*hidden_size
    t_left = tl.load(t_ptr + t_left_offset, mask=maskl)
    maskr = t_right_offset < seq_length*batch_size*head_num*hidden_size
    t_right = tl.load(t_ptr + t_right_offset, mask=maskr)

    output = t_left * cos_emb + t_right * sin_emb
    tl.store(out_ptr+t_left_offset, output, mask = maskl)

    output =  -t_left * sin_emb + t_right * cos_emb
    tl.store(out_ptr+t_right_offset, output, mask = maskr)
# @triton.jit
# def rope_bwd(t_ptr, emb_ptr, out_ptr, seq_length, batch_size, head_num, hidden_size, BLOCK_SIZE:tl.constexpr):
#     pid = tl.program_id(axis=0)

#     t_start = pid * hidden_size
#     emb_start = pid//(batch_size*head_num) * BLOCK_SIZE

#     emb_off = emb_start + tl.arange(0, BLOCK_SIZE//2)
#     emb = tl.load(emb_ptr + emb_off, mask = emb_off < seq_length*BLOCK_SIZE)
#     _cos, _sin = tl.cos(emb), tl.sin(emb)

#     t_off0 = t_start + tl.arange(0, BLOCK_SIZE//2)
#     t_off1 = t_start + tl.arange(0, BLOCK_SIZE//2) + BLOCK_SIZE//2
#     mask_t0 = t_off0 < (seq_length * batch_size * head_num * hidden_size)
#     mask_t1 = t_off1 < (seq_length * batch_size * head_num * hidden_size)
    
#     t0 = tl.load(t_ptr + t_off0, mask = mask_t0)
#     t1 = tl.load(t_ptr + t_off1, mask = mask_t1)
    
#     tl.store(out_ptr + t_off0, t0*_cos + t1*_sin, mask = mask_t0)
#     tl.store(out_ptr + t_off1, t1*_cos - t0*_sin, mask = mask_t1)

class TritonRoPEFunc(torch.autograd.Function):
    """
    Function for FusedRoPE

    This implementation assumes the input tensor to be in `sbhd`, `bshd` or `thd` format and
    the RoPE tensor to be of shape (s, 1, 1, d). It accepts arbitrary memory layouts to avoid
    the expensive `.contiguous()` calls, thus it may not achieve the best memory access pattern.
    """


    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
        cu_seqlens: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if tensor_format == "sbhd":
            output = t.clone().detach()
            seq_length = t.size()[0]
            batch_size = t.size()[1]
            head_num = t.size()[2]
            hidden_size = t.size()[3]
            rot_dim = freqs.size()[-1]
            grid = lambda meta: (triton.cdiv(seq_length*batch_size*head_num*hidden_size, meta['BLOCK_SIZE']),)
            rope_fwd[grid](t, freqs, output, seq_length, batch_size, head_num, hidden_size, BLOCK_SIZE=rot_dim)
        # elif tensor_format == "bshd":
        #     output = tex.fused_rope_forward(
        #         t.transpose(0, 1), freqs, True
        #     ).transpose(0, 1)
        # elif tensor_format == "thd":
        #     output = tex.fused_rope_thd_forward(t, cu_seqlens, freqs)
        else:
            raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
        ctx.save_for_backward(freqs, cu_seqlens)
        ctx.tensor_format = tensor_format

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        freqs, cu_seqlens = ctx.saved_tensors
        if ctx.tensor_format == "sbhd":
            grad_input = grad_output.clone().detach()
            seq_length = grad_output.size()[0]
            batch_size = grad_output.size()[1]
            head_num = grad_output.size()[2]
            hidden_size = grad_output.size()[3]
            rot_dim = freqs.size()[-1]
            grad_output = grad_output.contiguous()
            grid = lambda meta: (triton.cdiv(seq_length*batch_size*head_num*hidden_size, meta['BLOCK_SIZE']),)
            rope_bwd[grid](grad_output, freqs, grad_input, seq_length, batch_size, head_num, hidden_size, BLOCK_SIZE=rot_dim)
            # grad_input = tex.fused_rope_backward(grad_output, freqs, False)
        # elif ctx.tensor_format == "bshd":
        #     grad_input = tex.fused_rope_backward(
        #         grad_output.transpose(0, 1), freqs, True
        #     ).transpose(0, 1)
        # elif ctx.tensor_format == "thd":
        #     grad_input = tex.fused_rope_thd_backward(grad_output, cu_seqlens, freqs)
        # else:
        #     raise ValueError(f"Unsupported tensor_format: {ctx.tensor_format}.")

        return grad_input, None, None, None, None
        # return None, None, None, None, None

# @triton.jit
# def rope_fwd(t_ptr, emb_ptr, out_ptr, seq_length, batch_size, head_num, hidden_size, BLOCK_SIZE:tl.constexpr):
#     pid = tl.program_id(axis=0)

#     t_start = pid * hidden_size
#     emb_start = pid//(batch_size*head_num) * BLOCK_SIZE

#     emboffset = emb_start + tl.arange(0, BLOCK_SIZE//2)
#     maskemb = emboffset < seq_length*BLOCK_SIZE
#     emb = tl.load(emb_ptr + emboffset, mask=maskemb)
#     cos_emb = tl.cos(emb)
#     sin_emb = tl.sin(emb)

#     t_left_offset = t_start + tl.arange(0, BLOCK_SIZE//2)
#     t_right_offset = t_start + tl.arange(0, BLOCK_SIZE//2) + BLOCK_SIZE//2

#     maskl = t_left_offset < seq_length*batch_size*head_num*hidden_size
#     t_left = tl.load(t_ptr + t_left_offset, mask=maskl)
#     maskr = t_right_offset < seq_length*batch_size*head_num*hidden_size
#     t_right = tl.load(t_ptr + t_right_offset, mask=maskr)

#     output = t_left * cos_emb - t_right * sin_emb
#     tl.store(out_ptr+t_left_offset, output, mask = maskl)

#     output = t_right * cos_emb + t_left * sin_emb
#     tl.store(out_ptr+t_right_offset, output, mask = maskr)





@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_length", [2048, 4096])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("rotary_percent", [0.5, 1.0])
@pytest.mark.parametrize("margin", [0, 10])
@pytest.mark.parametrize("transpose", [None, (0, 1), (2, 3)])
@pytest.mark.parametrize("tensor_format", ["sbhd", "bshd"])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
def test_fused_rope(
    dtype: torch.dtype,
    seq_length: int,
    hidden_size: int,
    rotary_percent: float,
    margin: int,
    transpose: Union[Tuple, None],
    tensor_format: str,
    loss_func: Callable,
) -> None:
    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64
    t = torch.rand(
        (seq_length - margin, batch_size, head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    # sbhd
    # t: [4096, 2, 64, 256]
    print(t.size())
    if tensor_format == "bshd":
        t = t.transpose(0, 1).contiguous()
    if transpose:
        t = t.transpose(*transpose).contiguous().transpose(*transpose)
    t.requires_grad = True

    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent)
    emb = rotary_pos_emb(seq_length)
    # emb: [4096, 1, 1, 128]
    print(emb.size())
    # print(emb[1])
    rot_dim = emb.size()[-1]
    # unfused
    # t: [4096, 2, 64, 256]
    # emb: [4096, 1, 1, 128]
    # tensor_format: "sbhd"
    # fused: False
    output_unfused = apply_rotary_pos_emb(
        t, emb, tensor_format=tensor_format, fused=False
    )

    ## Triton Fwd
    print('############### Triton Fwd ###############')
    # device = torch.device("cuda:0")
    # BLOCK_SIZE = 128
    # batch_size, head_num, hidden_size = 2, 64, 4
    # t = torch.rand((1, 1, 1, hidden_size), device=device)
    # rotary_pos_emb = RotaryPositionEmbedding(hidden_size, 1.0)
    # emb = rotary_pos_emb(1)
    # out = torch.empty_like(t)

    ## Triton: funtion version
    # out = t.clone().detach()
    # grid = lambda meta: (triton.cdiv(seq_length*batch_size*head_num*hidden_size, meta['BLOCK_SIZE']),)
    # rope_fwd[grid](t, emb, out, seq_length, batch_size, head_num, hidden_size, BLOCK_SIZE=rot_dim)
    # # print(out)
    # torch.testing.assert_close(output_unfused, out, **get_tol(dtype))

    


    ## Test
    # unfused
    output_unfused = apply_rotary_pos_emb(
        t, emb, tensor_format=tensor_format, fused=False
    )
    loss_unfused = loss_func(output_unfused)
    loss_unfused.backward()
    grad_unfused = t.grad.detach().clone()
    t.grad = None

    # Triton
    out_class_fwd = TritonRoPEFunc.apply(t, emb, tensor_format, None)
    loss_triton = loss_func(out_class_fwd)
    loss_triton.backward()
    grad_triton = t.grad.detach().clone()
    t.grad = None

    torch.testing.assert_close(output_unfused, out_class_fwd, **get_tol(dtype))

    torch.testing.assert_close(out_class_fwd, output_unfused, **get_tol(dtype))
    torch.testing.assert_close(grad_triton, grad_unfused, **get_tol(dtype))
    assert grad_triton.is_contiguous()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("rotary_percent", [0.5, 1.0])
@pytest.mark.parametrize("transpose", [None, (1, 2)])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
def test_fused_rope_thd(
    dtype: torch.dtype,
    hidden_size: int,
    rotary_percent: float,
    transpose: Union[Tuple, None],
    loss_func: Callable,
) -> None:
    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64
    cu_seqlens = torch.tensor(
        [0, 400, 542, 711, 727, 752, 1270, 1426, 1450, 1954, 2044, 2048],
        dtype=torch.int32,
        device=device,
    )
    t = torch.rand(
        (cu_seqlens[-1], head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    if transpose:
        t = t.transpose(*transpose).contiguous().transpose(*transpose)
    t.requires_grad = True

    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent)
    emb = rotary_pos_emb(cu_seqlens[-1])

    # unfused
    output_unfused = apply_rotary_pos_emb_thd(t, cu_seqlens, emb)
    loss_unfused = loss_func(output_unfused)
    loss_unfused.backward()
    grad_unfused = t.grad.detach().clone()
    t.grad = None

    # fused
    output_fused = apply_rotary_pos_emb(
        t, emb, fused=True, tensor_format="thd", cu_seqlens=cu_seqlens
    )
    loss_fused = loss_func(output_fused)
    loss_fused.backward()
    grad_fused = t.grad.detach().clone()
    t.grad = None

    torch.testing.assert_close(output_fused, output_unfused, **get_tol(dtype))
    torch.testing.assert_close(grad_fused, grad_unfused, **get_tol(dtype))




if __name__=="__main__":
    dtype = torch.float16
    seq_length = 4096
    hidden_size = 256
    rotary_percent = 0.5
    margin = 0
    transpose = None
    tensor_format = "sbhd"
    loss_func = _overlapping_grad

    test_fused_rope(dtype, seq_length, hidden_size, rotary_percent, margin, transpose, tensor_format, loss_func)