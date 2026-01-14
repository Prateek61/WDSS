import torch
import torch.nn.functional as F
import pywt

import ptwt
from ptwt.constants import BoundaryMode, WaveletCoeff2d, WaveletDetailTuple2d
from ptwt.conv_transform_2 import (
    _is_dtype_supported,
    _swap_axes,
    _as_wavelet,
    _preprocess_tensor_dec2d,
    _get_filter_tensors,
    _construct_2d_filt,
    _translate_boundary_strings,
    partial,
    _unfold_axes,
    _map_result,
    _undo_swap_axes,
    _check_axes_argument,
    _check_if_tensor,
    _waverec2d_fold_channels_2d_list
)

from typing import Union, Optional, Tuple


def _circular_pad_2d(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    """Circular pad for 2D (last two dimensions), handles cases where padding > input size.
    
    Args:
        x: Input tensor of shape (..., H, W)
        pad: Tuple of (left, right, top, bottom) padding
        
    Returns:
        Padded tensor
    """
    # pad order: (left, right, top, bottom)
    padl, padr, padt, padb = pad
    h, w = x.shape[-2], x.shape[-1]
    
    # For circular padding in PyTorch, we need to use pad with (left, right, top, bottom)
    # but circular mode requires we handle it properly
    
    # If padding is larger than the dimension, we need to tile first
    need_tile_w = (padl >= w) or (padr >= w)
    need_tile_h = (padt >= h) or (padb >= h)
    
    if need_tile_w or need_tile_h:
        # Determine how many times we need to tile
        tile_w = max(1, (max(padl, padr) // w) + 2) if need_tile_w else 1
        tile_h = max(1, (max(padt, padb) // h) + 2) if need_tile_h else 1
        
        # Tile the tensor
        # x shape: (..., H, W) 
        ndim = x.dim()
        repeat_shape = [1] * ndim
        repeat_shape[-1] = tile_w
        repeat_shape[-2] = tile_h
        x_tiled = x.repeat(*repeat_shape)
        
        # Calculate new dimensions
        new_h, new_w = x_tiled.shape[-2], x_tiled.shape[-1]
        
        # Calculate start and end indices
        # We want the center portion plus padding on each side
        start_h = (tile_h // 2) * h - padt
        end_h = start_h + h + padt + padb
        start_w = (tile_w // 2) * w - padl
        end_w = start_w + w + padl + padr
        
        return x_tiled[..., start_h:end_h, start_w:end_w]
    else:
        # Standard circular padding - PyTorch handles this
        # pad format for 2D: (left, right, top, bottom)
        return torch.nn.functional.pad(x, (padl, padr, padt, padb), mode='circular')

def swavedec2(
    data: torch.Tensor,
    wavelet: Union[ptwt.Wavelet, str],
    *,
    mode: BoundaryMode = "reflect",
    level: Optional[int] = None,
    axes: Tuple[int, int] = (-2, -1)
) -> ptwt.WaveletCoeff2d:
    r"""Run a two-dimensional stationary wavelet transform (algorithm à trous).
    
    The filter dilation increases at each level: level j uses dilation 2^j.
    Uses circular padding to minimize border artifacts.
    
    This implementation follows ptwt's 1D SWT approach extended to 2D.
    """

    if not _is_dtype_supported(data.dtype):
        raise ValueError(f"Input dtype {data.dtype} not supported")

    if tuple(axes) != (-2, -1):
        if len(axes) != 2:
            raise ValueError("2D transforms work with two axes.")
        else:
            data = _swap_axes(data, list(axes))

    wavelet = _as_wavelet(wavelet)
    data, ds = _preprocess_tensor_dec2d(data)
    dec_lo, dec_hi, _, _ = _get_filter_tensors(
        wavelet, flip=True, device=data.device, dtype=data.dtype
    )
    dec_filt = _construct_2d_filt(lo=dec_lo, hi=dec_hi)
    
    filt_len = dec_lo.shape[-1]

    if level is None:
        level = pywt.dwtn_max_level([data.shape[-1], data.shape[-2]], wavelet)

    result_lst: list[WaveletDetailTuple2d] = []
    res_ll = data
    for current_level in range(level):
        # À trous algorithm: dilation doubles at each level
        # Level 0: dilation=1, Level 1: dilation=2, Level 2: dilation=4, etc.
        dilation = 2 ** current_level
        
        # Calculate circular padding - same approach as ptwt's 1D SWT
        # Using circular padding to minimize border artifacts
        padl = dilation * (filt_len // 2 - 1)
        padr = dilation * (filt_len // 2)
        padt = dilation * (filt_len // 2 - 1)
        padb = dilation * (filt_len // 2)
        
        res_ll = _circular_pad_2d(res_ll, (padl, padr, padt, padb))
        res = torch.nn.functional.conv2d(res_ll, dec_filt, stride=1, dilation=dilation)
        res_ll, res_lh, res_hl, res_hh = torch.split(res, 1, 1)
        to_append = WaveletDetailTuple2d(
            res_lh.squeeze(1), res_hl.squeeze(1), res_hh.squeeze(1)
        )
        result_lst.append(to_append)

    result_lst.reverse()
    res_ll = res_ll.squeeze(1)
    result: WaveletCoeff2d = res_ll, *result_lst

    if ds:
        _unfold_axes2 = partial(_unfold_axes, ds=ds, keep_no=2)
        result = _map_result(result, _unfold_axes2)

    if axes != (-2, -1):
        undo_swap_fn = partial(_undo_swap_axes, axes=axes)
        result = _map_result(result, undo_swap_fn)

    return result

def swaverec2(
    coeffs: ptwt.WaveletCoeff2d,
    wavelet: Union[ptwt.Wavelet, str],
    axes: Tuple[int, int] = (-2, -1)
) -> torch.Tensor:
    r"""Run a two-dimensional inverse stationary wavelet transform.
    
    Efficient, fully differentiable PyTorch implementation using convolutions.
    Uses the correct iSWT algorithm with conv1d for speed.
    """

    if tuple(axes) != (-2, -1):
        if len(axes) != 2:
            raise ValueError("2D transforms work with two axes.")
        else:
            _check_axes_argument(list(axes))
            swap_fn = partial(_swap_axes, axes=list(axes))
            coeffs = _map_result(coeffs, swap_fn)

    ds = None
    wavelet_obj = _as_wavelet(wavelet)

    res_ll = _check_if_tensor(coeffs[0])
    torch_device = res_ll.device
    torch_dtype = res_ll.dtype

    if res_ll.dim() >= 4:
        coeffs, ds = _waverec2d_fold_channels_2d_list(coeffs)
        res_ll = _check_if_tensor(coeffs[0])

    if not _is_dtype_supported(torch_dtype):
        raise ValueError(f"Input dtype {torch_dtype} not supported")

    # Get reconstruction filters
    _, _, rec_lo, rec_hi = _get_filter_tensors(
        wavelet_obj, flip=False, device=torch_device, dtype=torch_dtype
    )
    filt_len = rec_lo.shape[-1]
    
    num_levels = len(coeffs) - 1
    
    # Ensure batch dimension: (B, H, W)
    if res_ll.dim() == 2:
        res_ll = res_ll.unsqueeze(0)
        coeffs = (res_ll,) + tuple(
            WaveletDetailTuple2d(c[0].unsqueeze(0), c[1].unsqueeze(0), c[2].unsqueeze(0)) 
            for c in coeffs[1:]
        )
    
    output = res_ll
    
    # Process from coarsest to finest level
    for level_idx in range(num_levels):
        detail_tuple = coeffs[1 + level_idx]
        res_lh, res_hl, res_hh = detail_tuple
        
        step_size = 2 ** (num_levels - 1 - level_idx)
        
        output = _iswt2_level_fast(
            output, res_lh, res_hl, res_hh,
            rec_lo.squeeze(), rec_hi.squeeze(),
            step_size
        )

    if ds:
        output = _unfold_axes(output, list(ds), 2)

    if axes != (-2, -1):
        output = _undo_swap_axes(output, list(axes))

    return output


def _idwt1d_conv(ca: torch.Tensor, cd: torch.Tensor, 
                 rec_lo: torch.Tensor, rec_hi: torch.Tensor) -> torch.Tensor:
    """Single-level 1D inverse DWT using convolution (fast, differentiable).
    
    Args:
        ca: Approximation coefficients (B, N) or (B, H, N)
        cd: Detail coefficients (B, N) or (B, H, N)
        rec_lo: Low-pass reconstruction filter (K,)
        rec_hi: High-pass reconstruction filter (K,)
        
    Returns:
        Reconstructed signal with doubled last dimension
    """
    filt_len = rec_lo.shape[0]
    input_len = ca.shape[-1]
    output_len = 2 * input_len
    
    # Store original shape for reshaping later
    orig_shape = ca.shape[:-1]
    
    # Flatten to (batch, length) for conv1d
    ca_flat = ca.reshape(-1, input_len)
    cd_flat = cd.reshape(-1, input_len)
    batch_size = ca_flat.shape[0]
    
    # Upsample by 2 (insert zeros)
    ca_up = torch.zeros(batch_size, output_len, dtype=ca.dtype, device=ca.device)
    cd_up = torch.zeros(batch_size, output_len, dtype=cd.dtype, device=cd.device)
    ca_up[:, ::2] = ca_flat
    cd_up[:, ::2] = cd_flat
    
    # Add channel dimension for conv1d: (B, 1, L)
    ca_up = ca_up.unsqueeze(1)
    cd_up = cd_up.unsqueeze(1)
    
    # Circular pad on left with filt_len-1 for periodic boundary convolution
    # Handle case where padding exceeds signal length by tiling
    pad_size = filt_len - 1
    if pad_size >= output_len:
        # Tile then slice for large padding
        n_tiles = (pad_size // output_len) + 2
        ca_up = ca_up.repeat(1, 1, n_tiles)[..., :output_len + pad_size]
        cd_up = cd_up.repeat(1, 1, n_tiles)[..., :output_len + pad_size]
        # Shift to get correct circular alignment
        ca_up = torch.roll(ca_up, -pad_size, dims=-1)[..., :output_len + pad_size]
        cd_up = torch.roll(cd_up, -pad_size, dims=-1)[..., :output_len + pad_size]
    else:
        ca_up = F.pad(ca_up, (pad_size, 0), mode='circular')
        cd_up = F.pad(cd_up, (pad_size, 0), mode='circular')
    
    # Create conv1d filters: flip for convolution
    rec_lo_filt = rec_lo.flip(0).view(1, 1, -1)
    rec_hi_filt = rec_hi.flip(0).view(1, 1, -1)
    
    # Apply convolution
    out_lo = F.conv1d(ca_up, rec_lo_filt)
    out_hi = F.conv1d(cd_up, rec_hi_filt)
    
    output = (out_lo + out_hi).squeeze(1)  # (B, output_len)
    
    # Apply phase correction: roll by -(filt_len//2 - 1) to match pywt
    phase_shift = -(filt_len // 2 - 1)
    output = torch.roll(output, shifts=phase_shift, dims=-1)
    
    # Reshape back to original batch structure
    output = output.reshape(*orig_shape, output_len)
    
    return output


def _iswt1d_fast(ca: torch.Tensor, cd: torch.Tensor,
                 rec_lo: torch.Tensor, rec_hi: torch.Tensor,
                 step_size: int, dim: int = -1) -> torch.Tensor:
    """Fast 1D inverse SWT using convolution-based idwt.
    
    Args:
        ca: Approximation coefficients
        cd: Detail coefficients  
        rec_lo: Low-pass reconstruction filter
        rec_hi: High-pass reconstruction filter
        step_size: Step size for this level
        dim: Dimension to operate on
        
    Returns:
        Reconstructed tensor
    """
    # Move target dimension to last position
    if dim == -2:
        ca = ca.transpose(-2, -1)
        cd = cd.transpose(-2, -1)
    
    size = ca.shape[-1]
    output = torch.empty_like(ca)
    
    # Process each starting position
    for first in range(step_size):
        indices = torch.arange(first, size, step_size, device=ca.device)
        
        even_indices = indices[0::2]
        odd_indices = indices[1::2]
        
        if len(even_indices) == 0 or len(odd_indices) == 0:
            output[..., indices] = ca[..., indices]
            continue
        
        # Extract and reconstruct
        ca_even = ca[..., even_indices]
        cd_even = cd[..., even_indices]
        ca_odd = ca[..., odd_indices]
        cd_odd = cd[..., odd_indices]
        
        x1 = _idwt1d_conv(ca_even, cd_even, rec_lo, rec_hi)
        x2 = _idwt1d_conv(ca_odd, cd_odd, rec_lo, rec_hi)
        
        # Circular shift x2 right by 1 and average
        x2 = torch.roll(x2, shifts=1, dims=-1)
        avg = (x1 + x2) * 0.5
        
        output[..., indices] = avg
    
    if dim == -2:
        output = output.transpose(-2, -1)
    
    return output


def _iswt2_level_fast(
    ll: torch.Tensor, 
    lh: torch.Tensor, 
    hl: torch.Tensor, 
    hh: torch.Tensor,
    rec_lo: torch.Tensor,
    rec_hi: torch.Tensor,
    step_size: int
) -> torch.Tensor:
    """Fast 2D iSWT level reconstruction using separable 1D convolutions."""
    # Apply along columns (dim -2)
    L_col = _iswt1d_fast(ll, lh, rec_lo, rec_hi, step_size, dim=-2)
    H_col = _iswt1d_fast(hl, hh, rec_lo, rec_hi, step_size, dim=-2)
    
    # Apply along rows (dim -1)
    output = _iswt1d_fast(L_col, H_col, rec_lo, rec_hi, step_size, dim=-1)
    
    return output