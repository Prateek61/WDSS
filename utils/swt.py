import torch
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
    
    Fully differentiable PyTorch implementation using the correct iSWT algorithm.
    
    The algorithm follows pywt's iswt approach:
    - For each level, split coefficients into even/odd phases
    - Apply standard idwt (upsample + filter) to each phase
    - Shift and average the two phase results
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
        # avoid the channel sum, fold the channels into batches.
        coeffs, ds = _waverec2d_fold_channels_2d_list(coeffs)
        res_ll = _check_if_tensor(coeffs[0])

    if not _is_dtype_supported(torch_dtype):
        raise ValueError(f"Input dtype {torch_dtype} not supported")

    # Get reconstruction filters
    _, _, rec_lo, rec_hi = _get_filter_tensors(
        wavelet_obj, flip=False, device=torch_device, dtype=torch_dtype
    )
    filt_len = rec_lo.shape[-1]
    
    num_levels = len(coeffs) - 1  # Number of detail coefficient tuples
    
    # Ensure batch dimension: (B, H, W)
    if res_ll.dim() == 2:
        res_ll = res_ll.unsqueeze(0)
        coeffs = (res_ll,) + tuple(
            WaveletDetailTuple2d(c[0].unsqueeze(0), c[1].unsqueeze(0), c[2].unsqueeze(0)) 
            for c in coeffs[1:]
        )
    
    output = res_ll
    
    # Process from coarsest to finest level
    # coeffs[1] is coarsest detail, coeffs[-1] is finest detail
    for level_idx in range(num_levels):
        detail_tuple = coeffs[1 + level_idx]
        res_lh, res_hl, res_hh = detail_tuple
        
        # Step size for this level: coarsest has highest step_size
        # level_idx=0 (coarsest) -> step_size = 2^(num_levels-1)
        # level_idx=num_levels-1 (finest) -> step_size = 2^0 = 1
        step_size = 2 ** (num_levels - 1 - level_idx)
        
        # Reconstruct using separable 1D iSWT
        output = _iswt2_level(
            output, res_lh, res_hl, res_hh,
            rec_lo.squeeze(), rec_hi.squeeze(),
            step_size, filt_len
        )

    if ds:
        output = _unfold_axes(output, list(ds), 2)

    if axes != (-2, -1):
        output = _undo_swap_axes(output, list(axes))

    return output


def _idwt1d_single(ca: torch.Tensor, cd: torch.Tensor, 
                   rec_lo: torch.Tensor, rec_hi: torch.Tensor) -> torch.Tensor:
    """Single-level 1D inverse DWT (differentiable).
    
    Implements: upsample by 2, then convolve with reconstruction filters.
    Uses periodic boundary conditions. Matches pywt's idwt_single behavior.
    
    Args:
        ca: Approximation coefficients (..., N)
        cd: Detail coefficients (..., N)
        rec_lo: Low-pass reconstruction filter (K,)
        rec_hi: High-pass reconstruction filter (K,)
        
    Returns:
        Reconstructed signal (..., 2*N)
    """
    # Get dimensions
    input_len = ca.shape[-1]
    filt_len = rec_lo.shape[0]
    output_len = 2 * input_len
    
    # Upsample by 2 (insert zeros between samples)
    shape = list(ca.shape)
    shape[-1] = output_len
    
    ca_up = torch.zeros(shape, dtype=ca.dtype, device=ca.device)
    cd_up = torch.zeros(shape, dtype=cd.dtype, device=cd.device)
    
    # Even positions get the coefficients
    ca_up[..., ::2] = ca
    cd_up[..., ::2] = cd
    
    # Convolve with reconstruction filters using circular boundary
    # The phase offset matches pywt's idwt_single: offset = -(filt_len // 2 - 1)
    # This comes from the standard wavelet reconstruction phase alignment
    phase_offset = -(filt_len // 2 - 1)
    
    output = torch.zeros(shape, dtype=ca.dtype, device=ca.device)
    
    for k in range(filt_len):
        shift = k + phase_offset
        ca_shifted = torch.roll(ca_up, shifts=shift, dims=-1)
        cd_shifted = torch.roll(cd_up, shifts=shift, dims=-1)
        output = output + rec_lo[k] * ca_shifted + rec_hi[k] * cd_shifted
    
    return output


def _iswt1d(ca: torch.Tensor, cd: torch.Tensor,
            rec_lo: torch.Tensor, rec_hi: torch.Tensor,
            step_size: int, dim: int = -1) -> torch.Tensor:
    """1D inverse SWT along specified dimension (differentiable).
    
    Uses the correct iSWT algorithm from pywt:
    - Split coefficients into even/odd indexed subsets based on step_size
    - Apply idwt to each subset
    - Shift one result and average the two
    
    Args:
        ca: Approximation coefficients
        cd: Detail coefficients
        rec_lo: Low-pass reconstruction filter
        rec_hi: High-pass reconstruction filter
        step_size: Step size for this level (2^j for level j from coarsest)
        dim: Dimension to operate on (-1 for last, -2 for second to last)
        
    Returns:
        Reconstructed tensor
    """
    # Move target dimension to last position
    if dim == -2:
        ca = ca.transpose(-2, -1)
        cd = cd.transpose(-2, -1)
    
    size = ca.shape[-1]
    output = ca.clone()
    
    # Process each starting position from 0 to step_size-1
    for first in range(step_size):
        # Get indices for this phase: first, first+step_size, first+2*step_size, ...
        indices = torch.arange(first, size, step_size, device=ca.device)
        
        # Split into even and odd indexed positions within this subset
        even_indices = indices[0::2]
        odd_indices = indices[1::2]
        
        if len(even_indices) == 0 or len(odd_indices) == 0:
            continue
            
        # Extract coefficients at even and odd positions
        ca_even = ca[..., even_indices]
        cd_even = cd[..., even_indices]
        ca_odd = ca[..., odd_indices]
        cd_odd = cd[..., odd_indices]
        
        # Apply single-level idwt to each subset
        x1 = _idwt1d_single(ca_even, cd_even, rec_lo, rec_hi)
        x2 = _idwt1d_single(ca_odd, cd_odd, rec_lo, rec_hi)
        
        # Circular shift x2 right by 1
        x2 = torch.roll(x2, shifts=1, dims=-1)
        
        # Average and store back
        avg = (x1 + x2) / 2.0
        
        # Put back into output at the correct indices
        output[..., indices] = avg
    
    # Restore dimension order
    if dim == -2:
        output = output.transpose(-2, -1)
    
    return output


def _iswt2_level(
    ll: torch.Tensor, 
    lh: torch.Tensor, 
    hl: torch.Tensor, 
    hh: torch.Tensor,
    rec_lo: torch.Tensor,
    rec_hi: torch.Tensor,
    step_size: int,
    filt_len: int
) -> torch.Tensor:
    """Reconstruct one level of 2D iSWT using separable 1D operations.
    
    For 2D separable wavelet transform:
    - First apply 1D iSWT along columns (vertical direction)
    - Then apply 1D iSWT along rows (horizontal direction)
    
    Args:
        ll: Low-low subband (B, H, W)
        lh: Low-high subband (B, H, W) 
        hl: High-low subband (B, H, W)
        hh: High-high subband (B, H, W)
        rec_lo: 1D low-pass reconstruction filter
        rec_hi: 1D high-pass reconstruction filter
        step_size: Step size for this level
        filt_len: Filter length
        
    Returns:
        Reconstructed tensor (B, H, W)
    """
    # For 2D separable iSWT:
    # 1. Reconstruct columns: L = iswt1d_col(LL, LH), H = iswt1d_col(HL, HH)
    # 2. Reconstruct rows: out = iswt1d_row(L, H)
    
    # Apply along columns (dim -2)
    L_col = _iswt1d(ll, lh, rec_lo, rec_hi, step_size, dim=-2)
    H_col = _iswt1d(hl, hh, rec_lo, rec_hi, step_size, dim=-2)
    
    # Apply along rows (dim -1)
    output = _iswt1d(L_col, H_col, rec_lo, rec_hi, step_size, dim=-1)
    
    return output