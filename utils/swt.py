import torch
import pywt
import torch.nn.functional as F

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

    if data.dtype not in (torch.float32, torch.float64, torch.float16):
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
    
    Fully differentiable PyTorch implementation.
    
    For SWT, the inverse is computed by applying synthesis filter contributions
    at each position and averaging over filter shifts. This is the dual operation
    to the à trous forward transform.
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

    if torch_dtype not in (torch.float32, torch.float64, torch.float16):
         if not _is_dtype_supported(torch_dtype):
            raise ValueError(f"Input dtype {torch_dtype} not supported")

    # Get reconstruction filters
    # Ensure filters are created with the correct dtype
    _, _, rec_lo, rec_hi = _get_filter_tensors(
        wavelet_obj, flip=False, device=torch_device, dtype=torch_dtype
    )
    filt_len = rec_lo.shape[-1]
    
    num_levels = len(coeffs) - 1  # Number of detail coefficient tuples
    
    # Ensure batch dimension: (B, H, W)
    if res_ll.dim() == 2:
        res_ll = res_ll.unsqueeze(0)
        coeffs = (res_ll,) + tuple(
            (c[0].unsqueeze(0), c[1].unsqueeze(0), c[2].unsqueeze(0)) 
            for c in coeffs[1:]
        )
    
    output = res_ll
    
    # Process from coarsest to finest level
    # coeffs[1] is coarsest detail, coeffs[-1] is finest detail
    for level_idx in range(num_levels):
        detail_tuple = coeffs[1 + level_idx]
        res_lh, res_hl, res_hh = detail_tuple
        
        # Dilation for this level: coarsest has highest dilation
        # level_idx=0 (coarsest) -> dilation = 2^(num_levels-1)
        # level_idx=num_levels-1 (finest) -> dilation = 2^0 = 1
        dilation = 2 ** (num_levels - 1 - level_idx)
        
        # Reconstruct using separable 1D iSWT
        # For 2D separable: first do rows, then cols (or vice versa)
        # Each 1D iSWT combines shifted filter contributions
        
        output = _iswt2_level(
            output, res_lh, res_hl, res_hh,
            rec_lo.squeeze(), rec_hi.squeeze(),
            dilation
        )

    if ds:
        output = _unfold_axes(output, list(ds), 2)

    if axes != (-2, -1):
        output = _undo_swap_axes(output, list(axes))

    return output


def _iswt2_level(
    ll: torch.Tensor, 
    lh: torch.Tensor, 
    hl: torch.Tensor, 
    hh: torch.Tensor,
    rec_lo: torch.Tensor,
    rec_hi: torch.Tensor,
    dilation: int
) -> torch.Tensor:
    """Reconstruct one level of 2D iSWT using separable convolutions.
    
    Args:
        ll: Low-low subband (B, H, W)
        lh: Low-high subband (B, H, W) 
        hl: High-low subband (B, H, W)
        hh: High-high subband (B, H, W)
        rec_lo: 1D low-pass reconstruction filter
        rec_hi: 1D high-pass reconstruction filter
        dilation: Filter dilation for this level
        
    Returns:
        Reconstructed tensor (B, H, W)
    """
    filt_len = rec_lo.shape[0]
    
    # Ensure 4D for conv2d: (B, C=1, H, W)
    if ll.dim() == 3:
        ll = ll.unsqueeze(1)
        lh = lh.unsqueeze(1)
        hl = hl.unsqueeze(1)
        hh = hh.unsqueeze(1)
        
    # -----------------------------------------------------------
    # Vertical Pass (Columns)
    # -----------------------------------------------------------
    # Compute:
    # L_col = ConvCol(ll, Lo) + ConvCol(lh, Hi)
    # H_col = ConvCol(hl, Lo) + ConvCol(hh, Hi)
    #
    # We batch this using groups=2.
    # Input: (B, 4, H, W) -> concatenated [ll, lh, hl, hh]
    # Groups: 
    #   Group1: [ll, lh] -> filter [Lo, Hi] -> L_col
    #   Group2: [hl, hh] -> filter [Lo, Hi] -> H_col
    
    inp_col = torch.cat([ll, lh, hl, hh], dim=1)
    
    # Prepare filters for vertical conv (H, 1)
    lo_col = rec_lo.view(1, 1, filt_len, 1)
    hi_col = rec_hi.view(1, 1, filt_len, 1)
    
    # Core filter pair [Lo, Hi] of shape (1, 2, filt_len, 1)
    filt_pair_col = torch.cat([lo_col, hi_col], dim=1)
    
    # Stack for groups=2: (2, 2, filt_len, 1)
    filters_col = torch.cat([filt_pair_col, filt_pair_col], dim=0)
    
    # Circular padding for vertical dimension (top pad)
    # Operations look back: out[i] depends on in[i - k*d]
    pad_amt = (filt_len - 1) * dilation
    inp_padded = F.pad(inp_col, (0, 0, pad_amt, 0), mode='circular')
    
    # Convolution
    res_cols = F.conv2d(inp_padded, filters_col, groups=2, dilation=(dilation, 1))
    
    # Normalize
    res_cols = res_cols / filt_len
    
    # res_cols is (B, 2, H, W) containing [L_col, H_col]
    
    # -----------------------------------------------------------
    # Horizontal Pass (Rows)
    # -----------------------------------------------------------
    # Compute:
    # Output = ConvRow(L_col, Lo) + ConvRow(H_col, Hi)
    #
    # Input: res_cols (B, 2, H, W)
    # Filter: [Lo, Hi] of shape (1, 2, 1, filt_len)
    
    lo_row = rec_lo.view(1, 1, 1, filt_len)
    hi_row = rec_hi.view(1, 1, 1, filt_len)
    filters_row = torch.cat([lo_row, hi_row], dim=1)
    
    # Circular padding for horizontal dimension (left pad)
    inp_row_padded = F.pad(res_cols, (pad_amt, 0, 0, 0), mode='circular')
    
    output = F.conv2d(inp_row_padded, filters_row, dilation=(1, dilation))
    
    # Normalize
    output = output / filt_len
    
    return output.squeeze(1)