import torch
import torch.nn.functional as F
import pywt
import math

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

from typing import Union, Optional, Tuple, Dict

# Global filter cache to avoid recreating filters on every call
_FILTER_CACHE: Dict[tuple, torch.Tensor] = {}

# ============================================================================
# SPECIALIZED WAVELET CONSTANTS AND OPTIMIZED IMPLEMENTATIONS
# ============================================================================

# Haar wavelet coefficients (filter length 2)
_HAAR_COEF = 0.7071067811865476  # 1/sqrt(2)

# DB4 wavelet reconstruction filter coefficients (filter length 8)
_DB4_REC_LO = (0.2303778133088965, 0.7148465705529157, 0.6308807679298589, 
               -0.027983769416859854, -0.18703481171909309, 0.030841381835560764, 
               0.0328830116668852, -0.010597401785069032)
_DB4_REC_HI = (-0.010597401785069032, -0.0328830116668852, 0.030841381835560764, 
               0.18703481171909309, -0.027983769416859854, -0.6308807679298589, 
               0.7148465705529157, -0.2303778133088965)

# SYM4 wavelet reconstruction filter coefficients (filter length 8)
_SYM4_REC_LO = (0.0322231006040427, -0.012603967262037833, -0.09921954357684722,
                0.29785779560527736, 0.8037387518059161, 0.49761866763201545,
                -0.02963552764599851, -0.07576571478927333)
_SYM4_REC_HI = (-0.07576571478927333, 0.02963552764599851, 0.49761866763201545,
                -0.8037387518059161, 0.29785779560527736, 0.09921954357684722,
                -0.012603967262037833, -0.0322231006040427)

# Cached stacked reconstruction filters for conv_transpose1d, shape (2, 1, K)
_HAAR_FILT_CACHE: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}
_DB4_FILT_CACHE: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}
_SYM4_FILT_CACHE: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _get_haar_rec_filt(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Get cached Haar reconstruction filter stack, shape (2, 1, K)."""
    key = (device, dtype)
    if key not in _HAAR_FILT_CACHE:
        c = _HAAR_COEF
        lo = torch.tensor([[c, c]], device=device, dtype=dtype)  # (1, 2)
        hi = torch.tensor([[c, -c]], device=device, dtype=dtype)  # (1, 2)
        _HAAR_FILT_CACHE[key] = torch.stack([lo, hi], dim=0)  # (2, 1, 2)
    return _HAAR_FILT_CACHE[key]


def _get_db4_rec_filt(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Get cached DB4 reconstruction filter stack, shape (2, 1, K)."""
    key = (device, dtype)
    if key not in _DB4_FILT_CACHE:
        lo = torch.tensor([_DB4_REC_LO], device=device, dtype=dtype)  # (1, 8)
        hi = torch.tensor([_DB4_REC_HI], device=device, dtype=dtype)  # (1, 8)
        _DB4_FILT_CACHE[key] = torch.stack([lo, hi], dim=0)  # (2, 1, 8)
    return _DB4_FILT_CACHE[key]


def _get_sym4_rec_filt(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Get cached SYM4 reconstruction filter stack, shape (2, 1, K)."""
    key = (device, dtype)
    if key not in _SYM4_FILT_CACHE:
        lo = torch.tensor([_SYM4_REC_LO], device=device, dtype=dtype)  # (1, 8)
        hi = torch.tensor([_SYM4_REC_HI], device=device, dtype=dtype)  # (1, 8)
        _SYM4_FILT_CACHE[key] = torch.stack([lo, hi], dim=0)  # (2, 1, 8)
    return _SYM4_FILT_CACHE[key]


# ============================================================================
# FAST CONVOLUTION-BASED iSWT IMPLEMENTATION
# ============================================================================

def _iswt1d_conv(ca: torch.Tensor, cd: torch.Tensor,
                 rec_filt: torch.Tensor, filt_len: int, dilation: int = 1) -> torch.Tensor:
    """Fast 1D inverse SWT using conv_transpose1d with grouped convolution.
    
    Uses the algorithm from ptwt: stack approx and detail, apply conv_transpose1d
    with groups=2, then take mean across groups.
    
    Args:
        ca: Approximation coefficients (N, L)
        cd: Detail coefficients (N, L)
        rec_filt: Stacked reconstruction filters, shape (2, 1, K)
                  rec_filt[0] = rec_lo, rec_filt[1] = rec_hi
        filt_len: Filter length K
        dilation: Dilation factor (default 1 for level 1)
        
    Returns:
        Reconstructed signal (N, L)
    """
    N, L = ca.shape
    
    # Stack approx and detail: (N, 2, L)
    stacked = torch.stack([ca, cd], dim=1)
    
    # Circular padding
    padl = dilation * (filt_len // 2)
    padr = dilation * (filt_len // 2 - 1)
    stacked_pad = F.pad(stacked, (padl, padr), mode='circular')
    
    # conv_transpose1d with groups=2: each filter only sees its corresponding input channel
    # Input: (N, 2, L+pad), Filter: (2, 1, K), groups=2
    # Output: (N, 2, L)
    output = F.conv_transpose1d(stacked_pad, rec_filt, dilation=dilation, groups=2, padding=(padl + padr))
    
    # Average the two reconstructions
    return output.mean(dim=1)  # (N, L)


def _iswt2_conv(
    ll: torch.Tensor, 
    lh: torch.Tensor, 
    hl: torch.Tensor, 
    hh: torch.Tensor,
    rec_filt: torch.Tensor,
    filt_len: int,
    dilation: int = 1
) -> torch.Tensor:
    """Fast 2D iSWT level using separable 1D conv_transpose.
    
    For 2D separable reconstruction:
    1. Reconstruct columns: L = iswt1d(LL, LH), H = iswt1d(HL, HH)
    2. Reconstruct rows: out = iswt1d(L, H)
    
    Args:
        ll, lh, hl, hh: Wavelet subbands, shape (B, H, W)
        rec_filt: Stacked reconstruction filters, shape (2, 1, K)
        filt_len: Filter length K
        dilation: Dilation factor for multi-level
    """
    B, H, W = ll.shape
    
    # ========== COLUMNS (along H dimension) ==========
    # Reshape: (B, H, W) -> (B*W, H)
    ll_col = ll.permute(0, 2, 1).reshape(-1, H)
    lh_col = lh.permute(0, 2, 1).reshape(-1, H)
    hl_col = hl.permute(0, 2, 1).reshape(-1, H)
    hh_col = hh.permute(0, 2, 1).reshape(-1, H)
    
    L_col = _iswt1d_conv(ll_col, lh_col, rec_filt, filt_len, dilation)
    H_col = _iswt1d_conv(hl_col, hh_col, rec_filt, filt_len, dilation)
    
    # ========== ROWS (along W dimension) ==========
    # Reshape: (B*W, H) -> (B, W, H) -> (B, H, W) -> (B*H, W)
    L_row = L_col.view(B, W, H).permute(0, 2, 1).reshape(-1, W)
    H_row = H_col.view(B, W, H).permute(0, 2, 1).reshape(-1, W)
    
    output = _iswt1d_conv(L_row, H_row, rec_filt, filt_len, dilation)
    
    return output.view(B, H, W)


# ============================================================================
# WAVELET-SPECIFIC OPTIMIZED 2D iSWT FUNCTIONS
# ============================================================================

def _iswt2_haar(
    ll: torch.Tensor, 
    lh: torch.Tensor, 
    hl: torch.Tensor, 
    hh: torch.Tensor
) -> torch.Tensor:
    """Fast 2D iSWT for Haar wavelet using conv_transpose approach."""
    rec_filt = _get_haar_rec_filt(ll.device, ll.dtype)
    return _iswt2_conv(ll, lh, hl, hh, rec_filt, filt_len=2, dilation=1)


def _iswt2_db4(
    ll: torch.Tensor, 
    lh: torch.Tensor, 
    hl: torch.Tensor, 
    hh: torch.Tensor
) -> torch.Tensor:
    """Fast 2D iSWT for DB4 wavelet using conv_transpose approach."""
    rec_filt = _get_db4_rec_filt(ll.device, ll.dtype)
    return _iswt2_conv(ll, lh, hl, hh, rec_filt, filt_len=8, dilation=1)


def _iswt2_sym4(
    ll: torch.Tensor, 
    lh: torch.Tensor, 
    hl: torch.Tensor, 
    hh: torch.Tensor
) -> torch.Tensor:
    """Fast 2D iSWT for SYM4 wavelet using conv_transpose approach."""
    rec_filt = _get_sym4_rec_filt(ll.device, ll.dtype)
    return _iswt2_conv(ll, lh, hl, hh, rec_filt, filt_len=8, dilation=1)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
    
    Fast, fully differentiable PyTorch implementation.
    Uses convolution-based reconstruction that processes all levels from coarsest to finest.
    Optimized wavelet-specific implementations for haar, db4, sym4.
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
    wavelet_name = wavelet if isinstance(wavelet, str) else wavelet_obj.name

    res_ll = _check_if_tensor(coeffs[0])
    torch_device = res_ll.device
    torch_dtype = res_ll.dtype

    if res_ll.dim() >= 4:
        coeffs, ds = _waverec2d_fold_channels_2d_list(coeffs)
        res_ll = _check_if_tensor(coeffs[0])

    if torch_dtype not in (torch.float32, torch.float64, torch.float16):
         if not _is_dtype_supported(torch_dtype):
            raise ValueError(f"Input dtype {torch_dtype} not supported")

    # Get reconstruction filters and stack them for conv_transpose1d
    _, _, rec_lo, rec_hi = _get_filter_tensors(
        wavelet_obj, flip=False, device=torch_device, dtype=torch_dtype
    )
    filt_len = rec_lo.shape[-1]
    
    # Stack filters: (2, 1, K) for grouped conv_transpose1d
    rec_filt = torch.stack([rec_lo.squeeze().unsqueeze(0), 
                            rec_hi.squeeze().unsqueeze(0)], dim=0)  # (2, 1, K)
    
    num_levels = len(coeffs) - 1
    
    # Ensure batch dimension: (B, H, W)
    if res_ll.dim() == 2:
        res_ll = res_ll.unsqueeze(0)
        coeffs = (res_ll,) + tuple(
            WaveletDetailTuple2d(c[0].unsqueeze(0), c[1].unsqueeze(0), c[2].unsqueeze(0)) 
            for c in coeffs[1:]
        )
    
    output = res_ll
    
    # Process from coarsest to finest level (level_idx 0 = coarsest)
    for level_idx in range(num_levels):
        detail_tuple = coeffs[1 + level_idx]
        res_lh, res_hl, res_hh = detail_tuple
        
        # Dilation for this level: coarsest has highest dilation
        dilation = 2 ** (num_levels - 1 - level_idx)
        
        output = _iswt2_conv(
            output, res_lh, res_hl, res_hh,
            rec_filt, filt_len, dilation
        )

    if ds:
        output = _unfold_axes(output, list(ds), 2)

    if axes != (-2, -1):
        output = _undo_swap_axes(output, list(axes))

    return output