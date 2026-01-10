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
    
    Uses PyWavelets iswt2 internally for correct reconstruction.
    This is a reference implementation - can be optimized for GPU later.
    
    Matches the dilation pattern of swavedec2.
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
    
    # Track if input was 2D (will have batch dim of 1 added by swavedec2)
    squeeze_batch = res_ll.dim() == 3 and res_ll.shape[0] == 1

    if res_ll.dim() >= 4:
        # avoid the channel sum, fold the channels into batches.
        coeffs, ds = _waverec2d_fold_channels_2d_list(coeffs)
        res_ll = _check_if_tensor(coeffs[0])
        squeeze_batch = False  # Don't squeeze for 4D input

    if not _is_dtype_supported(torch_dtype):
        raise ValueError(f"Input dtype {torch_dtype} not supported")

    # Convert wavelet to string for pywt if it's a ptwt.Wavelet object
    wavelet_name = wavelet if isinstance(wavelet, str) else wavelet_obj.name
    
    # Handle batched tensors
    # res_ll shape could be (H, W) or (B, H, W)
    has_batch = res_ll.dim() == 3
    
    if has_batch:
        batch_size = res_ll.shape[0]
        results = []
        
        for b in range(batch_size):
            # Convert coeffs to pywt format for this batch element
            # pywt format: [(LL_n, (LH_n, HL_n, HH_n)), ..., (LL_1, (LH_1, HL_1, HH_1))]
            # where level n is coarsest and level 1 is finest
            
            num_levels = len(coeffs) - 1
            pywt_coeffs = []
            
            # Our coeffs format: (LL, (LH, HL, HH)_coarsest, ..., (LH, HL, HH)_finest)
            # coeffs[0] = LL (approximation at coarsest level)
            # coeffs[1] = (LH, HL, HH) at coarsest level
            # coeffs[-1] = (LH, HL, HH) at finest level
            
            ll_np = coeffs[0][b].cpu().numpy()
            
            for level_idx in range(num_levels):
                detail_tuple = coeffs[1 + level_idx]  # coeffs[1] is coarsest detail
                # Our format: (LH, HL, HH) - same as pywt
                lh_np = detail_tuple[0][b].cpu().numpy()
                hl_np = detail_tuple[1][b].cpu().numpy()
                hh_np = detail_tuple[2][b].cpu().numpy()
                
                if level_idx == 0:
                    # Coarsest level - include LL
                    pywt_coeffs.append((ll_np, (lh_np, hl_np, hh_np)))
                else:
                    # Other levels - LL is computed from previous level reconstruction
                    # pywt expects None or can be omitted for non-coarsest levels
                    pywt_coeffs.append((None, (lh_np, hl_np, hh_np)))
            
            # Reconstruct using pywt
            rec_np = pywt.iswt2(pywt_coeffs, wavelet_name)
            results.append(torch.from_numpy(rec_np).to(torch_device).to(torch_dtype))
        
        output = torch.stack(results, dim=0)
    else:
        # No batch dimension
        ll_np = res_ll.cpu().numpy()
        num_levels = len(coeffs) - 1
        
        pywt_coeffs = []
        for level_idx in range(num_levels):
            detail_tuple = coeffs[1 + level_idx]
            lh_np = detail_tuple[0].cpu().numpy()
            hl_np = detail_tuple[1].cpu().numpy()
            hh_np = detail_tuple[2].cpu().numpy()
            
            if level_idx == 0:
                pywt_coeffs.append((ll_np, (lh_np, hl_np, hh_np)))
            else:
                pywt_coeffs.append((None, (lh_np, hl_np, hh_np)))
        
        rec_np = pywt.iswt2(pywt_coeffs, wavelet_name)
        output = torch.from_numpy(rec_np).to(torch_device).to(torch_dtype)

    if ds:
        output = _unfold_axes(output, list(ds), 2)

    if axes != (-2, -1):
        output = _undo_swap_axes(output, list(axes))

    return output