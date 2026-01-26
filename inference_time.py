# inference_benchmark.py
from jupyter_notebooks.notebook_utils import *
import time
import numpy as np
import torch
import torch.nn.functional as F

# -----------------------------
# Global backend knobs (good defaults)
# -----------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


# -----------------------------
# Utilities
# -----------------------------
def ensure_4d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.ndim == 3 else x

def maybe_channels_last(x: torch.Tensor) -> torch.Tensor:
    # channels_last only valid for NCHW tensors
    return x.contiguous(memory_format=torch.channels_last) if x.ndim == 4 else x

def prepare_inputs(train, device, upscale=2, idx=2):
    """
    Loads one sample and returns lr_frame, hr_gbuffer, temporal, upscale_factor (int).
    """
    frame = train.get_item(idx, upscale_factor=upscale, no_patch=True)
    frame = WDSSDataset.batch_to_device(frame, device)

    lr_frame   = ensure_4d(frame[FrameGroup.LR_INP.value]).to(device, non_blocking=True)
    hr_gbuffer = ensure_4d(frame[FrameGroup.GB_INP.value]).to(device, non_blocking=True)
    temporal   = ensure_4d(frame[FrameGroup.TEMPORAL_INP.value]).to(device, non_blocking=True)
    gt         = frame[FrameGroup.GT.value].to(device, non_blocking=True)

    # FP16 for tensor cores (inputs)
    lr_frame   = maybe_channels_last(lr_frame).half()
    hr_gbuffer = maybe_channels_last(hr_gbuffer).half()
    temporal   = maybe_channels_last(temporal).half()

    upscale_factor = int(round(gt.shape[-2] / lr_frame.shape[-2]))
    return lr_frame, hr_gbuffer, temporal, upscale_factor


# -----------------------------
# Timers for profiling mode
# -----------------------------
class ModuleTimer:
    def __init__(self):
        self.starts = {}
        self.times = {}  # name -> list(ms)

    def pre_hook(self, name):
        def _pre(module, inputs):
            x = None
            if isinstance(inputs, (tuple, list)) and len(inputs) > 0 and torch.is_tensor(inputs[0]):
                x = inputs[0]
            if x is not None and x.is_cuda:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                self.starts[name] = ("cuda", s, e)
            else:
                self.starts[name] = ("cpu", time.perf_counter(), None)
        return _pre

    def post_hook(self, name):
        def _post(module, inputs, output):
            kind, a, b = self.starts[name]
            if kind == "cuda":
                s, e = a, b
                e.record()
                torch.cuda.synchronize()
                self.times.setdefault(name, []).append(float(s.elapsed_time(e)))
            else:
                t0 = a
                self.times.setdefault(name, []).append((time.perf_counter() - t0) * 1000.0)
        return _post

    @staticmethod
    def avg_ms(xs):
        return (sum(xs) / len(xs)) if xs else 0.0


class OpTimer:
    def __init__(self):
        self.starts = {}
        self.times = {}

    def start(self, key, x):
        if torch.is_tensor(x) and x.is_cuda:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            self.starts[key] = ("cuda", s, e)
        else:
            self.starts[key] = ("cpu", time.perf_counter(), None)

    def stop(self, key):
        kind, a, b = self.starts[key]
        if kind == "cuda":
            s, e = a, b
            e.record()
            torch.cuda.synchronize()
            # Guard in case something odd happens
            if not s.query() or not e.query():
                return
            self.times.setdefault(key, []).append(float(s.elapsed_time(e)))
        else:
            t0 = a
            self.times.setdefault(key, []).append((time.perf_counter() - t0) * 1000.0)

    def avg(self, key):
        xs = self.times.get(key, [])
        return (sum(xs) / len(xs)) if xs else 0.0


# -----------------------------
# Mode 1: Performance benchmark (compile + end-to-end)
# -----------------------------
def run_perf(settings_path="config/config.json", iters=100, warmup=20, dtype=torch.float16,
             compile_mode="max-autotune", upscale=2, idx=2):
    """
    Best throughput measurement:
    - no hooks
    - no monkey patches
    - torch.compile enabled
    - channels_last + AMP
    """
    settings = initialize_settings(settings_path)
    train, val, test = WDSSDataset.get_datasets(settings)

    device = torch.device("cuda")

    lr_frame, hr_gbuffer, temporal, upscale_factor = prepare_inputs(train, device, upscale=upscale, idx=idx)

    model = get_model(settings["model_config"]).to(device).eval().to(memory_format=torch.channels_last)

    # Compile (Linux + triton assumed working)
    model = torch.compile(model, mode=compile_mode)

    # Warmup
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        for _ in range(warmup):
            _ = model(lr_frame, hr_gbuffer, temporal, upscale_factor)
    torch.cuda.synchronize()

    # Timed
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    times_ms = []

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        for _ in range(iters):
            starter.record()
            _ = model(lr_frame, hr_gbuffer, temporal, upscale_factor)
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))

    print("\n=== PERFORMANCE MODE ===")
    print(f"compile_mode: {compile_mode} | dtype: {dtype} | iters: {iters} | warmup: {warmup}")
    print(f"mean: {np.mean(times_ms):.3f} ms | p50: {np.percentile(times_ms,50):.3f} ms | p90: {np.percentile(times_ms,90):.3f} ms")

    train.cleanup(); val.cleanup(); test.cleanup()
    return times_ms


# -----------------------------
# Mode 2: Profiling breakdown (no compile, per-module + optional ops)
# -----------------------------
def run_profile(settings_path="config/config.json", iters=50, warmup=10, dtype=torch.float16,
                profile_ops=True, upscale=2, idx=2):
    """
    Attribution/profiling:
    - no torch.compile
    - module hooks for major blocks
    - optional monkey-patches for pixel shuffle/unshuffle + wavelet IWT
    - many synchronizations (slow but informative)
    """
    settings = initialize_settings(settings_path)
    train, val, test = WDSSDataset.get_datasets(settings)

    device = torch.device("cuda")

    lr_frame, hr_gbuffer, temporal, upscale_factor = prepare_inputs(train, device, upscale=upscale, idx=idx)

    model = get_model(settings["model_config"]).to(device).eval().to(memory_format=torch.channels_last)

    # --- Hook timers
    module_timer = ModuleTimer()
    hooks = []

    # Top-level blocks
    direct_modules = {
        "lr_feat_extractor": model.lr_feat_extractor,
        "temporal_feat_extractor": model.temporal_feat_extractor,
        "hr_gb_feat_extractor": model.hr_gb_feat_extractor,
    }
    for name, m in direct_modules.items():
        hooks.append(m.register_forward_pre_hook(module_timer.pre_hook(name)))
        hooks.append(m.register_forward_hook(module_timer.post_hook(name)))

    # Children of fusion / fminr (fine-grained)
    def hook_all_children(root_module, prefix):
        for n, m in root_module.named_modules():
            if n == "":
                continue
            nm = f"{prefix}.{n}"
            hooks.append(m.register_forward_pre_hook(module_timer.pre_hook(nm)))
            hooks.append(m.register_forward_hook(module_timer.post_hook(nm)))

    hook_all_children(model.feature_fusion, "feature_fusion")
    hook_all_children(model.fminr, "fminr")

    # --- Op timers (optional)
    op_timer = OpTimer()
    patched = {"pixel": False, "iwt": False}

    if profile_ops:
        # Patch pixel shuffle/unshuffle
        _orig_pixel_unshuffle = F.pixel_unshuffle
        _orig_pixel_shuffle   = F.pixel_shuffle

        def timed_pixel_unshuffle(x, downscale_factor):
            op_timer.start("pixel_unshuffle", x)
            out = _orig_pixel_unshuffle(x, downscale_factor)
            op_timer.stop("pixel_unshuffle")
            return out

        def timed_pixel_shuffle(x, upscale_factor_):
            op_timer.start("pixel_shuffle", x)
            out = _orig_pixel_shuffle(x, upscale_factor_)
            op_timer.stop("pixel_shuffle")
            return out

        F.pixel_unshuffle = timed_pixel_unshuffle
        F.pixel_shuffle   = timed_pixel_shuffle
        patched["pixel"] = True

        # Patch wavelet IWT
        _orig_batch_iwt = WaveletProcessor.batch_iwt

        def timed_batch_iwt(x):
            # This will be called inside your model's FP32 block
            op_timer.start("inverse_wavelet_iwt", x)
            out = _orig_batch_iwt(x)
            op_timer.stop("inverse_wavelet_iwt")
            return out

        WaveletProcessor.batch_iwt = timed_batch_iwt
        patched["iwt"] = True

    # Warmup
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        for _ in range(warmup):
            _ = model(lr_frame, hr_gbuffer, temporal, upscale_factor)
    torch.cuda.synchronize()

    # Timed end-to-end (optional, still useful)
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    times_ms = []

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        for _ in range(iters):
            starter.record()
            _ = model(lr_frame, hr_gbuffer, temporal, upscale_factor)
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))

    # Summaries
    fusion_sum = 0.0
    fminr_sum  = 0.0
    for name, times in module_timer.times.items():
        a = ModuleTimer.avg_ms(times)
        if name.startswith("feature_fusion."):
            fusion_sum += a
        elif name.startswith("fminr."):
            fminr_sum += a

    print("\n=== PROFILING MODE ===")
    print(f"dtype: {dtype} | iters: {iters} | warmup: {warmup} | profile_ops: {profile_ops}")
    print(f"end-to-end mean: {np.mean(times_ms):.3f} ms | p50: {np.percentile(times_ms,50):.3f} ms | p90: {np.percentile(times_ms,90):.3f} ms")

    print("\nAverage forward time per module (ms):")
    print(f"lr_feat_extractor        : {ModuleTimer.avg_ms(module_timer.times.get('lr_feat_extractor', [])):.3f}")
    print(f"temporal_feat_extractor  : {ModuleTimer.avg_ms(module_timer.times.get('temporal_feat_extractor', [])):.3f}")
    print(f"hr_gb_feat_extractor     : {ModuleTimer.avg_ms(module_timer.times.get('hr_gb_feat_extractor', [])):.3f}")
    print(f"feature_fusion (sum kids): {fusion_sum:.3f}")
    print(f"fminr (sum kids)         : {fminr_sum:.3f}")

    if profile_ops:
        print("\nAverage op times (ms):")
        print(f"pixel_unshuffle     : {op_timer.avg('pixel_unshuffle'):.3f}")
        print(f"pixel_shuffle       : {op_timer.avg('pixel_shuffle'):.3f}")
        print(f"inverse_wavelet_iwt : {op_timer.avg('inverse_wavelet_iwt'):.3f}")

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # Restore patches
    if patched["pixel"]:
        F.pixel_unshuffle = _orig_pixel_unshuffle
        F.pixel_shuffle   = _orig_pixel_shuffle
    if patched["iwt"]:
        WaveletProcessor.batch_iwt = _orig_batch_iwt

    train.cleanup(); val.cleanup(); test.cleanup()
    return times_ms, module_timer, op_timer


# -----------------------------
# Main entry
# -----------------------------
if __name__ == "__main__":
    # Choose one:
    # 1) Best performance measurement
    times = run_perf(
        settings_path="config/config.json",
        iters=100,
        warmup=20,
        dtype=torch.float16,
        compile_mode="max-autotune",
        upscale=2,
        idx=2
    )

    # 2) Profiling breakdown (uncomment to run)
    times, module_timer, op_timer = run_profile(
        settings_path="config/config.json",
        iters=50,
        warmup=10,
        dtype=torch.float16,
        profile_ops=True,
        upscale=2,
        idx=2
    )
