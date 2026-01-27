from jupyter_notebooks.notebook_utils import *
import time
import torch
import torch.nn.functional as F  # noqa: F401
from tqdm import tqdm  # noqa: F401
from network.models import WDSSSWTResBlockGB, WDSSSWT2L, WDSSSWT, WDSSRegular, WDSSNoWavelet
import json

try:
    import triton # type: ignore
    _triton_available = True
except ImportError:
    _triton_available = False

# Check if system is windows or linux
_is_windows = sys.platform.startswith("win")
_is_linux = sys.platform.startswith("linux")


# ----------------------------------------------------
# Global Constants
# ----------------------------------------------------
WARMUP_ITERS = 100
EVALUATION_ITERS = 500

# ----------------------------------------------------
# Global backend knobs (good defaults)
# ----------------------------------------------------
torch.backends.cudnn.allow_tf32= True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

# ----------------------------------------------------
# Utilities
# ----------------------------------------------------
def prepare_inputs(upscale_factor: float = 2.0, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """ Prepare random input tensors for inference time measurement.

    Args:
        upscale_factor (float): The factor by which the input image is upscaled.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]: Low-res input tensor, high-res g-buffer tensor,
        temporal frame tensor, and the upscale factor.
    """
    # Define dimensions
    height, width = 1920, 1080
    lr_height, lr_width = int(height / upscale_factor), int(width / upscale_factor)
    batch = 1

    lr_frame = torch.randn(batch, 3, lr_height, lr_width).to(device, non_blocking=True, dtype=dtype)
    hr_gbuffer = torch.randn(batch, 12, height, width).to(device, non_blocking=True, dtype=dtype)
    temporal_frame = torch.randn(batch, 8, height, width).to(device, non_blocking=True, dtype=dtype)

    upscale_factor = float(hr_gbuffer.shape[-1] / lr_frame.shape[-1])
    return lr_frame, hr_gbuffer, temporal_frame, upscale_factor

# ----------------------------------------------------
# Timers for profiling mode
# ----------------------------------------------------
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
    

# ----------------------------------------------------
# Model Optimization Stuff
# ----------------------------------------------------
def optimize_model_for_inference(model: torch.nn.Module, compile_mode: str = "max-autotune") -> torch.nn.Module:
    """ Optimize the model for inference using TorchDynamo.

    Args:
        model (torch.nn.Module): The model to optimize.
        compile_mode (str): The compilation mode for TorchDynamo.

    Returns:
        torch.nn.Module: The optimized model.
    """
    if compile_mode == "none":
        return model

    try:
        model = torch.compile(model, mode=compile_mode)
        print(f"Model optimized with TorchDynamo using mode: {compile_mode}")
    except Exception as e:
        print(f"Failed to optimize model with TorchDynamo: {e}")
    return model

# ----------------------------------------------------
# Perf Test Functions
# ----------------------------------------------------
def run_perf(model: torch.nn.Module, dtype: torch.dtype = torch.float32, upscale_factor: float = 2.0, compile_mode: str = "max-autotune", evaluation_name: str = "") -> Tuple[float, float, float]:
    """Run performance test on the given model.
    Args:
        model (torch.nn.Module): The model to test.
        dtype (torch.dtype): The data type for the input tensors.
        upscale_factor (float): The factor by which the input image is upscaled.
        compile_mode (str): The compilation mode for TorchDynamo.
    Returns:
        Tuple[float, float, float]: Average inference time (ms), 50th percentile latency (ms), 95th percentile latency (ms).
    """

    lr_frame, hr_gbuffer, temporal_frame, upscale_factor = prepare_inputs(upscale_factor, dtype)
    model = model.to(device, dtype=dtype).eval().to(memory_format=torch.channels_last)
    model = optimize_model_for_inference(model, compile_mode)

    # Timed iterations
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times_ms = []

    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=dtype):
        for _ in tqdm(range(WARMUP_ITERS + EVALUATION_ITERS), desc=f"Measuring Inference Time ({evaluation_name})"):
            starter.record()
            _ = model(lr_frame, hr_gbuffer, temporal_frame, upscale_factor)
            ender.record()
            torch.cuda.synchronize()
            curr_time_ms = starter.elapsed_time(ender)
            times_ms.append(curr_time_ms)

    # Discard warm-up times
    times_ms = times_ms[WARMUP_ITERS:]

    print(f"({evaluation_name}) Average Inference Time over {EVALUATION_ITERS} runs: {sum(times_ms) / len(times_ms):.2f} ms")
    
    mean = np.mean(times_ms)
    p50 = np.percentile(times_ms, 50)
    p95 = np.percentile(times_ms, 95)

    return mean, p50, p95

# ----------------------------------------------------
# Profile Test for Module-level Timing
# ----------------------------------------------------
def run_profile(model: torch.nn.Module, dtype: torch.dtype = torch.float32, upscale_factor: float = 2.0, profile_ops: bool = True, evaluation_name: str = "") -> Dict[str, Any]:
    """Run profiling test on the given model to measure module-level timing.
    Args:
        model (torch.nn.Module): The model to test.
        dtype (torch.dtype): The data type for the input tensors.
        upscale_factor (float): The factor by which the input image is upscaled.
        profile_ops (bool): Whether to profile individual operations.
        Dict[str, Any]: A dictionary containing average times per module.
    """

    lr_frame, hr_gbuffer, temporal_frame, upscale_factor = prepare_inputs(upscale_factor, dtype)
    model = model.to(device, dtype=dtype).eval().to(memory_format=torch.channels_last)

    # Compile Model
    # model = optimize_model_for_inference(model)

    # Hook Timers
    module_timer = ModuleTimer()
    hooks = []

    # Top level modules
    direct_modules = {
        "lr_feat_extractor": model.lr_feat_extractor,
        "temporal_feat_extractor": model.temporal_feat_extractor,
        "hr_gb_feat_extractor": model.hr_gb_feat_extractor,
    }
    for name, module in direct_modules.items():
        hooks.append(module.register_forward_pre_hook(module_timer.pre_hook(name)))
        hooks.append(module.register_forward_hook(module_timer.post_hook(name)))

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

    # --- Op timers (optional) ---
    op_timer = OpTimer()
    patched = {'pixel': False, 'iwt': False}

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

    # Timed end-to-end (optional, still useful)
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    times_ms = []

    # Iterations
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=dtype):
        for _ in tqdm(range(WARMUP_ITERS + EVALUATION_ITERS), desc=f"Profiling Inference Time ({evaluation_name})"):
            starter.record()
            _ = model(lr_frame, hr_gbuffer, temporal_frame, upscale_factor)
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))

    # Discard warm-up times
    times_ms = times_ms[WARMUP_ITERS:]

    # Summaries
    fusion_sum = 0.0
    fminr_sum = 0.0
    for name, times in module_timer.times.items():
        avg_time = ModuleTimer.avg_ms(times)
        if name.startswith("feature_fusion."):
            fusion_sum += avg_time
        elif name.startswith("fminr."):
            fminr_sum += avg_time

    # Average results
    res = {}
    res["feature_fusion"] = fusion_sum
    res["fminr"] = fminr_sum
    res["lr_feat_extractor"] = ModuleTimer.avg_ms(module_timer.times.get("lr_feat_extractor", []))
    res["temporal_feat_extractor"] = ModuleTimer.avg_ms(module_timer.times.get("temporal_feat_extractor", []))
    res["hr_gb_feat_extractor"] = ModuleTimer.avg_ms(module_timer.times.get("hr_gb_feat_extractor", []))
    if profile_ops:
        res["pixel_unshuffle"] = op_timer.avg("pixel_unshuffle")
        res["pixel_shuffle"] = op_timer.avg("pixel_shuffle")
        res["inverse_wavelet_iwt"] = op_timer.avg("inverse_wavelet_iwt")

    return res


if __name__ == "__main__":
    # 1. Performance test

    perf_test_res: List[Dict[str, float]] = []

    name, model, dtype = "No Wavelet NoResblock FP16", WDSSNoWavelet.WDSSNoWavelet(resblock_gb=False), torch.float16
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model

    name, model, dtype = "No Wavelet Resblock Reparam FP16", WDSSNoWavelet.WDSSNoWavelet(resblock_gb=True, reparam_gb=True), torch.float16
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model

    name, model, dtype = "No Wavelet Resblock NoReparam FP16", WDSSNoWavelet.WDSSNoWavelet(resblock_gb=True, reparam_gb=False), torch.float16
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model

    WaveletProps.WAVELET_TRANSFORM_TYPE = 'dwt'

    # name, model, dtype = "WDSSRegular No Resblock FP16", WDSSRegular.WDSSRegular(), torch.float16
    # avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    # perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    # del model
    # torch.cuda.empty_cache()

    # name, model, dtype = "WDSSRegular Resblock NoReparam FP16", WDSSRegular.WDSSRegular(resblock_gb=True, reparam_gb=False), torch.float16
    # avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    # perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    # del model
    # torch.cuda.empty_cache()

    # name, model, dtype = "WDSSRegular Resblock Reparam FP16", WDSSRegular.WDSSRegular(resblock_gb=True, reparam_gb=True), torch.float16
    # avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    # perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    # del model
    # torch.cuda.empty_cache()

    WaveletProps.WAVELET_TRANSFORM_TYPE = 'swt'

    name, model, dtype = "WDSSSWTResBlockGB Reparam FP16", WDSSSWTResBlockGB.WDSSSWTResBlockGB(reparam=True), torch.float16
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    name, model, dtype = "WDSSSWT FP16", WDSSSWT.WDSSSWT(), torch.float16
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    name, model, dtype = "WDSSSWTResBlockGB NoReparam FP16", WDSSSWTResBlockGB.WDSSSWTResBlockGB(reparam=False), torch.float16
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    WaveletProps.DECOMPOSITION_LEVEL = 2
    name, model, dtype = "WDSSSWT2L Reparam FP16", WDSSSWT2L.WDSSSWT2L(reparam=True), torch.float16
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    name, model, dtype = "WDSSSWT2L No Reparam FP16", WDSSSWT2L.WDSSSWT2L(reparam=False), torch.float16
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    # Now the whole thing with FP32
    WaveletProps.WAVELET_TRANSFORM_TYPE = 'dwt'
    WaveletProps.DECOMPOSITION_LEVEL = 1

    name, model, dtype = "No Wavelet NoResblock FP32", WDSSNoWavelet.WDSSNoWavelet(resblock_gb=False), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model

    name, model, dtype = "No Wavelet Resblock Reparam FP32", WDSSNoWavelet.WDSSNoWavelet(resblock_gb=True, reparam_gb=True), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model

    name, model, dtype = "No Wavelet Resblock NoReparam FP32", WDSSNoWavelet.WDSSNoWavelet(resblock_gb=True, reparam_gb=False), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model

    name, model, dtype = "WDSSRegular No Resblock FP32", WDSSRegular.WDSSRegular(), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    name, model, dtype = "WDSSRegular Resblock NoReparam FP32", WDSSRegular.WDSSRegular(resblock_gb=True, reparam_gb=False), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    name, model, dtype = "WDSSRegular Resblock Reparam FP32", WDSSRegular.WDSSRegular(resblock_gb=True, reparam_gb=True), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    WaveletProps.WAVELET_TRANSFORM_TYPE = 'swt'

    name, model, dtype = "WDSSSWT FP32", WDSSSWT.WDSSSWT(), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    name, model, dtype = "WDSSSWTResBlockGB NoReparam FP32", WDSSSWTResBlockGB.WDSSSWTResBlockGB(reparam=False), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    name, model, dtype = "WDSSSWTResBlockGB Reparam FP32", WDSSSWTResBlockGB.WDSSSWTResBlockGB(reparam=True), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    WaveletProps.DECOMPOSITION_LEVEL = 2
    name, model, dtype = "WDSSSWT2L Reparam FP32", WDSSSWT2L.WDSSSWT2L(reparam=True), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    name, model, dtype = "WDSSSWT2L No Reparam FP32", WDSSSWT2L.WDSSSWT2L(reparam=False), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name, compile_mode="max-autotune")
    perf_test_res.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    with open("inference_time_results.json", "w") as f:
        json.dump(perf_test_res, f, indent=4)

    # Whole thing again with no compilation
    print("\n\n=== No Compilation Mode ===\n\n")

    perf_test_res_nocompile: List[Dict[str, float]] = []

    WaveletProps.WAVELET_TRANSFORM_TYPE = 'dwt'
    WaveletProps.DECOMPOSITION_LEVEL = 1

    # FP16 no-compilation
    name, model, dtype = "No Wavelet NoResblock FP16", WDSSNoWavelet.WDSSNoWavelet(resblock_gb=False), torch.float16
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model

    name, model, dtype = "No Wavelet Resblock Reparam FP16", WDSSNoWavelet.WDSSNoWavelet(resblock_gb=True, reparam_gb=True), torch.float16
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model

    name, model, dtype = "No Wavelet Resblock NoReparam FP16", WDSSNoWavelet.WDSSNoWavelet(resblock_gb=True, reparam_gb=False), torch.float16
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model

    WaveletProps.WAVELET_TRANSFORM_TYPE = 'dwt'

    # name, model, dtype = "WDSSRegular No Resblock FP16", WDSSRegular.WDSSRegular(), torch.float16
    # avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    # perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    # del model
    # torch.cuda.empty_cache()

    # name, model, dtype = "WDSSRegular Resblock NoReparam FP16", WDSSRegular.WDSSRegular(resblock_gb=True, reparam_gb=False), torch.float16
    # avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    # perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    # del model
    # torch.cuda.empty_cache()

    # name, model, dtype = "WDSSRegular Resblock Reparam FP16", WDSSRegular.WDSSRegular(resblock_gb=True, reparam_gb=True), torch.float16
    # avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    # perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    # del model
    # torch.cuda.empty_cache()

    WaveletProps.WAVELET_TRANSFORM_TYPE = 'swt'

    name, model, dtype = "WDSSSWTResBlockGB Reparam FP16", WDSSSWTResBlockGB.WDSSSWTResBlockGB(reparam=True), torch.float16
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    name, model, dtype = "WDSSSWT FP16", WDSSSWT.WDSSWT(), torch.float16
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    name, model, dtype = "WDSSSWTResBlockGB NoReparam FP16", WDSSSWTResBlockGB.WDSSSWTResBlockGB(reparam=False), torch.float16
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    WaveletProps.DECOMPOSITION_LEVEL = 2
    name, model, dtype = "WDSSSWT2L Reparam FP16", WDSSSWT2L.WDSSWT2L(reparam=True), torch.float16
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    name, model, dtype = "WDSSSWT2L No Reparam FP16", WDSSSWT2L.WDSSWT2L(reparam=False), torch.float16
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    # FP32 no-compilation
    WaveletProps.WAVELET_TRANSFORM_TYPE = 'dwt'
    WaveletProps.DECOMPOSITION_LEVEL = 1

    name, model, dtype = "No Wavelet NoResblock FP32", WDSSNoWavelet.WDSSNoWavelet(resblock_gb=False), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model

    name, model, dtype = "No Wavelet Resblock Reparam FP32", WDSSNoWavelet.WDSSNoWavelet(resblock_gb=True, reparam_gb=True), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model

    name, model, dtype = "No Wavelet Resblock NoReparam FP32", WDSSNoWavelet.WDSSNoWavelet(resblock_gb=True, reparam_gb=False), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model

    name, model, dtype = "WDSSRegular No Resblock FP32", WDSSRegular.WDSSRegular(), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    name, model, dtype = "WDSSRegular Resblock NoReparam FP32", WDSSRegular.WDSSRegular(resblock_gb=True, reparam_gb=False), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    name, model, dtype = "WDSSRegular Resblock Reparam FP32", WDSSRegular.WDSSRegular(resblock_gb=True, reparam_gb=True), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    WaveletProps.WAVELET_TRANSFORM_TYPE = 'swt'

    name, model, dtype = "WDSSSWT FP32", WDSSSWT.WDSSWT(), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    name, model, dtype = "WDSSSWTResBlockGB NoReparam FP32", WDSSSWTResBlockGB.WDSSSWTResBlockGB(reparam=False), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    name, model, dtype = "WDSSSWTResBlockGB Reparam FP32", WDSSSWTResBlockGB.WDSSSWTResBlockGB(reparam=True), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    WaveletProps.DECOMPOSITION_LEVEL = 2
    name, model, dtype = "WDSSSWT2L Reparam FP32", WDSSSWT2L.WDSSWT2L(reparam=True), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    name, model, dtype = "WDSSSWT2L No Reparam FP32", WDSSSWT2L.WDSSWT2L(reparam=False), torch.float32
    avg, p50, p95 = run_perf(model, dtype=dtype, evaluation_name=name+" (no compile)", compile_mode="none")
    perf_test_res_nocompile.append({"Model": name, "Avg (ms)": avg, "P50 (ms)": p50, "P95 (ms)": p95})
    del model
    torch.cuda.empty_cache()

    # Write the results to a JSON file
    with open("inference_time_results_nocompile.json", "w") as f:
        json.dump(perf_test_res_nocompile, f, indent=4)


    # 2. Profiling test
    WaveletProps.WAVELET_TRANSFORM_TYPE = 'swt'
    WaveletProps.DECOMPOSITION_LEVEL = 1

    profile_res_fp16 = run_profile(
        WDSSSWTResBlockGB.WDSSSWTResBlockGB(reparam=True),
        dtype=torch.float16,
        evaluation_name="WDSSSWTResBlockGB FP16",
        profile_ops=True,
    )
    profile_res_fp32 = run_profile(
        WDSSSWTResBlockGB.WDSSSWTResBlockGB(reparam=True),
        dtype=torch.float32,
        evaluation_name="WDSSSWTResBlockGB FP32",
        profile_ops=True,
    )

    # Write profiling results to JSON files
    with open("inference_profile_fp16.json", "w") as f:
        json.dump(profile_res_fp16, f, indent=4)
    with open("inference_profile_fp32.json", "w") as f:
        json.dump(profile_res_fp32, f, indent=4)

