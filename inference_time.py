from jupyter_notebooks.notebook_utils import *
import time
import torch

ITER = 100

settings = initialize_settings("config/config.json")
train, val, test = WDSSDataset.get_datasets(settings)
preprocessor = get_preprocessor(settings)

frame = train.get_item(2, upscale_factor=2, no_patch=True)
frame = WDSSDataset.batch_to_device(frame, device)
lr_frame = frame[FrameGroup.LR_INP.value]
gt = frame[FrameGroup.GT.value]
extra = frame[FrameGroup.EXTRA.value]
upscale_factor: float = gt.shape[-2] / lr_frame.shape[-2]

hr_gbuffer = frame[FrameGroup.GB_INP.value]
temporal = frame[FrameGroup.TEMPORAL_INP.value]

model = get_model(settings['model_config']).to(device).half()


# -----------------------------
# 1) Timers
# -----------------------------
class ModuleTimer:
    def __init__(self):
        self.starts = {}
        self.times = {}  # name -> list[ms]

    def pre_hook(self, name):
        def _pre(module, inputs):
            # Use the first tensor input to decide cuda/cpu timing
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
                self.times.setdefault(name, []).append(float(s.elapsed_time(e)))  # ms
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
            self.times.setdefault(key, []).append(float(s.elapsed_time(e)))  # ms
        else:
            t0 = a
            self.times.setdefault(key, []).append((time.perf_counter() - t0) * 1000.0)

    def avg(self, key):
        xs = self.times.get(key, [])
        return (sum(xs) / len(xs)) if xs else 0.0


# -----------------------------
# 2) Register module hooks
# -----------------------------
module_timer = ModuleTimer()
hooks = []

# Direct modules you want as a single block
direct_modules = {
    "lr_feat_extractor": model.lr_feat_extractor,
    "temporal_feat_extractor": model.temporal_feat_extractor,
    "hr_gb_feat_extractor": model.hr_gb_feat_extractor,
}

for name, m in direct_modules.items():
    hooks.append(m.register_forward_pre_hook(module_timer.pre_hook(name)))
    hooks.append(m.register_forward_hook(module_timer.post_hook(name)))

# For feature_fusion and fminr: hook *children* so it works even if .forward() is called directly.
def hook_all_children(root_module, prefix):
    for n, m in root_module.named_modules():
        if n == "":
            continue
        name = f"{prefix}.{n}"
        hooks.append(m.register_forward_pre_hook(module_timer.pre_hook(name)))
        hooks.append(m.register_forward_hook(module_timer.post_hook(name)))

hook_all_children(model.feature_fusion, "feature_fusion")
hook_all_children(model.fminr, "fminr")


# -----------------------------
# 3) Monkey-patch pixel shuffle/unshuffle and batch_iwt
# -----------------------------
op_timer = OpTimer()

# Save originals
_orig_pixel_unshuffle = F.pixel_unshuffle
_orig_pixel_shuffle   = F.pixel_shuffle

def timed_pixel_unshuffle(x, downscale_factor):
    op_timer.start("pixel_unshuffle", x)
    out = _orig_pixel_unshuffle(x, downscale_factor)
    op_timer.stop("pixel_unshuffle")
    return out

def timed_pixel_shuffle(x, upscale_factor):
    op_timer.start("pixel_shuffle", x)
    out = _orig_pixel_shuffle(x, upscale_factor)
    op_timer.stop("pixel_shuffle")
    return out

F.pixel_unshuffle = timed_pixel_unshuffle
F.pixel_shuffle   = timed_pixel_shuffle

# WaveletProcessor patch (your print shows it's utils.wavelet.WaveletProcessor, so this is correct)
_orig_batch_iwt = WaveletProcessor.batch_iwt

def timed_batch_iwt(x):
    op_timer.start("inverse_wavelet_iwt", x)
    out = _orig_batch_iwt(x)
    op_timer.stop("inverse_wavelet_iwt")
    return out

WaveletProcessor.batch_iwt = timed_batch_iwt


# -----------------------------
# 4) Ensure 4D inputs
# -----------------------------
def ensure_4d(x):
    if x.dim() == 3:
        return x.unsqueeze(0)
    return x

lr_frame   = ensure_4d(lr_frame).half()
hr_gbuffer = ensure_4d(hr_gbuffer).half()
temporal   = ensure_4d(temporal).half()

# -----------------------------
# 5) Run warmup + timed runs
# -----------------------------
warmup = 10

model.eval()
with torch.no_grad():
    for _ in range(warmup):
        _ = model(lr_frame, hr_gbuffer, temporal, upscale_factor)

    for _ in range(ITER):
        _ = model(lr_frame, hr_gbuffer, temporal, upscale_factor)


# -----------------------------
# 6) Summarize results
# -----------------------------
# Aggregate child-module time under fusion / fminr
fusion_sum = 0.0
fminr_sum  = 0.0

for name, times in module_timer.times.items():
    a = ModuleTimer.avg_ms(times)
    if name.startswith("feature_fusion."):
        fusion_sum += a
    elif name.startswith("fminr."):
        fminr_sum += a

print("\nAverage forward time per module (ms):")
print(f"lr_feat_extractor       : {ModuleTimer.avg_ms(module_timer.times.get('lr_feat_extractor', [])):.3f}")
print(f"temporal_feat_extractor : {ModuleTimer.avg_ms(module_timer.times.get('temporal_feat_extractor', [])):.3f}")
print(f"hr_gb_feat_extractor    : {ModuleTimer.avg_ms(module_timer.times.get('hr_gb_feat_extractor', [])):.3f}")
print(f"feature_fusion (sum kids): {fusion_sum:.3f}")
print(f"fminr (sum kids)         : {fminr_sum:.3f}")

print("\nAverage op times (ms):")
print(f"pixel_unshuffle     : {op_timer.avg('pixel_unshuffle'):.3f}")
print(f"pixel_shuffle       : {op_timer.avg('pixel_shuffle'):.3f}")
print(f"inverse_wavelet_iwt : {op_timer.avg('inverse_wavelet_iwt'):.3f}")


# -----------------------------
# 7) Cleanup: remove hooks + restore patched functions
# -----------------------------
for h in hooks:
    h.remove()

F.pixel_unshuffle = _orig_pixel_unshuffle
F.pixel_shuffle   = _orig_pixel_shuffle
WaveletProcessor.batch_iwt = _orig_batch_iwt

train.cleanup()
val.cleanup()
test.cleanup()
