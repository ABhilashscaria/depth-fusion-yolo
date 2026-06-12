import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import time
import json


# ─────────────────────────────────────────────────────────────────────────────
# Resolution presets — the #1 lever for latency vs. accuracy trade-off.
#
# Root cause of the 50ms → 400ms regression:
#   The old HF pipeline() call silently downscaled every input to 518×518
#   before running the depth model. The new raw model code feeds the depth
#   model at FULL native resolution — 8-10x more pixels → 8-10x slower.
#
# Fix: downscale the INPUT before depth inference, then upscale the depth
#      OUTPUT back to native res (cheap bicubic on GPU). FAST preset = old speed.
# ─────────────────────────────────────────────────────────────────────────────
DEPTH_RESOLUTION_PRESETS = {
    "FAST":     (518, 518),   # matches old HF pipeline default  → ~50 ms target
    "BALANCED": (756, 756),   # moderate detail gain              → ~120 ms
    "FULL":     None,         # native camera resolution          → ~400 ms
}


class DepthConditionedYOLO:
    def __init__(
        self,
        yolo_model='yolo11n-seg.pt',
        depth_model='depth-anything/Depth-Anything-V2-Small-hf',
        device=0,
        warmup_runs=3,
        depth_resolution="FAST",    # "FAST" | "BALANCED" | "FULL" | (H, W) tuple
    ):
        print("⚡ Initializing Raw VRAM Pipeline...")
        self.device_id = device
        self.device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        self.warmup_runs = warmup_runs

        # Resolve depth resolution setting
        if isinstance(depth_resolution, str):
            self.depth_size = DEPTH_RESOLUTION_PRESETS[depth_resolution]
        else:
            self.depth_size = depth_resolution   # custom tuple e.g. (640, 480)

        res_label = str(self.depth_size) if self.depth_size else "native (FULL)"
        print(f"   Depth resolution : {res_label}")

        if self.device != "cpu":
            cudnn.benchmark = True
            cudnn.deterministic = False

        # ── 1. YOLO ──────────────────────────────────────────────────────────
        print("-> Loading YOLO11n-Seg...")
        self.yolo = YOLO(yolo_model)
        self.yolo.to(self.device)

        # ── 2. Depth model — FP16 in a single fused load ─────────────────────
        # torch_dtype avoids the double-copy that .to(device).half() causes
        print("-> Loading Depth Anything V2 (FP16 fused)...")
        self.image_processor = AutoImageProcessor.from_pretrained(depth_model)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(
            depth_model,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.depth_model.eval()

        # ── 3. Warm-up — lets cudnn.benchmark settle and flushes JIT overhead ─
        print(f"-> Running {warmup_runs}x GPU warm-up passes...")
        self._warmup()

        print("✅ Dual-Engine Initialization Complete.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _sync(self):
        if self.device != "cpu":
            torch.cuda.synchronize(self.device)

    def _warmup(self):
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(self.warmup_runs):
            self._run_depth(dummy)
            self.yolo(dummy, verbose=False, retina_masks=True, device=self.device_id)
        self._sync()

    # ──────────────────────────────────────────────────────────────────────────
    # Depth sub-pipeline with resolution control
    # ──────────────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def _run_depth(self, frame_rgb_np):
        """
        Runs depth estimation at self.depth_size resolution then upscales back.

        Pipeline:
          frame (native res)
            └─ CPU resize to depth_size   [fast OpenCV]
            └─ image_processor + depth model   [GPU, FP16]
            └─ bicubic upscale to native res   [GPU]
            └─ normalize to [0, 1]             [GPU]

        When depth_size=None (FULL preset), the resize step is skipped and the
        model runs at native resolution (accurate but slow).
        """
        h_orig, w_orig = frame_rgb_np.shape[:2]

        if self.depth_size is not None:
            dh, dw = self.depth_size
            resized = cv2.resize(frame_rgb_np, (dw, dh), interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame_rgb_np

        inputs = self.image_processor(images=resized, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device).half()

        outputs = self.depth_model(pixel_values)
        predicted_depth = outputs.predicted_depth    # [1, dh', dw']

        # Upscale depth map back to original frame resolution on GPU
        depth_tensor = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h_orig, w_orig),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Normalize to [0, 1]
        d_min, d_max = depth_tensor.min(), depth_tensor.max()
        if d_max - d_min > 1e-5:
            depth_tensor = (depth_tensor - d_min) / (d_max - d_min)
        else:
            depth_tensor = torch.zeros_like(depth_tensor)

        return depth_tensor

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def predict(
        self,
        image_path_or_frame,
        min_volume=0.005,
        min_step_delta=0.05,
        profile=False,
    ):
        """
        Args:
            image_path_or_frame : str path or BGR numpy array
            min_volume          : variance threshold for 3-D volume gate
            min_step_delta      : depth-step threshold for pop-out gate
            profile             : print per-stage latency breakdown
        """
        if isinstance(image_path_or_frame, str):
            frame = cv2.imread(image_path_or_frame)
        else:
            frame = image_path_or_frame

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Stage 1: YOLO ─────────────────────────────────────────────────────
        self._sync()
        t0 = time.perf_counter()

        yolo_results = self.yolo(
            frame, verbose=False, retina_masks=True, device=self.device_id
        )[0]

        self._sync()
        t1 = time.perf_counter()

        # ── Stage 2: Depth at chosen resolution ──────────────────────────────
        depth_tensor = self._run_depth(frame_rgb)

        self._sync()
        t2 = time.perf_counter()

        # ── Stage 3: Fusion & Halo Logic ─────────────────────────────────────
        fused_predictions = []

        if yolo_results.boxes is not None and yolo_results.masks is not None:
            for i, box in enumerate(yolo_results.boxes):
                class_name = self.yolo.names[int(box.cls[0])]

                if class_name == 'person':
                    confidence = float(box.conf[0])
                    xyxy = box.xyxy[0].to(torch.int32)
                    x1, y1, x2, y2 = (
                        xyxy[0].item(), xyxy[1].item(),
                        xyxy[2].item(), xyxy[3].item()
                    )

                    binary_mask = yolo_results.masks.data[i].bool()
                    mask_float  = binary_mask.float().unsqueeze(0).unsqueeze(0)

                    # 5×5 halo expansion — 3 iterations ≈ 7 px ring
                    dilated_mask = mask_float
                    for _ in range(3):
                        dilated_mask = F.max_pool2d(
                            dilated_mask, kernel_size=5, stride=1, padding=2
                        )
                    halo_mask = (dilated_mask - mask_float).squeeze(0).squeeze(0).bool()

                    object_depths = depth_tensor[binary_mask]
                    halo_depths   = depth_tensor[halo_mask]

                    if object_depths.numel() > 10 and halo_depths.numel() > 10:
                        object_var    = float(torch.var(object_depths).item())
                        object_median = float(torch.median(object_depths).item())
                        halo_median   = float(torch.median(halo_depths).item())
                        step_delta    = object_median - halo_median
                    else:
                        object_var = object_median = halo_median = step_delta = 0.0

                    has_volume = object_var  > min_volume
                    pops_out   = step_delta  > min_step_delta

                    fused_predictions.append({
                        "box":        [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class":      class_name,
                        "depth_metrics": {
                            "object_variance": object_var,
                            "step_delta":      step_delta,
                            "has_volume":      has_volume,
                            "pops_out":        pops_out,
                        },
                        "is_real_3d": has_volume and pops_out,
                    })

        self._sync()
        t3 = time.perf_counter()

        if profile:
            yolo_ms  = (t1 - t0) * 1000
            depth_ms = (t2 - t1) * 1000
            fuse_ms  = (t3 - t2) * 1000
            total_ms = (t3 - t0) * 1000
            res_label = str(self.depth_size) if self.depth_size else "native"
            print(f"\n📊 Per-Stage Latency  [depth res: {res_label}]")
            print(f"   YOLO inference  : {yolo_ms:7.2f} ms")
            print(f"   Depth inference : {depth_ms:7.2f} ms")
            print(f"   Fusion / halo   : {fuse_ms:7.2f} ms")
            print(f"   ─────────────────────────────────")
            print(f"   Total           : {total_ms:7.2f} ms  ({1000/total_ms:.1f} FPS eq.)")

        return fused_predictions


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics — run once at startup to surface environment issues
# ─────────────────────────────────────────────────────────────────────────────
def print_env_diagnostics(device_id=0):
    import subprocess, sys
    import torch, cv2, ultralytics, transformers
    print("=" * 55)
    print("🔎 Environment Diagnostics")
    print("=" * 55)
    print(f"Python      : {sys.version.split()[0]}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"OpenCV      : {cv2.__version__}")
    print(f"Ultralytics : {ultralytics.__version__}")
    print(f"Transformers: {transformers.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device_id)
        print(f"\nGPU         : {props.name}")
        print(f"VRAM total  : {props.total_memory / 1024**2:.0f} MB")
        print(f"CUDA ver    : {torch.version.cuda}")
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=clocks.current.graphics,clocks.max.graphics,"
                 "temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                text=True,
            ).strip()
            cur_clk, max_clk, temp, pwr = [x.strip() for x in out.split(",")]
            throttled = int(cur_clk) < int(max_clk) * 0.9
            print(f"GPU clock   : {cur_clk}/{max_clk} MHz  "
                  f"{'⚠️ THROTTLED' if throttled else '✅ OK'}")
            print(f"Temperature : {temp} °C  {'⚠️ HOT' if int(temp) > 80 else '✅ OK'}")
            print(f"Power draw  : {pwr} W")
        except Exception:
            print("(nvidia-smi not reachable)")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — benchmarks all three presets so you can pick your trade-off
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_env_diagnostics()

    IMAGE = 'test1.webp'   # ← swap as needed

    for preset in ("FAST", "BALANCED", "FULL"):
        print(f"\n{'═' * 55}")
        print(f"  PRESET: {preset}")
        print(f"{'═' * 55}")

        engine = DepthConditionedYOLO(depth_resolution=preset, warmup_runs=3)

        if torch.cuda.is_available():
            torch.cuda.synchronize(0)

        t_start = time.perf_counter()
        results = engine.predict(IMAGE, profile=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize(0)

        wall_ms = (time.perf_counter() - t_start) * 1000
        print(f"\n⏱️  Wall-clock (GPU-synced): {wall_ms:.2f} ms  ({1000/wall_ms:.1f} FPS eq.)")
        print(json.dumps(results, indent=4))

        alloc = torch.cuda.memory_allocated(0) / 1024**2
        resrv = torch.cuda.memory_reserved(0)  / 1024**2
        print(f"VRAM  allocated: {alloc:.1f} MB  |  reserved: {resrv:.1f} MB")

        del engine
        torch.cuda.empty_cache()
