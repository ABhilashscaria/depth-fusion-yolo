import cv2
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
from ultralytics import YOLO
import time
import json


# ImageNet normalization constants — required by Depth Anything V2
# (trained on ImageNet-pretrained ViT backbone)
DEPTH_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEPTH_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ONNXSpatialEngine:
    def __init__(
        self,
        yolo_onnx_path='yolo11n-seg.onnx',
        depth_onnx_path='depth_anything_v2.onnx',
        depth_input_size=518,       # resize input to this before depth model
        warmup_runs=5,
    ):
        print("⚡ Initializing High-Performance ONNX Runtime Execution Context...")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.depth_input_size = depth_input_size

        # ── ONNX Runtime provider config ─────────────────────────────────────
        # gpu_mem_limit: 4GB is safe on a 7.6GB card; 2GB risks OOM under load
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,   # FIX: was 2GB, raised to 4GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }),
            'CPUExecutionProvider',
        ]

        # ── 1. YOLO ───────────────────────────────────────────────────────────
        print("-> Loading YOLO11n-Seg ONNX Engine...")
        self.yolo = YOLO(yolo_onnx_path, task='segment')

        # ── 2. Depth Anything V2 ONNX session ────────────────────────────────
        print("-> Loading Depth Anything V2 ONNX Session...")
        self.depth_session   = ort.InferenceSession(depth_onnx_path, providers=providers)
        self.depth_input_name = self.depth_session.get_inputs()[0].name

        # Verify which execution provider was actually granted (not just requested)
        active_providers = self.depth_session.get_providers()
        print(f"   Active EP: {active_providers[0]}")
        if active_providers[0] != 'CUDAExecutionProvider':
            print("   ⚠️  WARNING: CUDA EP not active — running on CPU. "
                  "Check onnxruntime-gpu installation.")

        # ── 3. Warmup ─────────────────────────────────────────────────────────
        print(f"-> Running {warmup_runs}x warm-up passes...")
        self._warmup(warmup_runs)

        print("✅ ONNX Pipeline Initialization Complete.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Warmup
    # ──────────────────────────────────────────────────────────────────────────
    def _warmup(self, n):
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(n):
            self.predict(dummy)
        if self.device != "cpu":
            torch.cuda.synchronize()

    # ──────────────────────────────────────────────────────────────────────────
    # Depth preprocessing
    # ──────────────────────────────────────────────────────────────────────────
    def _preprocess_depth(self, frame_bgr):
        """
        Resize → BGR2RGB → /255 → ImageNet normalize → BCHW float32 numpy array.
        Depth Anything V2 requires ImageNet normalization (mean/std subtraction).
        Skipping this step silently degrades depth quality without crashing.
        """
        s = self.depth_input_size
        img = cv2.resize(frame_bgr, (s, s), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # FIX: apply ImageNet normalization — was missing in original
        img = (img - DEPTH_MEAN) / DEPTH_STD

        # HWC → BCHW
        return np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    # ──────────────────────────────────────────────────────────────────────────
    # Main inference
    # ──────────────────────────────────────────────────────────────────────────
    def predict(self, image_path_or_frame, min_volume=0.005, min_step_delta=0.05):
        if isinstance(image_path_or_frame, str):
            frame = cv2.imread(image_path_or_frame)
        else:
            frame = image_path_or_frame

        h, w = frame.shape[:2]

        # ── Stage 1: YOLO ────────────────────────────────────────────────────
        yolo_results = self.yolo(frame, verbose=False, retina_masks=True)[0]

        # ── Stage 2: Depth ONNX ──────────────────────────────────────────────
        img_input = self._preprocess_depth(frame)
        onnx_outputs = self.depth_session.run(None, {self.depth_input_name: img_input})

        # FIX: squeeze() handles all common output shapes:
        #   (1, 1, H, W) → (H, W)   ← most common from DA-V2 export
        #   (1, H, W)    → (H, W)   ← some exporters
        #   (H, W)       → (H, W)   ← rare, already correct
        raw_depth = onnx_outputs[0].squeeze()
        if raw_depth.ndim != 2:
            raise ValueError(f"Unexpected depth output shape after squeeze: {raw_depth.shape}")

        # Normalize to [0, 1]
        d_min, d_max = raw_depth.min(), raw_depth.max()
        depth_normalized = (
            (raw_depth - d_min) / (d_max - d_min)
            if d_max - d_min > 1e-5
            else np.zeros_like(raw_depth)
        )

        # Move to GPU and upscale to native frame resolution
        depth_tensor = torch.from_numpy(depth_normalized).to(self.device)
        depth_tensor = F.interpolate(
            depth_tensor.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bicubic",        # bicubic > bilinear for depth upscaling
            align_corners=False,
        ).squeeze()

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

                    # FIX: 3 × 5px iterations ≈ 7px ring (was a single 31px dilation
                    # which captured far background instead of the immediate boundary)
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

        return fused_predictions


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    engine = ONNXSpatialEngine()

    # FIX: warmup and benchmark at the SAME resolution
    # Warming up at 1920×1080 then benchmarking at a different res
    # invalidates the cudnn kernel cache from warmup
    test_frame = cv2.imread('rajatest.jpg')
    if test_frame is None:
        raise FileNotFoundError("rajatest.jpg not found.")

    # Additional warmup at exact test resolution (engine __init__ used 480×640)
    for _ in range(3):
        _ = engine.predict(test_frame)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # FIX: synchronize before AND after for accurate GPU-inclusive timing
    print("🚀 Profiling ONNX Runtime Performance...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    results = engine.predict(test_frame)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency = (time.perf_counter() - start_time) * 1000
    print(f"⏱️  Single Frame Latency (GPU-synced): {latency:.2f} ms  "
          f"({1000/latency:.1f} FPS eq.)")

    print("\n--- Payload ---")
    print(json.dumps(results, indent=4))

    allocated_vram = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"\nActive VRAM Footprint: {allocated_vram:.2f} MB")
