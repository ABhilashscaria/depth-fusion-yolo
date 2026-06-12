import torch
import time
import mlflow
import numpy as np
import cv2
import onnxruntime as ort
from core_engine_onnx import ONNXSpatialEngine


class ONNXPipelineProfiler:
    def __init__(self, engine: ONNXSpatialEngine, device_id=0):
        self.engine    = engine
        self.device_id = device_id
        self.device    = f"cuda:{device_id}"

        # Detect whether ONNX depth session is actually running on GPU or CPU.
        # This matters for timing strategy:
        #   - GPU EP  → CUDA events are accurate for depth stage
        #   - CPU EP  → depth stage is pure CPU work; perf_counter is correct,
        #               CUDA events would show near-zero (no GPU kernels launched)
        active_ep = engine.depth_session.get_providers()[0]
        self.depth_on_gpu = (active_ep == 'CUDAExecutionProvider')

        ep_label = "GPU (CUDAExecutionProvider)" if self.depth_on_gpu else "CPU (CPUExecutionProvider)"
        print(f"   Depth execution provider : {ep_label}")
        if not self.depth_on_gpu:
            print("   ⚠️  Depth is on CPU — wall-clock timing used for depth stage.\n"
                  "      CUDA events still used for YOLO + GPU fusion stages.")

        # CUDA events for GPU-side timing of YOLO and fusion stages
        # (still valid even when depth is CPU-side)
        self.ev_start  = torch.cuda.Event(enable_timing=True)
        self.ev_post_yolo  = torch.cuda.Event(enable_timing=True)
        self.ev_post_fusion = torch.cuda.Event(enable_timing=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Force GPU out of idle power state before any measurement.
    # RTX 4060 Laptop idles at ~225 MHz — first real work arrives at degraded
    # clocks, inflating latency for the first ~20+ frames.
    # ──────────────────────────────────────────────────────────────────────────
    def _boost_gpu_clocks(self):
        print("⚡ Forcing GPU clock boost (idle → performance state)...")
        a = torch.randn(2000, 2000, device=self.device)
        b = torch.randn(2000, 2000, device=self.device)
        for _ in range(5):
            _ = a @ b
        torch.cuda.synchronize()
        time.sleep(1.0)   # let power governor fully commit to performance state
        del a, b
        torch.cuda.empty_cache()
        print("   ✅ Clocks boosted.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Warmup: runs real pipeline frames to settle ONNX Runtime's kernel cache
    # and ORT arena allocator. Must synchronize at end — ORT's CUDA EP launches
    # async kernels that may still be in-flight when predict() returns on CPU.
    # ──────────────────────────────────────────────────────────────────────────
    def _warmup(self, test_frame, num_warmup):
        print(f"🔥 Warming up ONNX pipeline ({num_warmup} frames)...")
        for _ in range(num_warmup):
            _ = self.engine.predict(test_frame)

        # Always sync — even CPU EP path touches GPU for halo/fusion stage
        torch.cuda.synchronize()
        print("   ✅ Warmup complete.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Validate warmup stability using coefficient of variation.
    # Uses wall-clock (perf_counter) for total latency since the pipeline
    # mixes CPU (ONNX depth) and GPU (YOLO + fusion) work.
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_warmup(self, test_frame, window=10, cv_threshold=0.05):
        probe_latencies = []
        for _ in range(window):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = self.engine.predict(test_frame)
            torch.cuda.synchronize()
            probe_latencies.append((time.perf_counter() - t0) * 1000)

        mean_ms = np.mean(probe_latencies)
        cv = np.std(probe_latencies) / mean_ms

        print(f"   Warmup validation: mean={mean_ms:.2f}ms  CV={cv:.3f} "
              f"({'✅ settled' if cv < cv_threshold else '⚠️  still noisy'})")
        return cv < cv_threshold

    # ──────────────────────────────────────────────────────────────────────────
    # Per-stage timed predict — wraps engine internals to get stage breakdown.
    #
    # Stage split:
    #   t0 → [YOLO]   → t1  : CUDA event (GPU work)
    #   t1 → [Depth]  → t2  : wall-clock (CPU if ORT on CPU, GPU if CUDA EP)
    #   t2 → [Fusion] → t3  : CUDA event (GPU work)
    #
    # This correctly attributes depth latency regardless of which EP is active.
    # ──────────────────────────────────────────────────────────────────────────
    def _timed_predict(self, frame):
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Stage 1: YOLO (GPU) ───────────────────────────────────────────────
        torch.cuda.synchronize()
        self.ev_start.record()

        yolo_results = self.engine.yolo(frame, verbose=False, retina_masks=True)[0]

        self.ev_post_yolo.record()
        torch.cuda.synchronize()
        yolo_ms = self.ev_start.elapsed_time(self.ev_post_yolo)

        # ── Stage 2: Depth ONNX (CPU or GPU depending on active EP) ──────────
        # Use wall-clock here — CUDA events won't capture CPU-side ORT work
        t_depth_start = time.perf_counter()

        img_input    = self.engine._preprocess_depth(frame)
        onnx_outputs = self.engine.depth_session.run(
            None, {self.engine.depth_input_name: img_input}
        )
        raw_depth = onnx_outputs[0].squeeze()
        d_min, d_max = raw_depth.min(), raw_depth.max()
        depth_normalized = (
            (raw_depth - d_min) / (d_max - d_min)
            if d_max - d_min > 1e-5
            else np.zeros_like(raw_depth)
        )

        # If depth EP is GPU, sync before stopping clock to include GPU work
        if self.depth_on_gpu:
            torch.cuda.synchronize()

        depth_ms = (time.perf_counter() - t_depth_start) * 1000

        # ── Stage 3: Depth post-proc + Fusion (GPU) ───────────────────────────
        import torch.nn.functional as F

        self.ev_start.record()

        depth_tensor = torch.from_numpy(depth_normalized).to(self.engine.device)
        depth_tensor = F.interpolate(
            depth_tensor.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        fused_predictions = []
        if yolo_results.boxes is not None and yolo_results.masks is not None:
            for i, box in enumerate(yolo_results.boxes):
                class_name = self.engine.yolo.names[int(box.cls[0])]
                if class_name == 'person':
                    confidence = float(box.conf[0])
                    xyxy = box.xyxy[0].to(torch.int32)
                    x1, y1, x2, y2 = (
                        xyxy[0].item(), xyxy[1].item(),
                        xyxy[2].item(), xyxy[3].item()
                    )
                    binary_mask = yolo_results.masks.data[i].bool()
                    mask_float  = binary_mask.float().unsqueeze(0).unsqueeze(0)

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

                    fused_predictions.append({
                        "box":        [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class":      class_name,
                        "depth_metrics": {
                            "object_variance": object_var,
                            "step_delta":      step_delta,
                            "has_volume":      object_var  > 0.005,
                            "pops_out":        step_delta  > 0.05,
                        },
                        "is_real_3d": (object_var > 0.005) and (step_delta > 0.05),
                    })

        self.ev_post_fusion.record()
        torch.cuda.synchronize()
        fusion_ms = self.ev_start.elapsed_time(self.ev_post_fusion)

        return fused_predictions, yolo_ms, depth_ms, fusion_ms

    # ──────────────────────────────────────────────────────────────────────────
    # Main benchmark loop
    # ──────────────────────────────────────────────────────────────────────────
    def profile_stream(self, test_frame, num_iterations=100, warmup=10):

        # Stage 1: boost GPU clocks
        self._boost_gpu_clocks()

        # Stage 2: warmup at exact benchmark resolution
        self._warmup(test_frame, warmup)

        # Stage 3: validate stability, extend if needed
        print("🔍 Validating warmup stability...")
        settled = self._validate_warmup(test_frame)
        if not settled:
            print("   ⚠️  Still noisy — running 10 extra warmup frames...")
            self._warmup(test_frame, 10)
            self._validate_warmup(test_frame)

        # Stage 4: clean memory baseline
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device_id)

        print(f"\n🚀 Executing {num_iterations} benchmark iterations...\n")

        total_latencies = []
        yolo_latencies  = []
        depth_latencies = []
        fusion_latencies = []

        active_ep = self.engine.depth_session.get_providers()[0]

        with mlflow.start_run(run_name="ONNX_Pipeline_Benchmark"):
            mlflow.log_param("hardware_target",   "RTX_4060_Laptop")
            mlflow.log_param("yolo_model",         "yolo11n-seg.onnx")
            mlflow.log_param("depth_model",        "depth_anything_v2.onnx")
            mlflow.log_param("depth_input_size",   self.engine.depth_input_size)
            mlflow.log_param("depth_ep",           active_ep)
            mlflow.log_param("num_iterations",     num_iterations)
            mlflow.log_param("warmup_frames",      warmup)

            for i in range(num_iterations):
                # Wall-clock wraps the full frame including CPU↔GPU transfers
                torch.cuda.synchronize()
                t_wall_start = time.perf_counter()

                preds, yolo_ms, depth_ms, fusion_ms = self._timed_predict(test_frame)

                torch.cuda.synchronize()
                total_ms = (time.perf_counter() - t_wall_start) * 1000

                total_latencies.append(total_ms)
                yolo_latencies.append(yolo_ms)
                depth_latencies.append(depth_ms)
                fusion_latencies.append(fusion_ms)

                mlflow.log_metric("total_latency_ms",  total_ms,   step=i)
                mlflow.log_metric("yolo_latency_ms",   yolo_ms,    step=i)
                mlflow.log_metric("depth_latency_ms",  depth_ms,   step=i)
                mlflow.log_metric("fusion_latency_ms", fusion_ms,  step=i)

            # ── Aggregate ────────────────────────────────────────────────────
            def stats(arr):
                a = np.array(arr)
                return {
                    "avg": float(np.mean(a)),
                    "std": float(np.std(a)),
                    "p50": float(np.percentile(a, 50)),
                    "p95": float(np.percentile(a, 95)),
                    "p99": float(np.percentile(a, 99)),
                }

            total_s  = stats(total_latencies)
            yolo_s   = stats(yolo_latencies)
            depth_s  = stats(depth_latencies)
            fusion_s = stats(fusion_latencies)

            fps = 1000.0 / total_s["avg"]
            peak_vram_mb    = torch.cuda.max_memory_allocated(self.device_id) / (1024 ** 2)
            reserved_vram_mb = torch.cuda.memory_reserved(self.device_id)    / (1024 ** 2)

            # Log aggregate metrics
            for k, v in total_s.items():
                mlflow.log_metric(f"total_{k}_ms", v)
            mlflow.log_metric("avg_fps",      fps)
            mlflow.log_metric("peak_vram_mb", peak_vram_mb)

            # ── Print report ─────────────────────────────────────────────────
            print("\n" + "=" * 55)
            print("📊 ONNX PIPELINE TELEMETRY SUMMARY")
            print("=" * 55)
            print(f"Depth EP          : {active_ep}")
            print(f"Iterations        : {num_iterations}")
            print(f"")
            print(f"{'Stage':<18} {'Avg':>8} {'Std':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
            print(f"{'─'*18} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
            for label, s in [("YOLO", yolo_s), ("Depth (ORT)", depth_s),
                              ("GPU Fusion", fusion_s), ("TOTAL", total_s)]:
                print(f"{label:<18} {s['avg']:>7.2f}ms {s['std']:>7.2f}ms "
                      f"{s['p50']:>7.2f}ms {s['p95']:>7.2f}ms {s['p99']:>7.2f}ms")
            print(f"")
            print(f"Throughput        : {fps:.2f} FPS")
            print(f"Peak VRAM Active  : {peak_vram_mb:.2f} MB")
            print(f"Total VRAM Locked : {reserved_vram_mb:.2f} MB")
            print("=" * 55)
            print("✅ Telemetry persisted to MLflow tracking server.")

        return {
            "avg_ms":        total_s["avg"],
            "p99_ms":        total_s["p99"],
            "fps":           fps,
            "peak_vram_mb":  peak_vram_mb,
            "stage_avg_ms": {
                "yolo":   yolo_s["avg"],
                "depth":  depth_s["avg"],
                "fusion": fusion_s["avg"],
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = ONNXSpatialEngine(warmup_runs=5)

    test_image = cv2.imread('maxresdefault.jpg')
    if test_image is None:
        raise FileNotFoundError("Test image not found. Check the path.")

    profiler = ONNXPipelineProfiler(engine)
    profiler.profile_stream(test_image, num_iterations=100, warmup=10)
