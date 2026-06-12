import torch
import time
import mlflow
import numpy as np
import cv2
from core_engine import DepthConditionedYOLO


class PipelineProfiler:
    def __init__(self, engine, device_id=0):
        self.engine = engine
        self.device = f"cuda:{device_id}"
        self.device_id = device_id

        # CUDA events for GPU-side timing (most accurate — not affected by CPU scheduling)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event   = torch.cuda.Event(enable_timing=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Force GPU out of idle power state BEFORE warmup.
    # RTX 4060 Laptop idles at ~225 MHz. First real work hits clocks still
    # ramping → inflated latency for first ~20 frames. This matmul forces a
    # full boost commit before any timed work starts.
    # ──────────────────────────────────────────────────────────────────────────
    def _boost_gpu_clocks(self):
        print("⚡ Forcing GPU clock boost (idle → performance state)...")
        a = torch.randn(2000, 2000, device=self.device)
        b = torch.randn(2000, 2000, device=self.device)
        for _ in range(5):
            _ = a @ b
        torch.cuda.synchronize()
        time.sleep(1.0)   # give power governor time to fully commit
        del a, b
        torch.cuda.empty_cache()
        print("   ✅ Clocks boosted.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Warmup — runs real pipeline frames to settle cudnn kernel cache.
    # MUST end with synchronize() so the benchmark loop doesn't start before
    # the last warmup frame finishes on the GPU.
    # ──────────────────────────────────────────────────────────────────────────
    def _warmup(self, test_frame, num_warmup):
        print(f"🔥 Warming up pipeline ({num_warmup} frames)...")
        for i in range(num_warmup):
            _ = self.engine.predict(test_frame)

        torch.cuda.synchronize()   # ← critical: wait for last warmup frame to fully complete
        print("   ✅ Warmup complete.\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Optional: validate warmup settled by checking latency std deviation over
    # a small window. If variance is still high, extend warmup automatically.
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_warmup(self, test_frame, window=10, cv_threshold=0.05):
        """
        Runs a small window of timed frames. If coefficient of variation (std/mean)
        exceeds threshold, the pipeline hasn't settled yet — extends warmup.
        Returns True if settled, False if more warmup needed.
        """
        probe_latencies = []
        for _ in range(window):
            self.start_event.record()
            _ = self.engine.predict(test_frame)
            self.end_event.record()
            torch.cuda.synchronize()
            probe_latencies.append(self.start_event.elapsed_time(self.end_event))

        mean_ms = np.mean(probe_latencies)
        cv = np.std(probe_latencies) / mean_ms   # coefficient of variation

        print(f"   Warmup validation: mean={mean_ms:.2f}ms  CV={cv:.3f} "
              f"({'✅ settled' if cv < cv_threshold else '⚠️  still noisy'})")
        return cv < cv_threshold

    # ──────────────────────────────────────────────────────────────────────────
    # Main benchmark loop
    # ──────────────────────────────────────────────────────────────────────────
    def profile_stream(self, test_frame, num_iterations=100, warmup=10):

        # Stage 1: boost GPU clocks before any measurement
        self._boost_gpu_clocks()

        # Stage 2: warmup
        self._warmup(test_frame, warmup)

        # Stage 3: validate warmup settled; extend if needed
        print("🔍 Validating warmup stability...")
        settled = self._validate_warmup(test_frame)
        if not settled:
            print("   ⚠️  Pipeline not settled — running 10 extra warmup frames...")
            self._warmup(test_frame, 10)
            self._validate_warmup(test_frame)

        # Stage 4: clean slate for memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device_id)

        print(f"\n🚀 Executing {num_iterations} benchmark iterations...")
        latencies = []

        with mlflow.start_run(run_name="Edge_Pipeline_Benchmark"):
            mlflow.log_param("hardware_target",  "RTX_4060_Laptop")
            mlflow.log_param("yolo_model",        "yolo11n-seg.pt")
            mlflow.log_param("depth_model",       "Depth-Anything-V2-Small")
            mlflow.log_param("num_iterations",    num_iterations)
            mlflow.log_param("warmup_frames",     warmup)

            for i in range(num_iterations):
                # CUDA event timing: most accurate GPU-side measurement.
                # record() enqueues a timestamp into the CUDA stream.
                # elapsed_time() is only valid AFTER synchronize() drains the stream.
                self.start_event.record()
                _ = self.engine.predict(test_frame)
                self.end_event.record()

                # synchronize() blocks CPU until the GPU stream reaches end_event.
                # This is what makes elapsed_time() safe to call.
                torch.cuda.synchronize()

                iteration_time_ms = self.start_event.elapsed_time(self.end_event)
                latencies.append(iteration_time_ms)
                mlflow.log_metric("latency_ms", iteration_time_ms, step=i)

            # ── Aggregate metrics ────────────────────────────────────────────
            latencies_arr  = np.array(latencies)
            avg_ms         = float(np.mean(latencies_arr))
            p50_ms         = float(np.percentile(latencies_arr, 50))
            p95_ms         = float(np.percentile(latencies_arr, 95))
            p99_ms         = float(np.percentile(latencies_arr, 99))
            std_ms         = float(np.std(latencies_arr))
            throughput_fps = 1000.0 / avg_ms

            peak_vram_mb   = torch.cuda.max_memory_allocated(self.device_id) / (1024 ** 2)
            reserved_vram_mb = torch.cuda.memory_reserved(self.device_id)   / (1024 ** 2)

            mlflow.log_metric("avg_latency_ms",  avg_ms)
            mlflow.log_metric("p50_latency_ms",  p50_ms)
            mlflow.log_metric("p95_latency_ms",  p95_ms)
            mlflow.log_metric("p99_latency_ms",  p99_ms)
            mlflow.log_metric("std_latency_ms",  std_ms)
            mlflow.log_metric("avg_fps",         throughput_fps)
            mlflow.log_metric("peak_vram_mb",    peak_vram_mb)

            print("\n" + "=" * 45)
            print("📊 SYSTEM TELEMETRY SUMMARY")
            print("=" * 45)
            print(f"Iterations        : {num_iterations}")
            print(f"Average FPS       : {throughput_fps:.2f} frames/sec")
            print(f"Average Latency   : {avg_ms:.2f} ms")
            print(f"Std Dev           : {std_ms:.2f} ms")
            print(f"P50 (median)      : {p50_ms:.2f} ms")
            print(f"P95               : {p95_ms:.2f} ms")
            print(f"P99 (tail latency): {p99_ms:.2f} ms")
            print(f"Peak VRAM Active  : {peak_vram_mb:.2f} MB")
            print(f"Total VRAM Locked : {reserved_vram_mb:.2f} MB")
            print("=" * 45)
            print("✅ Telemetry persisted to MLflow tracking server.")

        return {
            "avg_ms":  avg_ms,
            "p99_ms":  p99_ms,
            "fps":     throughput_fps,
            "peak_vram_mb": peak_vram_mb,
        }


if __name__ == "__main__":
    engine = DepthConditionedYOLO(depth_resolution="FAST", warmup_runs=5)

    test_image = cv2.imread('maxresdefault.jpg')
    if test_image is None:
        raise FileNotFoundError("Test image not found. Check the path.")

    profiler = PipelineProfiler(engine)
    profiler.profile_stream(test_image, num_iterations=100, warmup=10)
