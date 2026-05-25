import torch
import time
import mlflow
import numpy as np
from core_engine import DepthConditionedYOLO

class PipelineProfiler:
    def __init__(self, engine, device_id=0):
        self.engine = engine
        self.device = f"cuda:{device_id}"
        
        # GPU timing uses CUDA events
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        
    def profile_stream(self, test_frame, num_iterations=100, warmup=10):
        print(f"🔥 Warming up GPU pipeline for {warmup} frames...")
        for _ in range(warmup):
            _ = self.engine.predict(test_frame)
            
        print(f"🚀 Executing {num_iterations} benchmark iterations...")
        latencies = []
        
        # Start MLflow run to persist these metrics
        with mlflow.start_run(run_name="Edge_Pipeline_Benchmark"):
            mlflow.log_param("hardware_target", "RTX_4060_Class")
            mlflow.log_param("yolo_model", "yolo11n-seg.pt")
            mlflow.log_param("depth_model", "Depth-Anything-V2-Small")
            
            # Flush cache to get accurate memory baseline
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            for i in range(num_iterations):
                self.start_event.record()
                
                # Execute your pipeline
                _ = self.engine.predict(test_frame)
                
                self.end_event.record()
                
                # Synchronize forces the CPU to wait for the GPU to actually finish the math
                torch.cuda.synchronize() 
                
                # CUDA events return time in milliseconds
                iteration_time_ms = self.start_event.elapsed_time(self.end_event)
                latencies.append(iteration_time_ms)
                
                # Log continuous metrics to MLflow
                mlflow.log_metric("latency_ms", iteration_time_ms, step=i)
            
            # --- Compile Final Hardware Telemetry ---
            avg_latency_ms = np.mean(latencies)
            p99_latency_ms = np.percentile(latencies, 99) # The metric FAANG cares about most
            throughput_fps = 1000.0 / avg_latency_ms
            
            peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            reserved_vram_mb = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            
            # Log aggregate metrics
            mlflow.log_metric("avg_fps", throughput_fps)
            mlflow.log_metric("p99_latency_ms", p99_latency_ms)
            mlflow.log_metric("peak_vram_mb", peak_vram_mb)
            
            print("\n" + "="*40)
            print("📊 SYSTEM TELEMETRY SUMMARY")
            print("="*40)
            print(f"Average FPS       : {throughput_fps:.2f} frames/sec")
            print(f"Average Latency   : {avg_latency_ms:.2f} ms")
            print(f"P99 Latency (Tail): {p99_latency_ms:.2f} ms")
            print(f"Peak VRAM Active  : {peak_vram_mb:.2f} MB")
            print(f"Total VRAM Locked : {reserved_vram_mb:.2f} MB")
            print("="*40)
            print("✅ Telemetry persisted to MLflow tracking server.")

if __name__ == "__main__":
    import cv2
    
    # Initialize your core engine
    engine = DepthConditionedYOLO()
    
    # Load a test frame (e.g., your poster or reflection test image)
    test_image = cv2.imread('test1.webp')
    
    # Run the profiler
    profiler = PipelineProfiler(engine)
    profiler.profile_stream(test_image)
