# Depth-Conditioned YOLO11: False Positive Reduction at the Edge

A real-time, edge-optimized computer vision pipeline that fuses YOLO11 object detection with Depth Anything V2 (Small) monocular depth estimation to programmatically filter out 2D false positives.

## The Narrative:
This project is an architectural evolution of my 2023 MSc thesis at the University of Southampton, which originally explored depth-aware object detection using YOLOv3 and AdaBins. 

While the academic thesis proved the theoretical viability of sensor-fusion for spatial awareness, it suffered from critical production bottlenecks. The original work lacked true architectural integration; it relied on surface-level scripting to run two models sequentially, forcing heavy memory transfers between the GPU and CPU to fuse the outputs via NumPy. This resulted in massive latency, rendering the system unusable for real-time edge hardware.

**The Architectural Leap (2026 Iteration):**
This project Abandons surface-level scripting for a **Deep Fusion** approach. Instead of treating detection and depth as isolated scripts, this system wraps YOLO11n and Depth Anything V2 into a single, custom PyTorch `nn.Module`. 

All bounding-box-to-depth-map slicing and variance math is executed natively via PyTorch tensor operations strictly within the GPU VRAM (Zero-Copy Inference). By engineering this integration at the computational graph level—and compiling the entire unified architecture into a TensorRT FP16 engine—this pipeline successfully transitions the heavy academic concept into a high-speed, 30+ FPS edge-deployable system.

## The Business Problem: Alarm Fatigue
In industrial monitoring and autonomous navigation (e.g., forklift anti-collision systems), standard 2D object detection suffers from critical false positives. A YOLO model will confidently detect a real worker, a life-sized poster of a worker, and a reflection of a worker in a glass window with the exact same accuracy. This triggers constant false alarms, leading to operator "alarm fatigue."

**The Solution:**
Instead of relying on expensive 3D LiDAR, this pipeline mathematically analyzes the isolated depth tensor of the detected object:
* **3D Objects (Real Person):** High depth variance (the nose is closer than the shoulders; the person is closer than the background).
* **2D Surfaces (Reflections/Posters):** Near-zero depth variance (the "person" is perfectly flat and flush with the wall).

The system filters out bounding boxes with low variance directly within the neural architecture, ensuring alerts are only triggered for physical, 3D obstacles.