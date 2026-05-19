A real-time, edge-optimized computer vision pipeline that fuses YOLO11 object detection with Depth Anything V2 (Small) monocular depth estimation. This system programmatically filters out high-fidelity 2D false positives (e.g., reflections, cardboard cutouts, and life-sized posters) without requiring a physical LiDAR sensor.
🏢 The Business Problem: Alarm Fatigue

In industrial monitoring and autonomous navigation (e.g., smart AGVs or forklift anti-collision systems), standard 2D object detection suffers from a critical safety flaw. A standard vision model will confidently detect a real worker, a life-sized safety poster of a worker, and a reflection of a worker in a glass window with the exact same high confidence score (0.85+).

Attempting to fix this by raising the confidence threshold creates dangerous "false negatives," blinding the system to actual humans in low lighting. This fundamental flaw triggers constant false alarms, leading to operator "alarm fatigue."
🎓 The Academic Origin (University of Southampton)

This project is the industrial, edge-deployed evolution of my 2023 MSc thesis at the University of Southampton, which originally explored depth-aware object detection using YOLOv3 and AdaBins.

While the thesis successfully proved the theoretical viability of sensor-fusion for spatial awareness, it suffered from standard academic deployment bottlenecks:

    Sequential Scripting Bottleneck: The original architecture relied on surface-level scripting to run two models sequentially, forcing massive memory transfers between the GPU and CPU to fuse the outputs via NumPy.

    Unusable Latency: The constant CPU-GPU context switching resulted in extreme latency, rendering the academic system unusable for real-time edge robotics.

🚀 The Industrial Pivot: Graph-Level Concurrency & Zero-Copy VRAM

To transition this academic theory into a 30+ FPS edge-deployable system, this project introduces a Late-Fusion VRAM Bridge optimized for Ada Lovelace architectures (RTX 4000 series / Edge GPUs).

Why not a unified PyTorch nn.Module?
Naive architectural fusion attempts to wrap both a Transformer (Depth) and a CNN (YOLO) into a single PyTorch computational graph. While elegant in Python, this creates monolithic ONNX files that routinely crash the TensorRT C++ compiler due to unsupported multi-modal layer operators, severely limiting edge deployment.

The Solution:
This architecture bypasses the compiler bottleneck by keeping the models completely decoupled at the engine level, but mathematically fused in memory:

    Dual TRT Compilation: YOLO11 and Depth Anything V2 are compiled into distinct, highly optimized TensorRT FP16 .engine files.

    Concurrency: Both engines read the incoming video frame simultaneously using asynchronous CUDA streams.

    Zero-Copy Fusion: All bounding-box-to-depth-map slicing, median background isolation, and variance mathematics are executed using pure tensor operations strictly inside the GPU VRAM. The data never transfers back to the CPU.

📐 The Physics of the Filter

Instead of relying on a visual confidence score, this pipeline mathematically analyzes the physical 3D volume of the detected object:

    3D Objects (Real Person): High depth variance. The physics of a human body dictate that the nose is closer than the shoulders, and the body is closer than the background wall.

    2D Surfaces (Reflections/Posters): Near-zero depth variance. The "person" is perfectly flat and physically flush with the wall.

The system calculates this variance inside the bounding box, isolating the foreground via median depth mapping. If the variance is near zero, the system recognizes a 2D surface and silently suppresses the false alarm.