# 🛡️ Zero-Shot Spatial Verification Engine (Edge AI)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org)
[![TensorRT](https://img.shields.io/badge/NVIDIA-TensorRT-76B900.svg)](https://developer.nvidia.com/tensorrt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A VRAM-optimized, split-stream Computer Vision architecture for eliminating 2D spatial spoofs, specular reflections, and environmental hallucinations in industrial tracking pipelines.**

![System Demo Placeholder](https://via.placeholder.com/800x400.png?text=[Insert+Your+Live+Tracking+GIF+Here])

---

## 🧬 Project Origins & The Economic Premise
Originating from theoretical research conducted during my **MSc in Artificial Intelligence at the University of Southampton**, this architecture has been heavily refactored into a real-time, production-grade edge system. 

**The Economic Thesis:** Industrial autonomy (AGVs) and perimeter security currently rely on $5,000+ LiDAR and Time-of-Flight (ToF) sensors for perfect 3D mapping. However, these systems fail economically at scale and struggle with glass. This architecture democratizes 3D spatial awareness by extracting volumetric physics purely from standard RGB streams, achieving LiDAR-like spoof rejection on sub-$300 edge GPUs (e.g., RTX 4060, NVIDIA Jetson).

## ⚙️ The Engineering Problem
Industrial automation and retail analytics rely heavily on 2D object detection feeding into temporal trackers (e.g., DeepSORT, ByteTrack). However, these pipelines critically fail when encountering 2D artifacts. 
* **The Ghost ID Crisis:** Specular reflections on glass or polished floors cause trackers to assign IDs to "ghost" entities, leading to track fragmentation and corrupted analytics.
* **The False Stop Liability:** Autonomous Guided Vehicles (AGVs) trigger emergency brakes when 2D sensors detect human safety posters, wall murals, or reflections on shrink-wrapped pallets, causing massive operational downtime.

## 🚀 The Solution: Split-Stream Architecture
This project introduces a **Dual-Engine Spatial Verification Pipeline** that intercepts detections *before* they corrupt the tracking loop. By fusing YOLO11 Instance Segmentation with Depth Anything V2 natively in VRAM, the system calculates relative environmental physics without the PCIe bottleneck.

### Hardware Optimization & Zero-Copy Math
1. **Split-Stream Processing:** Maintains standard YOLO bounding box coordinates `[x1, y1, x2, y2]` for legacy tracking loops, while routing high-resolution boolean segmentation masks to the 3D verification engine.
2. **Zero-Copy VRAM Execution:** YOLO masks and Hugging Face depth tensors are cross-multiplied entirely on the GPU (`cuda:0`). By utilizing native boolean indexing (`depth_tensor[binary_mask]`), no data crosses to the CPU for NumPy processing until the final logic gate.
3. **Dynamic Halo Calibration:** Eliminates brittle, hardcoded variance thresholds. The system uses PyTorch 2D Max Pooling to mathematically dilate the object's mask natively on the GPU, creating an environmental "Halo" ring. The system self-calibrates to any lighting or background texture by calculating the physical distance between the object and this immediate background.

## 🚧 Overcoming Monocular Hallucination (Dual-Gate Physics)
During physical stress testing, a critical limitation of Vision Transformers was isolated: high-definition 2D cutouts with printed shadows trick monocular models into hallucinating 3D volume. To solve this limitation in software, the architecture employs a **Dual-Gate Physics** approach:

* **Gate 1 - Internal Volume (`has_volume`):** Measures the isolated depth variance of the silhouette. Evaluates if the object possesses internal 3D curves. *(Successfully filters out flat specular reflections and glass noise).*
* **Gate 2 - Environmental Step-Off (`pops_out`):** Compares the object's median depth to the dilated Halo mask. *(Successfully filters out hallucinated depth printed flush against a wall, such as posters or murals).*

---

## 📊 System Telemetry & Benchmarks
*Tested on an NVIDIA RTX 4060 (8GB) utilizing CUDA-accelerated PyTorch. Metrics generated via MLflow and CUDA Events.*

| Metric | Performance | Business Impact |
| :--- | :--- | :--- |
| **Throughput (FPS)** | `35.79 FPS` | Exceeds real-time 30 FPS camera hardware limits. |
| **Average Latency** | `27.94 ms` | Instantaneous spatial verification. |
| **P99 Latency (Tail)** | `30.86 ms` | Highly stable execution with no garbage-collection spikes, ensuring AGV braking safety. |
| **Peak VRAM Active** | `310.42 MB` | Extremely lightweight; leaves 95% of GPU memory free for heavy temporal tracking loops. |

---

## 🛠️ Quickstart & Installation

To ensure strict reproducibility and prevent CPU-bottlenecking, you must install the CUDA-accelerated PyTorch wheels specific to your hardware before installing the remaining pipeline dependencies.

```bash
# 1. Clone the repository
git clone [https://github.com/yourusername/zero-shot-spatial-engine.git](https://github.com/yourusername/zero-shot-spatial-engine.git)
cd zero-shot-spatial-engine

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install PyTorch with CUDA 12.1 support (Required for RTX 40-series/Ampere+)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 4. Install pipeline dependencies
pip install -r requirements.txt