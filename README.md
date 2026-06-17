# EdgeDI

This is the source code for our paper: **Joint Architecture Design and Workload Partitioning for DNN Inference on Industrial IoT Clusters**. A brief introduction of this work is as follows:

The advent of Deep Neural Networks (DNNs) has empowered numerous computer-vision applications. Due to the high computational intensity of DNN models, as well as the resource constrained nature of Industrial Internet-of-Things (IIoT) devices, it is generally very challenging to deploy and execute DNNs efficiently in the industrial scenarios. Substantial research has focused on model compression or edge-cloud offloading, which trades off accuracy for efficiency or depends on high-quality infrastructure support, respectively. In this article, we present EdgeDI, a framework for executing DNN inference in a partitioned, distributed manner on a cluster of IIoT devices. To improve the inference performance, EdgeDI exploits two key optimization knobs, including: (1) Model compression based on deep architecture design, which transforms the target DNN model into a compact one that reduces the resource requirements for IIoT devices without sacrificing accuracy; (2) Distributed inference based on adaptive workload partitioning, which achieves high parallelism by adaptively balancing the workload distribution among IIoT devices under heterogeneous resource conditions. We have implemented EdgeDI based on PyTorch, and evaluated its performance with the NEU-CLS defect classification task and two typical DNN models (i.e., VGG and ResNet) on a cluster of heterogeneous Raspberry Pi devices. The results indicate that the proposed two optimization approaches significantly outperform the existing solutions in their specific domains. When they are well combined, EdgeDI can provide scalable DNN inference speedups that are very close to or even much higher than the theoretical speedup bounds, while still maintaining the desired accuracy.

深度神经网络（DNN）的出现推动了众多计算机视觉应用的发展。由于DNN模型的高计算强度以及工业物联网（IIoT）设备资源受限的特性，在工业场景中高效部署和执行DNN通常面临巨大挑战。现有研究主要聚焦模型压缩或边缘-云端卸载方案，前者以精度换取效率，后者则依赖于高质量的基础设施支持。本文提出EdgeDI框架，可在IIoT设备集群上以分区分布式方式执行DNN推理。为提升推理性能，EdgeDI采用两个关键优化维度：（1）基于深度架构设计的模型压缩技术，将目标DNN模型转换为紧凑版本，在保持精度的同时降低IIoT设备资源需求；（2）基于自适应工作负载划分的分布式推理机制，通过在异构资源条件下动态平衡IIoT设备间的工作负载分配来实现高度并行化。我们基于PyTorch实现了EdgeDI，并采用NEU-CLS缺陷分类任务和两种典型DNN模型（VGG与ResNet）在异构树莓派设备集群上评估其性能。结果表明：所提出的两种优化方法在各自领域显著优于现有方案。当二者协同工作时，EdgeDI可提供接近甚至远超理论加速上限的可扩展推理加速，同时保持预期精度。

This work has been published by ACM Transactions on Internet Technology (ACM ToIT). Click [here](https://dl.acm.org/doi/10.1145/3571815) for our paper online. You can also refer to another relevant work [EdgeLD](https://github.com/fangvv/EdgeLD) from our team.

## Required software

- Python 3.7+
- PyTorch (1.x)
- torchvision
- NumPy
- torchstat
- memory_profiler
- psutil
- Matplotlib (optional, for visualization)

## Project Structure

```
EdgeDI/
├── VGG13.py            # Vanilla VGG-13 (features + classifier)
├── VGG13Block.py       # VGG-13 variant with Feature-enhancement (SE-like) blocks
├── VGG16.py            # Vanilla VGG-16
├── VGG16Block.py       # VGG-16 variant with Feature-enhancement blocks
├── ResNet18.py         # Vanilla ResNet-18
├── ResNet18Block.py    # ResNet-18 variant with Feature-enhancement blocks
├── ResNet50.py         # Vanilla ResNet-50 (Bottleneck)
├── ResNet50Block.py    # ResNet-50 variant with Feature-enhancement blocks
├── PruningVGG1316.py   # L1-norm channel pruning for VGG-13/16
├── PruningResnet18.py  # L1-norm channel pruning for ResNet-18
├── PruningResnet50.py  # L1-norm channel pruning for ResNet-50
├── OptimalSplit.py     # 1-D adaptive workload partitioner for heterogeneous IIoT devices
├── init_model.py       # FLOPs/statistics tool (used by OptimalSplit)
├── NEUCLSDataLoad.py   # NEU-CLS surface-defect dataset loader
├── train.py            # Training entry (VGG-13 by default, SGD + step LR)
├── test.py             # Test entry (VGG-13 pruned variant)
├── time_test.py        # Per-block inference latency benchmarking
├── memory_test.py      # Per-block memory profiling
└── README.md
```

## Core Modules

### DNN Backbones (`VGG13.py` / `VGG16.py` / `ResNet18.py` / `ResNet50.py`)

Standard PyTorch implementations of the four target models. Each class accepts an optional `layer_nums` list so that the channel width of every conv layer can be customized after pruning, without modifying the source code.

| Model | Default `layer_nums` | Block | Notes |
|---|---|---|---|
| `vgg13` | `[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M']` | Conv-BN-ReLU + MaxPool | 13 weight layers |
| `vgg16` | `[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']` | Conv-BN-ReLU + MaxPool | 16 weight layers |
| `resnet18` | `[64,64,64,128,128,256,256,512,512]` | BasicBlock (3×3, 3×3) | First conv + 8 BasicBlocks |
| `resnet50` | `[64,256,256,256,512,512,512,512,1024,1024,1024,1024,1024,1024,2048,2048,2048]` | Bottleneck (1×1, 3×3, 1×1) | First conv + 16 Bottlenecks |

All `forward` methods execute only the convolutional features (the classifier head is not used), since the goal is per-block feature computation and inter-device transfer, not end-to-end classification accuracy.

### Block Variants (`VGG13Block.py` / `VGG16Block.py` / `ResNet18Block.py` / `ResNet50Block.py`)

Each `*Block` class re-implements the corresponding backbone with a channel-wise **Feature Enhancement (FE)** module inserted at every block boundary. The FE module is a SE-like gating unit that re-weights feature maps before they are shipped to the next device, so that accuracy lost by aggressive channel pruning can be partially recovered without adding inference cost on the IIoT side.

```python
class Feature_enhancement(nn.Module):
    # Global AvgPool -> FC(channel->channel) -> Sigmoid -> channel-wise scale
    def forward(self, x):
        out = self.globalAvgpool(x)
        out = self.fc(out.view(out.size(0), -1))
        out = self.sigmoid(out).view(...)
        return out * x
```

The `*Block` classes are used as the deployable model after pruning, and they consume the same `layer_nums` list as their vanilla counterparts.

### Pruning (`PruningVGG1316.py` / `PruningResnet18.py` / `PruningResnet50.py`)

Channel pruning driven by the **L1-norm** of each convolutional filter. The procedure is identical for all three scripts:

1. Build the original model with the default `original_layer` widths.
2. Load pre-trained weights from `./SaveInfo/<Model>/Para/...`.
3. For every `nn.Conv2d` layer, compute `weights_l1 = sum(|w|, axis=(1,2,3))` and sort filters ascending.
4. Iteratively remove the `k` filters with the smallest L1-norm per layer, fine-tune for one epoch, and re-evaluate on the test set. Stop when test accuracy drops below `base_min_acc`.
5. Persist the surviving channel widths and the new weights to `SaveInfo/`.

| Hyperparameter | VGG-13/16 | ResNet-18 | ResNet-50 |
|---|---|---|---|
| `base_min_acc` (accuracy floor, %) | 98.00 | 97.00 | 96.67 |
| Pruning criterion | L1-norm of `Conv2d.weight` | L1-norm of `Conv2d.weight` | L1-norm of `Conv2d.weight` |
| Search strategy | Iterative per-layer | Iterative per-layer | Iterative per-layer |

Key functions: `_computer_L1_value()` computes per-filter L1 norms, `_get_minValue_index()` returns the indices of the `n` smallest filters, and `_get_layer_name()` maps a layer index back to its parameter name for weight I/O.

### Optimal Workload Partitioning (`OptimalSplit.py`)

The runtime partitioner that decides how to split a feature map across a heterogeneous IIoT cluster (in the paper, Raspberry Pi 3B+ and Pi 4B).

**Core data:**

| Symbol | Meaning |
|---|---|
| `device[i]` | Device type, e.g. `'Pi3B+'` or `'Pi4B'` |
| `dc[i]` | Relative compute capacity (Pi3B+ = 1, Pi4B = 1.5) |
| `bandwidth[i]` | Link bandwidth to device `i` (Mbps) |
| `feature_size = [H, W]` | Current feature map size to be split |
| `model_type` | Channel widths of the conv block to run |
| `stride_type` | Stride of the corresponding conv layers |
| `w[i]` | Height assigned to device `i` (partition result) |

**Key methods:**

- `Get_flops(feature_size, model_type, stride_type)` — Shells out to `init_model.py` with `torchstat` to obtain the FLOPs of a single conv block at the given feature size.

- `read_liner_model(name)` — Loads the pre-fitted linear model (`liner_model/<device>.txt`) that maps FLOPs to inference latency for a given device type.

- `Predicted_time(...)` — Returns the predicted wall-clock time for device `i`, including compute time (from the linear model) plus the upload/download time of the split feature maps over `bandwidth[i]`.

- `Optimal_One_Dimensional_Partition(device, feature_size, model_type, stride_type, bandwidth)` — Greedy 1-D partitioner. Initializes the split proportionally to `dc`, then repeatedly moves one row of work from the slowest device to the fastest one as long as the latency gap keeps shrinking. Returns the final width list `w` that balances latency across the cluster.

### FLOPs Helper (`init_model.py`)

A minimal `init_model` class that builds a sequential stack of `Conv2d → ReLU → BatchNorm2d` from the `model_type` channel list. Used only as a target for `torchstat.stat()` so that `OptimalSplit` can query FLOPs without instantiating the full backbone.

CLI usage:

```bash
python init_model.py --fm=200-200 --mt=3-64-64-64 --st=1-1-1
```

### Dataset (`NEUCLSDataLoad.py`)

`NEUCLASSDATA` wraps `torchvision.datasets.ImageFolder` on the NEU-CLS surface-defect corpus (`./Data/train_data`, `./Data/test_data`). Applies `ToTensor` + `Normalize([0.5]*3, [0.5]*3)` and returns `(train_dataset, test_dataset)`. The dataset contains 6 defect categories and is used for both training (`train.py`) and accuracy evaluation during pruning.

### Training, Testing, and Profiling

| Script | Purpose |
|---|---|
| `train.py` | Trains VGG-13 on NEU-CLS. SGD with momentum 0.9, initial LR 1e-3, step decay per epoch, `BATCH_SIZE=20`, `NUM_EPOCHS=50`, Cross-Entropy loss. Saves the best checkpoint to `SaveInfo/`. |
| `test.py` | Loads a pruned VGG-13 variant (`vgg13Block`), evaluates on the NEU-CLS test split, and appends per-iteration accuracy to `SaveInfo/VGG13/TEST_ACC/`. |
| `time_test.py` | Runs the pruned block `N` times on a fixed input, reports `max / min / trimmed-mean` latency. |
| `memory_test.py` | Wraps the block with `memory_profiler.profile` to track per-line memory consumption during a single forward pass. |

## Usage

```bash
# Install dependencies
pip install torch torchvision numpy torchstat memory_profiler psutil

# 1. (Optional) Train the full model to obtain initial weights
python train.py

# 2. Channel-prune a backbone (L1-norm, accuracy-preserving)
python PruningVGG1316.py     # or PruningResnet18.py / PruningResnet50.py

# 3. Evaluate the pruned model
python test.py

# 4. Compute the optimal workload split for a given cluster
python OptimalSplit.py

# 5. Benchmark per-block latency and memory on the target IIoT device
python time_test.py
python memory_test.py
```

## Citation

If you find EdgeDI useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@article{fang2023joint,
    title={Joint Architecture Design and Workload Partitioning for DNN Inference on Industrial IoT Clusters},
    author={Fang, Weiwei and Xu, Wenyuan and Yu, Chongchong and Xiong, Neal N},
    journal={ACM Transactions on Internet Technology},
    volume={23},
    number={1},
    pages={1--21},
    year={2023},
    publisher={ACM New York, NY}
}
```

## For more

We have another work on [EdgeLD](https://github.com/fangvv/EdgeLD) for your reference, and you may also be interested in our DRL-based edge-computing work [UAV-DDPG](https://github.com/fangvv/UAV-DDPG) and [VN-MADDPG](https://github.com/fangvv/VN-MADDPG).

## Contact

Wenyuan Xu (19120419@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
