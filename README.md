# EdgeDI

This is the source code for our paper: **Joint Architecture Design and Workload Partitioning for DNN Inference on Industrial IoT Clusters**. A brief introduction of this work is as follows:

> The advent of Deep Neural Networks (DNNs) has empowered numerous computer-vision applications. Due to the high computational intensity of DNN models, as well as the resource constrained nature of Industrial Internet-of-Things (IIoT) devices, it is generally very challenging to deploy and execute DNNs efficiently in the industrial scenarios. Substantial research has focused on model compression or edge-cloud offloading, which trades off accuracy for efficiency or depends on high-quality infrastructure support, respectively. In this article, we present EdgeDI, a framework for executing DNN inference in a partitioned, distributed manner on a cluster of IIoT devices. To improve the inference performance, EdgeDI exploits two key optimization knobs, including: (1) Model compression based on deep architecture design, which transforms the target DNN model into a compact one that reduces the resource requirements for IIoT devices without sacrificing accuracy; (2) Distributed inference based on adaptive workload partitioning, which achieves high parallelism by adaptively balancing the workload distribution among IIoT devices under heterogeneous resource conditions. We have implemented EdgeDI based on PyTorch, and evaluated its performance with the NEU-CLS defect classification task and two typical DNN models (i.e., VGG and ResNet) on a cluster of heterogeneous Raspberry Pi devices. The results indicate that the proposed two optimization approaches significantly outperform the existing solutions in their specific domains. When they are well combined, EdgeDI can provide scalable DNN inference speedups that are very close to or even much higher than the theoretical speedup bounds, while still maintaining the desired accuracy.

It is published by ACM Transactions on Internet Technology (ACM ToIT). You can also refer to another relevant work [EdgeLD](https://github.com/fangvv/EdgeLD) from our team.

## Required software

PyTorch

## Contact

Wenyuan Xu (19120419@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
