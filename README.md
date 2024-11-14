# EdgeTPUParallelism

A project for CS 6501: Hardware Accelerators, taught by Professor Kevin Skadron at the University of Virginia in Spring 2024.

Group Members:
- Param Damle
- Pawan Jayakumar

The full paper and presentation material can be read in the PDF file included in the repository. All the source code is also included, particularly the iPython Notebook to generate .tflite models and the Python scripts to run locally on the machine with a connected CPU, GPU, and USB TPU Accelerator. Here is a basic abstract of the project:

Machine learning inference is becoming an increasingly popular
workload in modern data centers due to the explosion of interest in
large language models. This paper explores optimizing inference
on heterogeneous platforms composed of multiple different types
of accelerated processors. More specifically, we run experiments to
investigate whether the inclusion of Google’s Edge Tensor Process-
ing Unit (Edge TPU) along with an Nvidia Graphics Processing Unit
(GPU) can speed up inference on deep neural networks. We explore
data level parallelism which involves running the same workload
across both the TPU and GPU, as well as model level parallelism
where the model is split into separate portions, each running on a
different device. Through our research, we find that while the TPU
is optimized for small-scale, simple neural network workloads, it
does not scale well and delegating large batch sizes or more intricate
operations to it would result in suboptimal performance.