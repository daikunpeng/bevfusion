
## 训练
训练一个视觉检测版本模型：
```python
torchpack dist-run -np 8 python tools/train.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```
这个命令的含义是使用 `torchpack` 工具在 8 个进程上运行 `tools/train.py` 脚本，使用 `configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml` 配置文件训练一个目标检测模型。该模型使用 `pretrained/swint-nuimages-pretrained.pth` 预训练模型作为其 `backbone`。请注意，这个命令需要在正确的环境中运行，包括正确的 Python 环境、正确的 PyTorch 版本和正确的 CUDA 版本。如果您需要更多关于 PyTorch、CUDA 和 `torchpack` 的信息，请访问 ¹²³.

-np: number of process
python tools/train.py python 是执行的程序，后面是执行的脚本。上述脚本指定了使用8个进程同步进行训练过程，但是如果我们的主机只有一个GPU卡，则无法运行上述命令。它需要8张卡。

按照配置文件 default.yaml 去运行。default.yaml 中存储了很多关于模型的信息，包括但不限于网络结构，数据，学习率，学习算法，输入输出路径等等。

使用的bachbone的预训练权重为 swint-nuimages-pretrained.pth。

