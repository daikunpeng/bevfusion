---
title: Quickstart
---

# Quickstart

This guide helps you set up the environment and run basic training and evaluation.

## Prerequisites

- Python >= 3.8, < 3.9
- CUDA-capable GPUs
- Recommended: Docker with NVIDIA runtime

Install the repository in development mode:

```bash
python setup.py develop
```

## Datasets

Follow the nuScenes preparation instructions linked in the repository `README.md`. Ensure `data/nuscenes` is prepared as required.

## Evaluate a pretrained model

```bash
./tools/download_pretrained.sh
torchpack dist-run -np 8 \
  python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth --eval bbox
```

## Train a model

Camera-only detection example:

```bash
torchpack dist-run -np 8 \
  python tools/train.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

## Minimal Python API example

```python
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from mmdet3d.apis import train_model

cfg = Config.fromfile("configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml")
dataset = build_dataset(cfg.data.train)
model = build_model(cfg.model)
model.init_weights()
train_model(model, [dataset], cfg, distributed=False, validate=True)
```

