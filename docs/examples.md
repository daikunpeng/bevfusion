# Examples

## Use `single_gpu_test` for simple inference

```python
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test

cfg = Config.fromfile("configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml")
cfg.data.test.test_mode = True

dataset = build_dataset(cfg.data.test)
dataloader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, "pretrained/bevfusion-det.pth", map_location="cpu")

outputs = single_gpu_test(model, dataloader)
print(type(outputs), len(outputs))
```

## Train with Python API

```python
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.apis import train_model

cfg = Config.fromfile("configs/nuscenes/seg/fusion-bev256d2-lss.yaml")
dataset = build_dataset(cfg.data.train)
model = build_model(cfg.model)
model.init_weights()
train_model(model, [dataset], cfg, distributed=False, validate=True)
```

