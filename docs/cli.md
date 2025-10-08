# CLI Usage

The repository provides two main entry points under `tools/`:

- `tools/train.py`: distributed training driver
- `tools/test.py`: distributed evaluation/inference driver

## Training

```bash
torchpack dist-run -np 8 \
  python tools/train.py <CONFIG> [--run-dir DIR] [key=value ...]
```

- **config**: YAML config path under `configs/`
- `--run-dir`: optional run directory
- Additional overrides can be passed as `key=value` pairs (parsed by `torchpack` configs).

### Examples

- Camera-only detection:

```bash
torchpack dist-run -np 8 python tools/train.py \
  configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml \
  --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

- LiDAR-only detection:

```bash
torchpack dist-run -np 8 python tools/train.py \
  configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml
```

## Evaluation

```bash
torchpack dist-run -np 8 \
  python tools/test.py <CONFIG> <CHECKPOINT> \
  [--eval bbox|map ...] [--out results.pkl] [--show] [--show-dir DIR]
```

### Examples

```bash
torchpack dist-run -np 8 python tools/test.py \
  configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
  pretrained/bevfusion-seg.pth --eval map
```

