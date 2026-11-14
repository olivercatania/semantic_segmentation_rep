## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Train one single-organ model

```bash
uv run python -m scripts.train_single \
    --architecture deeplabv3 --organ liver --epochs 100
```

Replace deeplabv3 with segformer to use SegFormer.

## Train all models

Trains 22 single-organ models and 2 combined models.

```bash
uv run python -m scripts.train_all --epochs 100
```

## Train one combined model

Trains one shared-encoder model covering all eleven organs.

```bash
uv run python -m scripts.train_multi \
    --architectures deeplabv3 --epochs 100 --batch-size 1
```
