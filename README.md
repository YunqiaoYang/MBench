![MBench](./assets/MBench-Logo.png)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2606.00793)
[![HuggingFace Leaderboard](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-blue)](https://huggingface.co/spaces/study-overflow/MBench_Leaderboard)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-red)](https://huggingface.co/datasets/studyOverflow/MBench-Data)
[![Project Page](https://img.shields.io/badge/Project-Website-green?logo=googlechrome&logoColor=green)](https://peanutup.github.io/MBench-project/)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/study-overflow/MBench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# MBench

[**English**](./README.md) | [**中文**](./README_zh.md)

**MBench** is a benchmark for the **memory capability** of long-video world models. Most existing benchmarks reward single-frame quality or short-horizon prompt following; MBench targets a harder question: when a subject leaves the frame and returns, when the camera departs from a viewpoint and comes back, or when an off-screen physical process keeps evolving, can the model maintain a consistent world state? We decompose this into three orthogonal capability axes — **Entity / Environment / Causal Consistency** — and evaluate them under two complementary settings: **MBench-A** (action-conditioned, for action-conditioned world models) and **MBench-T** (text-segment-conditioned, for long-video text continuation models).

This repository ships the full evaluation pipeline: a contract-driven, plugin-based CLI tool `mbench`, 12 official metric implementations spanning both settings, and an integrated VLM trigger judge for trigger-conditioned scoring.

If you run into problems, please open an [issue](https://github.com/study-overflow/MBench/issues).

## Table of Contents

- [Overview](#overview)
- [Updates](#updates)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Evaluation Dimensions](#evaluation-dimensions)
- [Trigger-Conditioned Scoring](#trigger-conditioned-scoring)
- [Score Schema](#score-schema)
- [Usage Reference](#usage-reference)
- [Output Layout](#output-layout)
- [Citation](#citation)
- [License](#license)

<a name="overview"></a>
## Overview

MBench formalises "long-video memory capability" as three orthogonal capability axes:

- **Entity Consistency** — does a subject's identity, appearance, geometry, and texture stay consistent after it exits the frame and returns?
- **Environment Consistency** — after the camera departs from a viewpoint and returns, does the 3D spatial layout, lighting, and rendering style recover?
- **Causal Consistency** — does an off-screen physical process continue plausibly? Does the model faithfully respond to external instructions (camera actions, segment-level captions)?

Each axis decomposes into 4 sub-dimensions for a total of **12 evaluation dimensions**. Each dimension is computed by a dedicated automatic metric and normalised to the `[0, 1]` range. Where the setting calls for it, the metric is gated by a Vision-Language Model trigger so that **samples that never enter the memory challenge (e.g. completely static videos) cannot inflate model rankings**. See the [Trigger-Conditioned Scoring](#trigger-conditioned-scoring) section below.

The two settings:

| Setting | `dataset_id` | Conditioning | Typical models |
|---|---|---|---|
| **MBench-A** | `mbencha` | Camera action + a single caption sentence | hy_worldplay, matrix_game_3, yume, infinite_world, lingbot_world, matrix_game_2 |
| **MBench-T** | `mbencht` | Five-segment text prompts (`condition_id=text`) | cosmos, longcat, skyreels, longlive, helios, memflow, self_forcing, causal_forcing |

Both settings share the same `samples/{subset}/{sample_id}/` directory layout; at run time only the metrics under the matching prefix are dispatched.

<a name="updates"></a>
## Updates

- **2026-06** — Initial public release: 12 metric implementations, MBench-A / MBench-T data adapter, canonical score schema, integrated VLM trigger judge, contract-validated CLI.

<a name="installation"></a>
## Installation

**Requirements**: Python ≥ 3.10.

```bash
git clone https://github.com/study-overflow/MBench.git
cd MBench
pip install -e .
```

**Additional dependencies**:

| Metric family | Required extras |
|---|---|
| Spatial epipolar / reprojection, object geometry | `numpy`, `opencv-python`, plus a precomputed camera-pose artifact (you can produce it with [DepthAnything v3](https://github.com/bytedance-seed/depth-anything-3)) |
| Rendering style | `torch`, `torchvision`, `lpips`, `transformers` (DINOv2), VGG-19 weights |
| Rendering lighting | `opencv-python` |
| Entity human (identity / appearance) | `insightface` (Buffalo_L), `torch`, `transformers` (DINOv2) |
| Entity object (texture / geometry) | `groundingdino`, `segment-anything-2`, `transformers` (DINOv2) |
| Causal text / action interaction | `open_clip_torch` |
| Causal self-evolution (state / correctness) | An OpenAI-compatible VLM endpoint |

A typical full setup:

```bash
pip install torch torchvision torchaudio   # match your CUDA
pip install opencv-python lpips transformers open_clip_torch insightface
pip install "segment-anything-2 @ git+https://github.com/facebookresearch/segment-anything-2.git"
pip install "groundingdino @ git+https://github.com/IDEA-Research/GroundingDINO.git"
```

Verify the install:

```bash
mbench list-metrics    # should print 23 registered metrics
```

<a name="quick-start"></a>
## Quick Start

Assuming your data is already laid out under `data/MBench-A-Setup/` (see the next section):

```bash
# 1. Validate that the metric's required fields are present on a few samples.
mbench validate data/MBench-A-Setup \
    --metrics mbencha.environment.spatial_epipolar \
    --models my_model --subsets environment --limit 4

# 2. Run a real evaluation: one metric, one model.
mbench eval data/MBench-A-Setup \
    --metrics mbencha.environment.spatial_epipolar \
    --models my_model --subsets environment \
    --output runs/my_first_run
```

For an MBench-T run with a VLM trigger:

```bash
export MBENCH_VLM_BASE_URL="https://your-openai-compat-endpoint.example.com/v1"
export MBENCH_VLM_API_KEY="sk-..."
export MBENCH_VLM_MODEL="gpt-4o"

mbench eval data/MBench-T-Setup \
    --metrics mbencht.entity.human_identity_consistency \
    --models my_model --subsets human \
    --vlm-judge openai-compatible \
    --output runs/my_t_run_with_trigger
```

Without `--vlm-judge`, the same command still runs in **raw-consistency mode** (no API call); the metric still emits `score`, just without trigger gating.

<a name="data-preparation"></a>
## Data Preparation

You can follow this section to lay out the data your own model produces, and evaluate your own model:

```
data/MBench-{A|T}-Setup/
├── dataset.yaml
├── samples/{subset}/{sample_id}/
│   ├── sample.json                   # metadata (object_anchor, caption_segments, …)
│   └── reference.png / source_video.mp4         # optional ground-truth assets
└── models/{model_id}/
    ├── samples.jsonl                 # one line per (sample × condition) generation
    ├── outputs/{subset}/{sample_id}/{condition_id}/video.mp4
    └── artifacts/{subset}/{sample_id}/{condition_id}/da3/results.npz
                                      # only required for spatial / object-geometry / causal-action metrics
```

The four subsets (`human`, `object`, `environment`, `causal`) match the metric subsets one-to-one. Currently `condition_id` is `{action}_{length}` for MBench-A (e.g. `left_then_right_25s`, `forward_then_backward_10s`) and the literal string `text` for MBench-T. You are free to add new conditions of your own (e.g. new camera-trajectory conditions).

A minimal `dataset.yaml`:

```yaml
dataset_id: mbencha           # or mbencht
dataset_name: My MBench-A Setup
version: '1.0'
path_mode: relative_to_dataset_root

subsets:
  environment: {description: Environment / spatial memory.}
  human:       {description: Human entity identity memory.}
  object:      {description: Object entity identity memory.}
  causal:      {description: Causal / interaction memory.}

condition_id:
  pattern: "{action}_{length}"

paths:
  output_video: models/{model_id}/outputs/{subset}/{sample_id}/{condition_id}/video.mp4
  artifact_da3: models/{model_id}/artifacts/{subset}/{sample_id}/{condition_id}/da3/results.npz
```

A `samples.jsonl` line per generated rollout:

```json
{"item_id": "human:my_sample_001:left_then_right_25s", "dataset_id": "mbencha", "subset": "human", "sample_id": "my_sample_001", "condition_id": "left_then_right_25s", "model_id": "my_model", "media": {"videos": [{"path": "outputs/human/my_sample_001/left_then_right_25s/video.mp4", "role": "generated"}]}, "artifacts": {"da3": {"path": "artifacts/human/my_sample_001/left_then_right_25s/da3/results.npz"}}}
```

**`sample.json` required fields by subset**:

- `object`: T side requires `metadata.object_card`, A side requires `metadata.object_anchor`, describing the target object.
- `causal`: T side requires `metadata.caption_segments`, A side requires `annotations.action` or `metadata.caption`.
- `environment` / `human`: only `media.videos` is required; segment-level captions help the T-side trigger judge.

DA3 camera-pose artifacts are produced externally — see [DA3](https://github.com/lemmingforever/DA3); MBench only consumes the resulting `results.npz`.

<a name="evaluation-dimensions"></a>
## Evaluation Dimensions

The 12 dimensions and their registration names:

| Axis | Sub-dimension | MBench-A | MBench-T |
|---|---|---|---|
| **Entity** | Human Identity | `mbencha.entity.human_identity_consistency` | `mbencht.entity.human_identity_consistency` |
| | Human Appearance | `mbencha.entity.human_appearance_consistency` | `mbencht.entity.human_appearance_consistency` |
| | Object Texture | `mbencha.entity.object_texture_consistency` | `mbencht.entity.object_texture_consistency` |
| | Object Geometry | `mbencha.entity.object_geometry_consistency` | `mbencht.entity.object_geometry_consistency` |
| **Environment** | Spatial Epipolar | `mbencha.environment.spatial_epipolar` | `mbencht.environment.spatial_epipolar` |
| | Spatial Reprojection | `mbencha.environment.spatial_reprojection` | `mbencht.environment.spatial_reprojection` |
| | Rendering Style | `mbencha.environment.rendering_style` | `mbencht.environment.rendering_style` |
| | Rendering Lighting | `mbencha.environment.rendering_lighting` | `mbencht.environment.rendering_lighting` |
| **Causal** | Action Interaction | `mbencha.causal.camera_interaction` | – |
| | Text Interaction | `mbencha.causal.prompt_interaction` | `mbencht.causal.prompt_interaction` |
| | Self-Evolution: State | `mbencha.causal.state_progress` | `mbencht.causal.state_progress` |
| | Self-Evolution: Correctness | `mbencha.causal.progress_correctness` | `mbencht.causal.progress_correctness` |

`mbench list-metrics` prints the live registry with one-line descriptions per metric.

<a name="trigger-conditioned-scoring"></a>
## Trigger-Conditioned Scoring

A model can game a memory benchmark by generating static, overly conservative content — never engaging with the challenge — and walk away with an inflated consistency score. MBench prevents this by **decoupling reliability from coverage**:

- **Memory Reliability** $S^{\text{rel}}$ — average consistency, computed **only on samples that actually triggered the memory challenge**.
- **Trigger Coverage** $C^{\text{trig}}$ — fraction of samples that triggered the memory challenge.
- **M-Score** — the harmonic mean of the two: $\text{M-Score} = 2 \cdot S^{\text{rel}} \cdot C^{\text{trig}} / (S^{\text{rel}} + C^{\text{trig}})$.

**Where the trigger fires**:

| Setting | Trigger source | Default behaviour |
|---|---|---|
| **MBench-T** | A VLM judge over 8 uniformly sampled frames + caption segments | OFF; opt in by passing `--vlm-judge openai-compatible` |
| **MBench-A** | Not used (action-conditioned rollouts deterministically actuate the memory challenge — paper §App) | Always OFF, even if `--vlm-judge` is passed |

Configure the VLM via CLI flags or environment variables:

```bash
export MBENCH_VLM_BASE_URL="..."     # OpenAI-compatible /v1 endpoint
export MBENCH_VLM_API_KEY="..."
export MBENCH_VLM_MODEL="..."        # any vision-capable model id

# or pass per-run:
mbench eval ... --vlm-judge openai-compatible \
    --vlm-base-url "$MBENCH_VLM_BASE_URL" \
    --vlm-api-key  "$MBENCH_VLM_API_KEY" \
    --vlm-model    "$MBENCH_VLM_MODEL"
```

For ablation runs, pass `--ignore-trigger` to force the trigger gate off even when `--vlm-judge` is set.

<a name="score-schema"></a>
## Score Schema

Regardless of A or T, regardless of internal computation, every metric emits the **same** score shape at the unit, item, and summary layers:

| Key | Range | When present | Meaning |
|---|---|---|---|
| `score` | [0, 1] | Always (when valid) | The normalised score. |
| `score_raw` | native unit | Distance- or VLM-1-5-based metrics | Raw signal before normalisation; omitted when `raw == score`. |
| `triggered` | {0.0, 1.0} | T-side metrics only | Whether the VLM trigger fired (0 → sample skipped from the reliability average). |
| `trigger_score` | [0, 1] | T-side metrics only | Normalised VLM trigger confidence. |

Examples (after `mbench eval`):

```json
// mbencha.environment.spatial_epipolar  units.jsonl line
{"unit_id": "...__f12_f48", "scores": {"score": 0.5523, "score_raw": 2.964}}

// mbencht.causal.progress_correctness  summary.json group
{"score": 0.74, "score_raw": 3.7, "triggered": 1.0, "trigger_score": 0.6}

// mbencht.* untriggered sample
{"triggered": 0.0, "trigger_score": 0.2}
```

<a name="usage-reference"></a>
## Usage Reference

### List registered metrics

```bash
mbench list-metrics
```

### Validate (strict)

```bash
mbench validate <source> --metrics <csv> [--models …] [--subsets …] [--limit N]
```

Any contract violation exits non-zero.

### Eval (lenient — skips bad items, runs the rest)

```bash
mbench eval <source> --metrics <csv> --output runs/<run_name>
    [--models <csv>] [--subsets <csv>] [--conditions <csv>]
    [--sample-ids <csv>] [--limit N]
    [--vlm-judge openai-compatible
       --vlm-base-url <URL> --vlm-api-key <KEY> --vlm-model <ID>
       --vlm-n-frames 8 --vlm-max-retries 3]
    [--workers N]                 # parallel across items
    [--ignore-trigger]            # bypass VLM trigger gate (ablation only)
    [--no-progress-bar] [--quiet] [--no-log-file]
```

### Prepare auxiliary artifacts (never auto-invoked during eval)

```bash
mbench prepare <source> --producer <name> --output artifacts/
```

### Python API

```python
from mbench.core.pipeline import run_eval
from mbench.core.registry import metric_registry, adapter_registry, aggregator_registry
import mbench.cli; mbench.cli._bootstrap()    # register all built-in plugins

adapter = adapter_registry.get("mbench-dir")
items = adapter.load("data/MBench-A-Setup", subsets=["environment"], limit=4)
metric = metric_registry.get("mbencha.environment.spatial_epipolar")
aggregator = aggregator_registry.get("mean")

summary = run_eval(items, [metric], aggregator, "runs/python_api_demo")
```

<a name="output-layout"></a>
## Output Layout

```
runs/<run_name>/
├── items.jsonl              # one line per evaluated sample (item_id, model_id)
├── eval.log                 # per-item progress log (timestamps, scores, errors)
└── metrics/<metric_name>/
    ├── units.jsonl          # one UnitResult per evaluation unit (frame pair, track, segment, …)
    ├── items.jsonl          # one ItemResult per sample (aggregated from units)
    └── summary.json         # SummaryResult grouped by model_id
```

`UnitResult.metadata['raw_scores']` preserves the metric's original internal fields (the ones dropped by the canonical projection), useful for after-the-fact debugging or re-deriving alternative score keys offline.

<a name="citation"></a>
## Citation

If MBench helps your research, please cite us:

```bibtex
@article{zhang2026mbench,
  title         = {MBench: A Comprehensive Benchmark on Memory Capability for Video World Models},
  author        = {Zhang, Shengjun and Zhang, Zhang and Huang, Simin and Tang, Zhenyu and Wang, Hanyang and Dai, Chensheng and Chen, Min and Li, Yifan and Li, Yuxin and Chen, Yingjie and Liu, Hao and Li, Chen and Duan, Yueqi},
  year          = {2026},
  eprint        = {2606.00793},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2606.00793}
}
```

<a name="license"></a>
## License

The code is released under the [MIT License](LICENSE).
