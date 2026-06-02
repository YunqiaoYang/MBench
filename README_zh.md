![MBench](./assets/MBench-Logo.png)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2606.00793)
[![HuggingFace Leaderboard](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-blue)](https://huggingface.co/spaces/study-overflow/MBench_Leaderboard)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-red)](https://huggingface.co/datasets/studyOverflow/MBench-Data)
[![Project Page](https://img.shields.io/badge/Project-Website-green?logo=googlechrome&logoColor=green)](https://peanutup.github.io/MBench-project/)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/study-overflow/MBench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# MBench

[**English**](./README.md) | [**中文**](./README_zh.md)

**MBench** 是一套面向长视频世界模型的**记忆能力**评测基准。当前主流基准多以单帧画质或短时 prompt 跟随作为奖励信号，而 MBench 关注的是更难的问题：当主体离开画面后再出现、相机偏离视点后再回到原处、或者画面外有物理过程在持续演化时，模型能否维持一致的世界状态。我们将这一问题分解为三条正交的能力线 —— **Entity / Environment / Causal Consistency**，并在两种互补设定下评测：**MBench-A**（动作条件，针对 action-conditioned 世界模型）与 **MBench-T**（文本分段条件，针对长视频文本续写模型）。

本仓库提供完整的评测流水线：契约驱动、插件化的命令行工具 `mbench`、横跨两种设定的 12 个官方 metric 实现、与 metric 内嵌的 VLM trigger judge（用于 trigger-conditioned 打分）。

如有问题，请在 [GitHub Issues](https://github.com/study-overflow/MBench/issues) 中提出。

## 目录

- [概览](#overview)
- [更新日志](#updates)
- [安装](#installation)
- [快速开始](#quick-start)
- [数据准备](#data-preparation)
- [评测维度](#evaluation-dimensions)
- [Trigger-Conditioned 打分](#trigger-conditioned-scoring)
- [分数 Schema](#score-schema)
- [使用参考](#usage-reference)
- [输出结构](#output-layout)
- [引用](#citation)
- [许可证](#license)

<a name="overview"></a>
## 概览

MBench 把"长视频记忆能力"这件事形式化为三条正交的能力线：

- **Entity Consistency** —— 主体的身份、外观、几何、纹理在离开画面再出现后是否保持一致？
- **Environment Consistency** —— 相机偏离视点再回到原处后，3D 空间结构、光照、渲染风格是否能恢复？
- **Causal Consistency** —— 画面外的物理过程能否合乎逻辑地继续？模型能否忠实响应外部指令（相机动作、分段 caption）？

每条能力线再细分 4 个子维度，共 **12 个评测维度**。每个维度都用一个专门的自动指标计算，归一化到 `[0, 1]` 范围。在合适的设定下，metric 还会被一个视觉语言模型 trigger 门控住，**避免那些根本没进入记忆挑战的样本（例如完全静态的视频）拉高模型排名**。详见下方 [Trigger-Conditioned 打分](#trigger-conditioned-scoring) 章节。

两种设定：

| 设定 | `dataset_id` | 条件形式 | 典型模型 |
|---|---|---|---|
| **MBench-A** | `mbencha` | 相机动作 + 单句caption描述 | hy_worldplay、matrix_game_3、yume、infinite_world、lingbot_world、matrix_game_2 |
| **MBench-T** | `mbencht` | 五段文本 prompt（`condition_id=text`） | cosmos、longcat、skyreels、longlive、helios、memflow、self_forcing、causal_forcing |

两种设定共用同一份 `samples/{subset}/{sample_id}/` 目录结构；运行时只调度对应前缀下的 metric。

<a name="updates"></a>
## 更新日志

- **2026-06** — 首次公开发布：12 个 metric 实现、MBench-A / MBench-T 数据适配器、canonical 分数 schema、内嵌 VLM trigger judge、契约校验 CLI。

<a name="installation"></a>
## 安装

**环境要求**：Python ≥ 3.10。

```bash
git clone https://github.com/study-overflow/MBench.git
cd MBench
pip install -e .
```

**其他依赖**：

| Metric 类别 | 所需额外依赖 |
|---|---|
| 空间 epipolar / reprojection、object 几何 | `numpy`、`opencv-python`，以及预先算好的相机位姿 artifact(可以使用[DepthAnything v3](https://github.com/bytedance-seed/depth-anything-3)得到) |
| 渲染风格 | `torch`、`torchvision`、`lpips`、`transformers`（DINOv2）、VGG-19 权重 |
| 渲染光照 | `opencv-python` |
| Entity human（identity / appearance） | `insightface`（Buffalo_L）、`torch`、`transformers`（DINOv2） |
| Entity object（texture / geometry） | `groundingdino`、`segment-anything-2`、`transformers`（DINOv2） |
| Causal 文本 / 动作交互 | `open_clip_torch` |
| Causal self-evolution（state / correctness） | OpenAI 兼容协议的 VLM endpoint |

典型完整安装：

```bash
pip install torch torchvision torchaudio   # 选择匹配你的 CUDA 版本
pip install opencv-python lpips transformers open_clip_torch insightface
pip install "segment-anything-2 @ git+https://github.com/facebookresearch/segment-anything-2.git"
pip install "groundingdino @ git+https://github.com/IDEA-Research/GroundingDINO.git"
```

验证安装：

```bash
mbench list-metrics    # 应输出 23 个已注册 metric
```

<a name="quick-start"></a>
## 快速开始

假设你的数据已按 `data/MBench-A-Setup/` 组织（见下一节）：

```bash
# 1. 校验该 metric 所需字段在若干样本上是否齐全
mbench validate data/MBench-A-Setup \
    --metrics mbencha.environment.spatial_epipolar \
    --models my_model --subsets environment --limit 4

# 2. 跑一次真实评测：单 metric、单 model
mbench eval data/MBench-A-Setup \
    --metrics mbencha.environment.spatial_epipolar \
    --models my_model --subsets environment \
    --output runs/my_first_run
```

带 VLM trigger 的 MBench-T 评测：

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

不传 `--vlm-judge` 时，相同命令也能跑通——以 **raw-consistency 模式**（不调任何 API），metric 仍然给出 `score`，只是不做 trigger 门控。

<a name="data-preparation"></a>
## 数据准备

你可以参考本节内容准备自己模型生成的数据对自己的模型进行评测：

```
data/MBench-{A|T}-Setup/
├── dataset.yaml
├── samples/{subset}/{sample_id}/
│   ├── sample.json                   # 元数据（object_anchor、caption_segments…）
│   └── reference.png / source_video.mp4         # 可选的真值素材
└── models/{model_id}/
    ├── samples.jsonl                 # 每条 (sample × condition) 一行
    ├── outputs/{subset}/{sample_id}/{condition_id}/video.mp4
    └── artifacts/{subset}/{sample_id}/{condition_id}/da3/results.npz
                                      # 仅 spatial / object-geometry / causal-action 类 metric 需要
```

四个 subset（`human`、`object`、`environment`、`causal`）与 metric 的子集一一对应。目前，`condition_id` 在 MBench-A 下是 `{action}_{length}`（如 `left_then_right_25s`、`forward_then_backward_10s`），在 MBench-T 下固定为字符串 `text`。你可以根据自己的要求补充新的条件（比如新的相机运动轨迹条件）。

最简 `dataset.yaml`：

```yaml
dataset_id: mbencha           # 或 mbencht
dataset_name: My MBench-A Setup
version: '1.0'
path_mode: relative_to_dataset_root

subsets:
  environment: {description: 环境/空间记忆。}
  human:       {description: 人物身份记忆。}
  object:      {description: 物体身份记忆。}
  causal:      {description: 因果/交互记忆。}

condition_id:
  pattern: "{action}_{length}"

paths:
  output_video: models/{model_id}/outputs/{subset}/{sample_id}/{condition_id}/video.mp4
  artifact_da3: models/{model_id}/artifacts/{subset}/{sample_id}/{condition_id}/da3/results.npz
```

每条生成视频对应的 `samples.jsonl` 行：

```json
{"item_id": "human:my_sample_001:left_then_right_25s", "dataset_id": "mbencha", "subset": "human", "sample_id": "my_sample_001", "condition_id": "left_then_right_25s", "model_id": "my_model", "media": {"videos": [{"path": "outputs/human/my_sample_001/left_then_right_25s/video.mp4", "role": "generated"}]}, "artifacts": {"da3": {"path": "artifacts/human/my_sample_001/left_then_right_25s/da3/results.npz"}}}
```

**`sample.json` 按 subset 必填字段**:

- `object`：T 侧需要 `metadata.object_card`，A 侧需要 `metadata.object_anchor`，描述目标物体。
- `causal`：T 侧需要 `metadata.caption_segments`，A 侧需要 `annotations.action` 或 `metadata.caption`。
- `environment` / `human`：仅 `media.videos` 必填；分段 caption 有助于 T 侧 trigger 判定。

DA3 相机位姿 artifact 在外部生成 —— 见 [DA3](https://github.com/lemmingforever/DA3)；MBench 只消费它产出的 `results.npz`。

<a name="evaluation-dimensions"></a>
## 评测维度

12 个维度及其注册名：

| 能力线 | 子维度 | MBench-A | MBench-T |
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

`mbench list-metrics` 可输出实时注册表与每个 metric 的简要描述。

<a name="trigger-conditioned-scoring"></a>
## Trigger-Conditioned 打分

模型可以通过生成静态、过度保守的内容来"作弊"——根本不进入记忆挑战，从而拿到虚高的一致性分。MBench 通过**把可靠性与覆盖率解耦**来防止这种作弊：

- **Memory Reliability** $S^{\text{rel}}$ —— 平均一致性，**仅在真正触发了记忆挑战的样本上**计算。
- **Trigger Coverage** $C^{\text{trig}}$ —— 触发了记忆挑战的样本占比。
- **M-Score** —— 二者的调和平均：$\text{M-Score} = 2 \cdot S^{\text{rel}} \cdot C^{\text{trig}} / (S^{\text{rel}} + C^{\text{trig}})$。

**Trigger 在哪里生效**：

| 设定 | Trigger 来源 | 默认行为 |
|---|---|---|
| **MBench-T** | VLM judge 对 8 帧均匀采样 + caption 分段做判定 | 默认关闭；通过 `--vlm-judge openai-compatible` 显式启用 |
| **MBench-A** | 不使用（动作条件 rollout 在物理上确定性地触发记忆挑战 —— 论文 §App） | 永远关闭，即使传 `--vlm-judge` 也不调 trigger |

通过命令行参数或环境变量配置 VLM：

```bash
export MBENCH_VLM_BASE_URL="..."     # OpenAI 兼容 /v1 endpoint
export MBENCH_VLM_API_KEY="..."
export MBENCH_VLM_MODEL="..."        # 任意带视觉能力的模型 id

# 或者每次运行单独传
mbench eval ... --vlm-judge openai-compatible \
    --vlm-base-url "$MBENCH_VLM_BASE_URL" \
    --vlm-api-key  "$MBENCH_VLM_API_KEY" \
    --vlm-model    "$MBENCH_VLM_MODEL"
```

消融场景下传 `--ignore-trigger`，可强制跳过 trigger 门控，即便 `--vlm-judge` 已传也照旁路。

<a name="score-schema"></a>
## 分数 Schema

不论是 A 还是 T、不论 metric 内部计算什么，每个 metric 在 unit、item、summary 三层都吐出**统一**的分数形态：

| Key | 取值范围 | 何时存在 | 含义 |
|---|---|---|---|
| `score` | [0, 1] | 始终（valid 时） | 归一化后的分数。 |
| `score_raw` | 各自原始单位 | 距离类 / VLM 1-5 分类 metric | 归一化前的原始信号；当 `raw == score` 时省略。 |
| `triggered` | {0.0, 1.0} | 仅 T 侧 metric | VLM trigger 是否触发（0 → 该样本不参与 reliability 平均）。 |
| `trigger_score` | [0, 1] | 仅 T 侧 metric | VLM trigger 置信度的归一化值。 |

实例（`mbench eval` 执行后）：

```json
// mbencha.environment.spatial_epipolar  units.jsonl 一行
{"unit_id": "...__f12_f48", "scores": {"score": 0.5523, "score_raw": 2.964}}

// mbencht.causal.progress_correctness  summary.json 中某一组
{"score": 0.74, "score_raw": 3.7, "triggered": 1.0, "trigger_score": 0.6}

// mbencht.* 中未触发的样本
{"triggered": 0.0, "trigger_score": 0.2}
```

<a name="usage-reference"></a>
## 使用参考

### 列出所有已注册 metric

```bash
mbench list-metrics
```

### 校验（严格模式）

```bash
mbench validate <source> --metrics <csv> [--models …] [--subsets …] [--limit N]
```

任何契约不满足都会非零退出。

### 评测（宽松模式 —— 跳过有问题的样本，把其他都跑完）

```bash
mbench eval <source> --metrics <csv> --output runs/<run_name>
    [--models <csv>] [--subsets <csv>] [--conditions <csv>]
    [--sample-ids <csv>] [--limit N]
    [--vlm-judge openai-compatible
       --vlm-base-url <URL> --vlm-api-key <KEY> --vlm-model <ID>
       --vlm-n-frames 8 --vlm-max-retries 3]
    [--workers N]                 # item 维度并行
    [--ignore-trigger]            # 旁路 VLM trigger 门控（仅消融用）
    [--no-progress-bar] [--quiet] [--no-log-file]
```

### 制备 artifact（不会在 eval 中自动调用）

```bash
mbench prepare <source> --producer <name> --output artifacts/
```

### Python API

```python
from mbench.core.pipeline import run_eval
from mbench.core.registry import metric_registry, adapter_registry, aggregator_registry
import mbench.cli; mbench.cli._bootstrap()    # 注册所有内置插件

adapter = adapter_registry.get("mbench-dir")
items = adapter.load("data/MBench-A-Setup", subsets=["environment"], limit=4)
metric = metric_registry.get("mbencha.environment.spatial_epipolar")
aggregator = aggregator_registry.get("mean")

summary = run_eval(items, [metric], aggregator, "runs/python_api_demo")
```

<a name="output-layout"></a>
## 输出结构

```
runs/<run_name>/
├── items.jsonl              # 每条评测样本一行 (item_id, model_id)
├── eval.log                 # 逐样本进度日志（时间戳、分数、错误）
└── metrics/<metric_name>/
    ├── units.jsonl          # 每个 evaluation unit 一条 UnitResult（帧对、track、segment…）
    ├── items.jsonl          # 每条样本一条 ItemResult（由 unit 聚合）
    └── summary.json         # SummaryResult，按 model_id 分组
```

`UnitResult.metadata['raw_scores']` 保存了 metric 原始的内部字段（在 canonical 投影中被 drop 掉的那些），方便事后调试或离线重新计算其他备选 score key。

<a name="citation"></a>
## 引用

如果 MBench 对你的研究有帮助，欢迎引用：

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
## 许可证

代码采用 [MIT 许可证](LICENSE) 发布。
