# MBench Demo

End-to-end smoke test that covers **every metric × every subset × both sides**:
8 real samples (4 MBench-A action-conditioned + 4 MBench-T text-conditioned),
spanning all four subsets (`causal` / `environment` / `human` / `object`),
and 23 registered metrics (12 on the A side + 11 on the T side).

`run_demo.sh` evaluates each metric independently and prints a categorized
summary so you can see at a glance which metrics produced real scores and
which were skipped, with the cause for every skip.

## What ships in this directory

```
demo/
├── README.md                  # this file
├── run_demo.sh                # one-shot 23-metric runner
└── dataset/
    ├── a/                     # MBench-A subset
    │   ├── dataset.yaml
    │   ├── samples/{causal,environment,human,object}/<sample_id>/
    │   │   ├── sample.json                   ← tracked
    │   │   ├── source_video.mp4              ← gitignored (heavy)
    │   │   ├── first_frame.png               ← gitignored
    │   │   └── reference.png  (object only)  ← gitignored
    │   └── models/matrix_game_2/
    │       ├── samples.jsonl                 ← tracked (4 items)
    │       ├── outputs/<subset>/<sid>/<cond>/video.mp4   ← gitignored
    │       └── artifacts/<subset>/<sid>/<cond>/da3/results.npz  ← gitignored
    └── t/                     # MBench-T subset (longcat, 4 items, same shape)
```

The directory tree, the `dataset.yaml`, all `sample.json` files, and the per-model
`samples.jsonl` are tracked. Videos, ground-truth media, and DA3 artifacts are
**gitignored** because they total >1 GB; on a fresh `git clone` you get the
metadata only and need to repopulate the heavy files (see *Heavy files* below).

## Sample coverage

|     | A — `matrix_game_2`                                                      | T — `longcat`                                       |
|-----|--------------------------------------------------------------------------|-----------------------------------------------------|
| causal      | `a00110_01560` / `left_then_right_25s` (DA3 ✓)                   | `a00786_00064_dimming` / `text` (DA3 ✗)             |
| environment | `sample_051_a5b4a90f` / `left_then_right_25s` (DA3 ✓)            | `mem_dl3dv_bd3da6c5…107ac3a959` / `text` (DA3 ✓)     |
| human       | `mem_openhumanvid_9da45b…1a6023` / `right_then_left_25s` (DA3 ✗) | `mem_openhumanvid_784aff…b91b733` / `text` (DA3 ✗)   |
| object      | `sample_115_b7816bd8` / `left_then_right_25s` (DA3 ✓)            | `vobj_sample_068_3f64c617` / `text` (DA3 ✓)          |

5/8 samples ship with DA3 artifacts. The 3 without DA3 will skip
spatial/geometry metrics with `missing DA3 artifact` (intentional — covers the
"no DA3 yet" code path).

## Running

From the `submission/` root:

```bash
bash demo/run_demo.sh
```

Each metric prints `pass` / `skip <reason>` / `fail`, then a summary block.
Per-metric outputs land under `demo/output/<side>/<metric>/`, and the raw
stdout for each metric lands under `demo/output/logs/<side>__<metric>.log`.

## Expected output (zero-dependency, no GPU, no VLM key)

The script is designed to never crash. On a vanilla CPU machine with only
`mbench`'s minimal dependencies installed, you should see exactly:

```
Passed (≥1 valid item with real score): 5
  ✓ a/mbencha.environment.rendering_lighting (4/4)
  ✓ a/mbencha.causal.camera_interaction      (3/3)
  ✓ t/mbencht.environment.spatial_epipolar   (2/2)
  ✓ t/mbencht.environment.spatial_reprojection (2/2)
  ✓ t/mbencht.environment.rendering_lighting (4/4)

Skipped (env/dep/data missing): 18
  ... (categorized list with the missing dep / artifact / GPU / VLM)

Failed (script/import error): 0
```

These 5 metrics are the ones whose entire dependency surface is
`numpy + opencv + scipy + the bundled DA3 artifacts` — they verify that the
plugin registry, dataset adapter, contract layer, eval pipeline, canonical
score projection, mean aggregator, and result writer all work end-to-end on
real MBench data.

## Skip taxonomy: how to unlock the other 18 metrics

| Skip reason                                  | Affected metrics | How to unlock |
|----------------------------------------------|---|---|
| `missing Python dep (insightface)`            | T-human × 2, A-human × 2 (silent) | `pip install insightface onnxruntime` and download the `buffalo_l` ArcFace pack |
| `missing Python dep (lpips)`                  | T-rendering_style                  | `pip install lpips` (downloads VGG weights on first call) |
| `no GPU / CUDA driver`                        | T-prompt_interaction (OpenCLIP)    | Run on a CUDA host, or wait for the CPU code path |
| `VLM judge not configured (--vlm-judge)`      | state_progress × 2, progress_correctness × 2 | `export MBENCH_VLM_BASE_URL=… MBENCH_VLM_API_KEY=… MBENCH_VLM_MODEL=…` and add `--vlm-judge openai-compatible` to the eval command |
| `missing DA3 artifact`                        | object_geometry × 2                | Download the DA3 pack (see below) and copy `results.npz` into the matching `artifacts/<subset>/<sid>/<cond>/da3/` directory |
| `SIFT/DA3: no valid frame-pair scores`        | A spatial_epipolar / spatial_reprojection / rendering_style on `matrix_game_2` | A 25-second `left_then_right_25s` rollout has very little camera-direction overlap; this is a real downstream "metric saw no usable evidence" outcome (not a bug). Try a `*_360`/`*_720` condition for these dimensions instead. |
| `no valid items` (catch-all)                  | A-entity × 3, A/T-object_texture, A-prompt_interaction | Usually a downstream signal-not-found case (no face/object track recovered, or CLIP segment count mismatch). See the per-metric log under `demo/output/logs/`. |

## Heavy files (videos / DA3) — cloud-drive download

To keep the public repo small (<1 MB), every `video.mp4`, `source_video.mp4`,
`first_frame.png`, `reference.png`, and `da3/results.npz` is gitignored.
Download the demo bundle from the cloud drive and unpack it on top of
`demo/dataset/` so each path lines up with the existing tree:

```
https://drive.google.com/drive/folders/1Fpa21xpbfC8MN-GO86mMvJegGda8UDMK?usp=sharing
```

After unpacking you should have, e.g.:

```
demo/dataset/a/models/matrix_game_2/outputs/causal/a00110_01560/left_then_right_25s/video.mp4
demo/dataset/a/models/matrix_game_2/artifacts/causal/a00110_01560/left_then_right_25s/da3/results.npz
demo/dataset/a/samples/causal/a00110_01560/source_video.mp4
...
```

Then `bash demo/run_demo.sh` should reproduce the 5/18/0 split above (more
metrics will pass once you also install the optional deps in the table).

## Output layout (per metric)

```
demo/output/
├── logs/<side>__<metric>.log        # raw mbench stdout/stderr
└── <side>/<metric>/
    ├── eval.log                     # mbench's own progress log
    ├── items.jsonl                  # per-item results (one row per sample)
    └── metrics/<metric>/
        ├── units.jsonl              # per-evaluation-unit raw scores
        ├── items.jsonl              # canonical item-level scores
        └── summary.json             # aggregated per-model
```

`summary.json` is what gets read by leaderboard tooling; the canonical
schema is `{"score": float in [0,1], "score_raw"?: float}` on the A side and
adds `{"triggered", "trigger_score"}` on the T side. See the project root
README for the full canonical-schema spec.
