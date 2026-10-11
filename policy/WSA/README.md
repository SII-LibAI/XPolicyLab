# WSA

**Contributor:** WSA Team | **Paper:** [WSA: A 3D-Centric World-Spatial-Action Model for Generalizable Robot Control](https://openreview.net/pdf?id=bfphFLldXq) | **Original code:** https://github.com/zaleni/WSA | **Checkpoint:** [zaleni/wsa-b-clean2rand-delta-causal](https://huggingface.co/zaleni/wsa-b-clean2rand-delta-causal)

`WSA` integrates the WSA-Base RoboTwin 2.0 clean-to-random checkpoint with XPolicyLab. It supports the dual-arm `aloha_agilex` embodiment, joint-space delta actions, the three standard RGB cameras, action chunks, and WSA-Base image history. Integration files live at this directory level; the vendored upstream source is in `WSA/`. The adapter also recognizes WSA-Large checkpoint layouts, but the reproducible checkpoint and benchmark evidence in this submission are for WSA-Base.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official benchmark protocol: [RoboTwin 2.0 Leaderboard](https://robotwin-platform.github.io/leaderboard).

## Installation

Create the upstream environment first, then install the adapter. WSA's tested stack uses Python 3.10, CUDA 12.8, and PyTorch 2.7.1.

```bash
conda create -y -n wsa python=3.10
conda activate wsa
conda install -c conda-forge ffmpeg=7.1.1 svt-av1 -y
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

cd XPolicyLab/policy/WSA
bash install.sh
```

`install.sh` installs the vendored WSA and XPolicyLab in editable mode, installs WSA's runtime/test extras and the websocket dependencies, pins `transformers==4.57.1`, and applies WSA's Qwen3-VL cached-inference patch. RoboTwin runs also need the simulator environment on the environment-client side; it is not installed into the policy environment.

The vendored tree does not embed WSA's external Git submodules. Released-checkpoint evaluation disables the DA3 teacher and does not need its source. For Base/Large training with the upstream nonzero 3D loss, clone [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) separately and export `DA3_CODE_ROOT=/path/to/Depth-Anything-3` before `train.sh`.

## Data Processing

WSA trains on LeRobot v3.0. `process_data.sh` calls XPolicyLab's official `scripts/transform_lerobot_v30_format.py`; its keys exactly match the official mapping (`observation.state`, `action`, and `observation.images.cam_high/cam_left_wrist/cam_right_wrist`). No custom image decoder is used. Because the official converter labels its potentially mixed-robot output as `unified_robot`, the wrapper retags this adapter's single supported `aloha_agilex` export as WSA's equivalent `aloha` metadata schema before computing normalization statistics; tensor and image fields are unchanged.

Place raw XPolicyLab trajectories under the parent workspace's `data/<bench>/<task>/<env_cfg>/data/` tree, then run:

```bash
cd XPolicyLab/policy/WSA
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [max_episode]

# Example
WSA_ACTION_MODE=delta \
  bash process_data.sh RoboTwin cotrain aloha_agilex joint 200
```

The default output is `data/lerobot/<bench>-<ckpt>-<env>-<action>/`; stats are written to `data/norm_stats/aloha/<delta|abs>/stats.json`. `WSA_SOURCE_DATA_TYPE` defaults to `RoboDojo`, the shared XPolicyLab v1.0 trajectory reader used for both benchmark directory names. To use the published RoboTwin LeRobot-v3.0 export instead, set `WSA_ROBOTWIN_ROOT` during training and skip conversion.

## Model Assets

The adapter accepts a local exported checkpoint, a standard XPolicyLab run directory, or a Hugging Face repo id. The checkpoint used for the reported C2C/C2R results is `zaleni/wsa-b-clean2rand-delta-causal`. Download it explicitly for reproducible or air-gapped evaluation:

```bash
cd XPolicyLab/policy/WSA
bash download_checkpoints.sh base
```

WSA-Base also requires the Qwen3-VL model/processor and Cosmos tokenizer assets. Download them locally before evaluation and point all three environment variables at the local directories; do not rely on an implicit network download during a benchmark run:

```bash
hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir /absolute/path/to/Qwen3-VL-2B-Instruct
hf download nvidia/Cosmos-Tokenizer-CI8x8 \
  --local-dir /absolute/path/to/Cosmos-Tokenizer-CI8x8

export QWEN3_VL_PRETRAINED_PATH=/absolute/path/to/Qwen3-VL-2B-Instruct
export QWEN3_VL_PROCESSOR_PATH=/absolute/path/to/Qwen3-VL-2B-Instruct
export COSMOS_TOKENIZER_PATH_OR_NAME=/absolute/path/to/Cosmos-Tokenizer-CI8x8
```

WSA-Large can load the Wan text encoder directly (the default for evaluation) or use the model/config asset paths documented by upstream WSA.

## Training

The wrapper preserves the XPolicyLab checkpoint convention and launches upstream WSA fine-tuning. `action_type` remains `joint`; `WSA_ACTION_MODE` selects whether the policy predicts `delta` or `abs` joint actions.

```bash
cd XPolicyLab/policy/WSA
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# WSA-Base example on GPUs 0-7
WSA_VARIANT=base WSA_ACTION_MODE=delta \
  bash train.sh RoboTwin cotrain aloha_agilex joint 0 0,1,2,3,4,5,6,7

# WSA-Large example
WSA_VARIANT=large WSA_ACTION_MODE=delta LOAD_TEXT_ENCODER=true \
  bash train.sh RoboTwin cotrain-large aloha_agilex joint 0 0,1,2,3,4,5,6,7
```

Checkpoints land under `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`. Useful overrides are `WSA_ROBOTWIN_ROOT`, `WSA_STATS_ROOT`, `WSA_POLICY_INIT_PATH`, `WSA_PROC_PER_NODE`, and the upstream launch variables documented in `WSA/README.md`. The released training recipe used 8 NVIDIA H200 GPUs.

## Evaluation

```bash
cd XPolicyLab/policy/WSA
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Offline protocol/action-shape check; no checkpoint or simulator required
EVAL_ENV_TYPE=debug WSA_ALLOW_DUMMY_POLICY=true \
  bash eval.sh RoboTwin stack_bowls dummy aloha_agilex joint 0 0 0 wsa base

# WSA-Base clean-to-random checkpoint (downloaded automatically if not local)
bash eval.sh RoboTwin adjust_bottle zaleni/wsa-b-clean2rand-delta-causal \
  aloha_agilex joint 0 0 0 wsa robotwin
```

Run the debug check again with `DEBUG_OBS_ENCODED=1`; decoding remains server-side and `model.py` receives RGB arrays. Leave `EVAL_ENV_TYPE` unset (or set it to `sim`) for RoboTwin. The adapter reads per-task `infer_horizon` and `binarize_gripper` from `WSA/configs/robotwin_eval_config.yaml`; explicit environment overrides take precedence.

For official simulator evaluation, select the scene setting explicitly through RoboTwin's scheduler. `env_cfg_type=aloha_agilex` selects the action profile; it does not select clean versus randomized scenes. The latter is controlled by `--task-config`:

```bash
cd RoboTwin
export WSA_CHECKPOINT_PATH=/absolute/path/to/wsa-b-clean2rand-delta-causal
export WSA_ACTION_MODE=delta
export EVAL_ENV_TYPE=sim
export QWEN3_VL_PRETRAINED_PATH=/absolute/path/to/Qwen3-VL-2B-Instruct
export QWEN3_VL_PROCESSOR_PATH=/absolute/path/to/Qwen3-VL-2B-Instruct
export COSMOS_TOKENIZER_PATH_OR_NAME=/absolute/path/to/Cosmos-Tokenizer-CI8x8

# Clean-to-clean (C2C)
bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/all_tasks.yml \
  --policy-name WSA \
  --ckpt-name wsa-b-clean2rand-delta-causal \
  --env-cfg-type aloha_agilex \
  --policy-conda-env wsa \
  --eval-env-conda-env robotwin \
  --action-type joint \
  --task-config demo_clean \
  --test-num 100

# Clean-to-random (C2R): use the same command with
# --task-config demo_randomized
```

To use one environment for both policy and RoboTwin, pass the same Conda environment name to `--policy-conda-env` and `--eval-env-conda-env`.

## RoboTwin 2.0 Results

The released WSA-Base checkpoint was trained on clean demonstrations and evaluated over the 50-task RoboTwin 2.0 clean-to-random protocol. C2C uses `demo_clean`, C2R uses `demo_randomized`, and the reported average is `(C2C + C2R) / 2`.

| Checkpoint | C2C | C2R | Average |
| --- | ---: | ---: | ---: |
| `zaleni/wsa-b-clean2rand-delta-causal` | 83.8 | 77.8 | 80.8 |

## Configuration

Important `deploy.yml` settings are `action_mode`, `infer_horizon`, `binarize_gripper`, `dtype`, `checkpoint_path`, `wsa_root`, and `wsa_large_stats_path`. The server launcher maps these environment variables to overrides: `WSA_CHECKPOINT_PATH`, `WSA_ACTION_MODE`, `WSA_DTYPE`, `WSA_INFER_HORIZON`, `WSA_BINARIZE_GRIPPER`, `WSA_LARGE_STATS_PATH`, the Qwen/Cosmos/DA3 variables above, and WSA-Large's `WAN_MODEL_ID`, `WAN_TOKENIZER_MODEL_ID`, `ACTION_DIT_PRETRAINED_PATH`, `FUTURE_3D_PRETRAINED_PATH`, and `WSA_LARGE_LOAD_TEXT_ENCODER`.

## Notes

- Supported combination: `bench_name=RoboTwin`, `env_cfg_type=aloha_agilex`, `action_type=joint`; released checkpoints use `action_mode=delta`.
- The adapter infers Base versus Large from `config.json`; no evaluation-side variant flag is required.
- `eval_batch` is disabled. `get_action_batch` remains contract-complete and evaluates requested environments sequentially because one WSA forward is already GPU-intensive.
- A standalone XPolicyLab checkout has no outer `env_cfg/`; `model.py` first uses `get_robot_action_dim_info()` and only falls back to `utils/robot/_robot_info.json` for the debug-only standalone case.
- `allow_dummy_policy` returns hold-position chunks and is only for protocol checks, never for benchmark results.
