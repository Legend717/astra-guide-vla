# Installation (Linux)

The runtime uses POSIX file locks and process groups. Windows source inspection is
supported; use Linux/WSL for the agent. GPU rendering and VLA inference need a
compatible NVIDIA/CUDA installation. No model packages are required for offline
agent unit tests.

Already have compatible environments? Follow [reusing existing environments](reproduction.md#use-an-existing-environment)
and run the prerequisite checks before installing anything.

## Agent environment

```bash
python3.11 -m venv .venv-agent
.venv-agent/bin/pip install -r requirements-agent.txt
.venv-agent/bin/pip install -e packages/openpi-client
cp .env.example .env
```

Set `BASE_URL` and `API_KEY` in `.env` or the environment. The server must support
the Responses API, the requested model, encrypted reasoning and original-detail
image inputs. JEV mode also requires JEV credentials.

Use the exact names `BASE_URL` and `API_KEY` shown in `.env.example`; the client
reads these names, rather than `OPENAI_BASE_URL` or `OPENAI_API_KEY`. Environment
variables override values in the file selected by `--env-file`. `BASE_URL` is the
API root (for example, `https://api.openai.com/v1`); the SDK appends `/responses`.
A compatible gateway must accept Responses-style `instructions`, `input`, function
`tools`, `parallel_tool_calls=false`, `store=false`, `reasoning.effort`, and
`include=["reasoning.encrypted_content"]`. Images are PNG/JPEG data URLs in
`input_image` blocks with `detail="original"`. The agent replays response items and
returns tool results as `function_call_output` with their matching `call_id`.
Chat Completions-only endpoints are not supported. Select the model and reasoning
level with `--model` and `--reasoning` on `run_robomme_agent.py`; the evaluation
driver uses the report's model and reasoning settings.

The VLA uses a separate WebSocket service started by `serve_robomme_policy.py`,
selected with `--policy-host` and `--policy-port`.

## Offline tests

To run the agent and client tests, install the test dependencies in the agent
environment and run from the repository root:

```bash
.venv-agent/bin/pip install 'pytest>=8' 'imageio>=2.36.1'
.venv-agent/bin/python -m pytest scripts packages/openpi-client -q
```

## Simulator (separate environment)

```bash
git submodule update --init third_party/robomme_benchmark
cd third_party/robomme_benchmark
uv sync
uv pip install -e .
cd ../..
export ROBOMME_SIM_PYTHON="$PWD/third_party/robomme_benchmark/.venv/bin/python"
"$ROBOMME_SIM_PYTHON" -m pip install -e packages/openpi-client
```

If the uv environment has no pip, use `uv pip install --python "$ROBOMME_SIM_PYTHON" -e packages/openpi-client`.
Follow the pinned benchmark README for simulator dependencies/assets. The supplied
launcher preserves the installed renderer by default. Optional CPU rendering
requires a locally installed Mesa lavapipe Vulkan ICD; set `ROBOMME_CPU_RENDER=1`
and `VK_ICD_FILENAMES` to that ICD.
The legacy `--cpu-render-launcher` argument can select a custom Bash launcher.

## Policy (separate environment)

Use Python 3.11–3.12 and install uv. From the repository root:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Get `symbolic-grounded-subgoal` from the official
[MME-VLA collection](https://huggingface.co/Yinpei/mme_vla_suite).
Keep `history_config.txt` beside the `79999` directory; do not substitute the
perceptual model or pi05 base. The checkpoint must include `params/` and
`assets/robomme/norm_stats.json`. Unpack any archives supplied by the model host.
Download model weights and datasets separately.

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_robomme_policy.py \
  --checkpoint /absolute/path/symbolic-grounded-subgoal/79999 \
  --mode grounded --seed 42 --host 127.0.0.1 --port 8000
```

The server isolates random state per connection. Keep the checkpoint and model
configuration fixed for all tasks. Existing checkpoint normalization assets are
preferred over recomputing statistics. Training priors can be rebuilt with
`scripts/build_scheduling_priors.py --help`; `--data-root` is explicit and required.
