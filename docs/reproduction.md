# Reproduction

This guide describes how to evaluate the harness using the setup in the
technical report. Reported results and evaluation settings are in [results.md](results.md).

## Choose the starting experience

| Preset | Contents | Default updates |
| --- | --- | --- |
| `initial` | Shared global guidelines and training-demonstration duration priors | Episodes 0–4: after either outcome; episodes 5–49: after failure |
| `final` | Effective entries from the completed 800-episode step-budget experience store, excluding JEV-dependent advice | Disabled |

The initial preset starts with training-derived duration priors. The final preset
includes experience collected during the reported test evaluation. Report results
for each preset separately. See [final experience](../assets/final-experience/README.md).

Both presets include the shared initial global guidelines. The final preset also
contains guidance added during evaluation.
Both presets are copied into a new run directory. Use `--experience-updates frozen`
to disable updates or `--experience-updates protocol` to use the episode-dependent
update schedule with either preset.

## Use an existing environment

If compatible environments are already installed, use their Python interpreters
directly. From this repository's root, set paths for your machine:

```bash
export AGENT_PYTHON=/absolute/path/to/agent/environment/bin/python
export POLICY_PYTHON=/absolute/path/to/policy/environment/bin/python
export ROBOMME_SIM_PYTHON=/absolute/path/to/simulator/environment/bin/python
export ROBOMME_ENV_FILE=/absolute/path/to/credentials.env
export ROBOMME_CHECKPOINT=/absolute/path/to/symbolic-grounded-subgoal/79999
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:$PWD/third_party/robomme_benchmark/src"
```

Initialize the pinned simulator submodule using the installation guide. These
source paths ensure a shared environment imports this checkout's code. For CPU
rendering, pass your existing launcher with `--cpu-render-launcher /path/to/launcher.sh`.

Before launching services, run the checks with your existing agent interpreter:

```bash
"$AGENT_PYTHON" scripts/check_reproduction.py \
  --sim-python "$ROBOMME_SIM_PYTHON" --policy-python "$POLICY_PYTHON" \
  --checkpoint "$ROBOMME_CHECKPOINT" --env-file "$ROBOMME_ENV_FILE" \
  --experience-preset initial
# Repeat with --experience-preset final for the final experience snapshot.
```

This checks imports, checkpoint files, episode metadata, experience structure,
credential configuration, and GPU visibility. Check API access and end-to-end
execution with a live episode.

Start the policy server when ready to run an experiment, choosing an available GPU:

```bash
CUDA_VISIBLE_DEVICES=0 "$POLICY_PYTHON" scripts/serve_robomme_policy.py \
  --checkpoint "$ROBOMME_CHECKPOINT" --mode grounded --seed 42 \
  --host 127.0.0.1 --port 8000
```

## Prepare the initial duration advice

Ready-to-use global guidance and task duration priors are included in `assets/initial-experience`.
Use that directory directly with `--initial-experience`. To rebuild them,
obtain the official preprocessed RoboMME training data and follow
[installation.md](installation.md) or reuse existing environments as described above.
Build the initial experience from training demonstrations, without importing past
evaluation experience:

```bash
"$POLICY_PYTHON" scripts/build_scheduling_priors.py \
  --data-root /absolute/path/to/preprocessed_data \
  --output-dir runs/initial-priors
# Add the shared global instruction context to the rebuilt task priors:
cp assets/initial-experience/general.json runs/initial-priors/experience/general.json
```

The builder writes duration statistics and an `experience` directory containing
`general.json` and `tasks/<Task>.json`. Keep that initial directory unchanged;
the evaluation driver copies it to a new writable run directory. The initial task
files contain training-duration priors only.

## Single episode

Start the official GroundSG policy server as described in installation.md. Then:

```bash
mkdir -p runs
cp -r assets/initial-experience runs/demo-experience
"$AGENT_PYTHON" scripts/run_robomme_agent.py \
  --task PickXtimes --episode 0 --dataset test \
  --run-dir runs/demo-pickxtimes-0 \
  --sim-python "$ROBOMME_SIM_PYTHON" --env-file "$ROBOMME_ENV_FILE" \
  --policy-backend real --policy-host 127.0.0.1 --policy-port 8000 \
  --model gpt-6-astra --reasoning high --supervision-mode step-budget \
  --experience runs/demo-experience/general.json \
  --task-experience runs/demo-experience/tasks/PickXtimes.json \
  --experience-update --experience-update-policy always \
  --prefix 16 --max-scheduled-steps 200 \
  --max-calls 80 --max-output-tokens 8192 --max-context-tokens 300000
```

## Full evaluation

The driver runs all 16 tasks, episodes 0–49 per task. It shares the global advice
store and keeps task stores separate. With the initial preset, updates use `always`
for episodes 0–4 and `failure-only` for episodes 5–49, matching the paper. Each episode reads an
immutable snapshot; accepted updates become available to subsequent episodes.
The driver runs tasks serially in the table order. Because global advice is shared,
task order can affect later episodes; the original run order is not recorded here.

```bash
# Preview: prints commands without files, API calls or simulator execution.
"$AGENT_PYTHON" scripts/evaluate_release.py \
  --output runs/reproduction-01 \
  --experience-preset initial \
  --sim-python "$ROBOMME_SIM_PYTHON" --env-file "$ROBOMME_ENV_FILE"

# Add --execute to run when the model and simulator services are ready.
"$AGENT_PYTHON" scripts/evaluate_release.py \
  --output runs/reproduction-01 \
  --experience-preset initial \
  --sim-python "$ROBOMME_SIM_PYTHON" --env-file "$ROBOMME_ENV_FILE" --execute

# Once all planned episodes have completed:
"$AGENT_PYTHON" scripts/summarize_step_budget.py \
  --experiment-dir runs/reproduction-01
```

To preview evaluation with the final experience snapshot:

```bash
"$AGENT_PYTHON" scripts/evaluate_release.py \
  --output runs/reproduction-final-01 --experience-preset final \
  --sim-python "$ROBOMME_SIM_PYTHON" --env-file "$ROBOMME_ENV_FILE"
```

Add `--execute` only when ready to run. The final preset defaults to frozen
experience; use `--experience-updates protocol` to continue updating it.

`--tasks PickXtimes` selects one complete task. Add `--episodes 0 1 2 3 4`
to run only those test episodes for each selected task. `--max-steps` configures the
execution ceiling, with a default of 1,300 controls. The driver uses a new output
directory and stops if an agent process exits unsuccessfully.
Retain outputs when resuming with the underlying agent's `--attach` or
`--resume-review` options.

## Other modes

`astra-only`, `rules-only` and `jev` are retained for comparison and recovery.
JEV uses an external adviser through `--jev-config`; example profiles are under
`scripts/robomme_agent/profiles`. Default step-budget mode has no JEV dependency.
