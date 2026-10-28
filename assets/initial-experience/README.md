# Initial experience

Every run starts with shared global guidelines in `general.json`, alongside
task-specific duration priors calculated from the official RoboMME training
demonstrations. The global guidelines are part of the initial instruction context
for both evaluation presets. The task files contain training duration advice.

Recommended budgets use the lower-quartile duration, rounded down to ten controls
and capped at 200. Durations below 30 controls request close review instead.

Copy the files to a writable run directory before use. See
[reproduction](../../docs/reproduction.md) for evaluation commands and instructions
to rebuild the task duration advice from training data.
