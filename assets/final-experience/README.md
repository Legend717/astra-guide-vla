# Final experience

This snapshot contains effective entries from the experience store at the end
of the 800-episode step-budget evaluation. It includes duration advice, general
execution guidance, and task-specific findings. Superseded entries and JEV-dependent
advice, including summaries based on JEV evidence, are excluded. All entries are
in English.

Use `--experience-preset final` with `scripts/evaluate_release.py`. Updates are
disabled by default so each run evaluates the same snapshot. To enable the
first-five/all-outcomes, remaining/failure-only update schedule, add
`--experience-updates protocol`.

This snapshot includes experience collected during the reported test evaluation.
Report results separately from runs starting with the initial preset. The snapshot
has been filtered and translated.

The driver copies the snapshot into the run directory; keep these bundled files
unchanged. See [reproduction](../../docs/reproduction.md) for commands.
