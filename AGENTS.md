# GraphCliff + ACA workspace boundary

- This is the independent repository `D:/GraphCliff-ACA`, branch `codex/graphcliff-aca`. It is not a linked worktree of GraphCliff-Pair.
- Scope: test supervised triplet learning inspired by ACANet on the original single-molecule GraphCliff predictor. First establish an official ACA transfer baseline before proposing a new triplet variant.
- Keep `D:/GraphCliff-Pair`, `D:/GraphCliff-Pair-FPPool-LongPoly`, and `D:/GraphCliff-main` read-only. Do not edit their code, plans, Git state, experiments, or running jobs.
- Do not import FPPool, cross-attention replacements, reference residual transfer, RelSup, or reliability modules into the initial ACA comparison.
- Preserve upstream LICENSE and attribution. Upstream snapshot is `d18857db0303346c0be715a56febf71da1768592`; provenance and decisions are in `notes.md`.
- `README.md` is the upstream document; `README_ACA.md` is this project's entry point. Published upstream scores are not results from this experiment.
- Persist this direction's plan in `task_plan.md` and findings in `notes.md`.
- Before implementation, inspect the current files and uncommitted changes. Keep a runnable plain-loss baseline and make the ACA path optional.
- Mine triplets using training molecules and labels within the same task only. Select hyperparameters and checkpoints using development validation; keep official test labels out of training, mining, and tuning.
- Separate regression-loss changes from triplet-loss changes. Record effective triplet counts, loss terms, label scale, source hashes, configurations, and seed/split identities.
- Define a bounded pilot before training. Creating this repository does not execute or validate ACA, and does not restart any previous experiment queue.
- All writable outputs belong in this repository. No writable links to other projects. Never push to the authors' upstream repository. The user authorized public publication to origin https://github.com/yxxiaocainiao/GraphCliff-ACA; retain this task's scope and exclude local checkpoints/runtime artifacts.
