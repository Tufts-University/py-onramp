# py-onramp

An onramp to the modern data / scientific python workflow.

## Docs Modernization TODO

### Must Have

- [x] `docs/index.md`
  - [x] Replace the Anaconda-first setup guidance with a `uv`-first workflow.
  - [x] Add a short "getting started" path for local development: install Python, create/sync the environment, and run examples with `uv run`.
- [x] `docs/packaging.md`
  - [x] Rewrite the installable-project section around `pyproject.toml` instead of `setup.py`.
  - [x] Prefer modern packaging guidance based on `uv` and PEP 517/518/621 rather than `setuptools` plus manual `pip install -e . --user`.
  - [x] Add examples for editable installs, dependency groups, and lockfile-driven reproducibility.
- [ ] `docs/usability.md`
  - [ ] Replace `setup.py` `entry_points` examples with `[project.scripts]` in `pyproject.toml`.
  - [ ] Update install/run examples so they reflect `uv run`, project-local environments, and modern CLI/TOML workflows.
- [ ] `docs/performance.md`
  - [ ] Replace or supplement the `Numba` section with a more current accelerator/JIT section built around JAX.
  - [ ] Refresh the parallelization discussion to mention current options such as JAX, Dask, multiprocessing/joblib, and when each is appropriate.
  - [ ] Add guidance on CPU vs GPU execution and the tradeoff between vectorization, JIT compilation, and distributed execution.
- [ ] New docs section: numerical reliability and scientific testing
  - [ ] Cover floating-point behavior, tolerances, regression tests for computed results, and the difference between exact and approximate equality.
  - [ ] Add examples of testing mathematical invariants, edge cases, and algorithmic correctness, not just code paths.
  - [ ] Consider a short introduction to property-based testing for mathematical code.
- [ ] New docs section: reproducible research workflows
  - [ ] Cover pinned dependencies, lockfiles, runtime metadata, seeds, and JAX deterministic PRNG patterns.
  - [ ] Show how to organize inputs, outputs, and experiment artifacts so results can be rerun months later.
  - [ ] Include guidance on config files and recording command invocations or parameters used for a run.
- [ ] New docs section: notebooks versus scripts and packages
  - [ ] Explain when notebooks are useful, when code should move into modules, and how to avoid notebooks becoming the only source of truth.
  - [ ] Include basic reproducibility guidance for notebook-heavy workflows.
- [ ] New docs section: git and collaboration for research code
  - [ ] Cover branching, small commits, reviewable changes, `.gitignore`, and what data or generated artifacts should stay out of version control.
  - [ ] Frame this as basic research hygiene rather than only "software engineering for teams".
- [ ] New docs section: running on a SLURM cluster
  - [ ] Add a dedicated page covering `sbatch`, `srun`, resource requests, log files, modules, and activating project environments on the cluster.
  - [ ] Include a minimal batch script template and an example workflow for launching Python jobs reproducibly.
  - [ ] Cover common HPC topics that beginners hit immediately: job arrays, scratch space, file staging, checkpointing, and debugging failed jobs.
- [ ] New docs section: code quality and automation
  - [ ] Add a short section on `ruff`, formatting, optional type checking, and `pre-commit`.
  - [ ] Explain the minimum automation worth adopting even for a small research codebase.
  - [ ] Add CI guidance so tests and docs build automatically on each push. Maybe.

### Nice to Have

- [ ] New docs section: plotting and communication
  - [ ] Cover publication-quality figures, labeling, styles, vector exports, and reproducible figure generation with Matplotlib.
  - [ ] TikZ? Liz?
- [ ] New docs section: core scientific Python tools for math research
  - [ ] Add guidance on when to use NumPy, SciPy, sparse matrices, optimization/integration/linear algebra routines, and where JAX fits relative to NumPy/SciPy.
- [ ] New docs section: research project layout
  - [ ] Show a recommended directory structure for `src`, tests, docs, scripts, data, results, and generated artifacts.
- [ ] New docs section: remote and cluster-native development
  - [ ] Cover remote editing, data transfer, checkpoint/restart patterns, and environment portability across laptop and cluster.
