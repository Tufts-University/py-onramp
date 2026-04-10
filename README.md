# py-onramp

An onramp to the modern data / scientific python workflow.

## Acknowledgements
This is forked from the excellent [Python 102](https://python-102.readthedocs.io/) writeup, just modernized for tooling and with HPC sections.

## Docs Modernization TODO

- `docs/index.md`
  - Replace the Anaconda-first setup guidance with a `uv`-first workflow.
  - Add a short "getting started" path for local development: install Python, create/sync the environment, and run examples with `uv run`.
  - Clarify what is expected on a laptop versus on a shared HPC system. The HPC stuff should come later.
- `docs/packaging.md`
  - Rewrite the installable-project section around `pyproject.toml` instead of `setup.py`.
  - Prefer modern packaging guidance based on `uv` and PEP 517/518/621 rather than `setuptools` plus manual `pip install -e . --user`.
  - Add examples for editable installs, dependency groups, and lockfile-driven reproducibility.
- `docs/usability.md`
  - Replace `setup.py` `entry_points` examples with `[project.scripts]` in `pyproject.toml`.
  - Update install/run examples so they reflect `uv run`, project-local environments, and modern CLI / TOML worksflows.
- `docs/performance.md`
  - Replace or supplement the `Numba` section with a more current accelerator/JIT section built around JAX.
  - Refresh the parallelization discussion to mention current options such as JAX, Dask, multiprocessing/joblib, and when each is appropriate.
  - Add guidance on CPU vs GPU execution and the tradeoff between vectorization, JIT compilation, and distributed execution.
- New docs section: running on a SLURM cluster
  - Add a dedicated page covering `sbatch`, `srun`, resource requests, log files, modules, and activating project environments on the cluster.
  - Include a minimal batch script template and an example workflow for launching Python jobs reproducibly.
  - Cover common HPC topics that beginners hit immediately: job arrays, scratch space, file staging, and debugging failed jobs.
- Modernization ideas
  - Add a short section on code quality tooling such as `ruff`, formatting, and optional type checking.
  - Add reproducibility guidance for research projects: pinned dependencies, lockfiles, and recording runtime metadata. JAX deterministic PRNG.
  - Consider a brief notebooks section covering when to use notebooks versus scripts/packages.
