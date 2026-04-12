# Parametrizing a research script

!!! note

    TODO: Redo this with a repo that they can clone and work in..
    This seciton generally needs work.

When you are experimenting, you change things constantly.

Maybe you want to try a wider network, a different learning rate, a
larger batch size, or a new random seed. At first, the simplest thing is
often to open the file and edit a few numbers by hand. That is a
perfectly reasonable place to start.

But eventually you want more than that:

- you want to rerun an old experiment exactly
- you want to compare several runs without repeatedly editing the file
- you want your commands to record what changed

This section uses a fully dummy script, `code/dummy_train.py`, as the
baseline example.

Nothing here is about machine learning. The point is to show how
different parametrization styles all feed the same
`ExperimentConfig` dataclass and eventually call the same
`train(config)` function.

The runnable examples below use packages from this repository's
`examples` dependency group, so the commands use `uv run --group
examples ...`.

```python title="dummy_train.py"
--8<-- "code/dummy_train.py"
```

There is no single best way to parametrize a script. More machinery can
make experiments more reproducible, but it also makes the codebase more
abstract and harder for a newcomer to follow. A good rule is to start
with the smallest tool that fits the current problem.

## Stage 0: Do nothing, edit the file directly

!!! note 
    
    "It is not enough to do nothing. One must also be doing nothing." - Zhuang Zhou 

The original script is useful because it is obvious.

If you are still understanding the script and only changing one or two
values occasionally, editing the file may be the right choice. There is
no parser to debug, no config format to learn, and no hidden indirection.

The downside is that your experiment settings now live in your edit
history rather than in a command or a config file. That becomes awkward
once you want to compare multiple runs.

## Stage 1: command line arguments with `argparse`

The first step up is usually to expose the most important
hyper-parameters as command line flags. Python's standard library
already includes [`argparse`](https://docs.python.org/3/library/argparse.html),
so this adds very little machinery.

In this repository, the `argparse` version lives in
`code/dummy_train/argparse_main.py`. The shared dataclass and
placeholder `train(config)` function live in `code/dummy_train/common.py`,
so that each parametrization style can focus on how parameters are
collected.

The dataclass itself looks like this:

```python
@dataclass
class ExperimentConfig:
    """A small bundle of hyperparameters passed through the example flows."""
    width: int = 64
    depth: int = 2
    lr: float = 3e-3
    n_epochs: int = 200
    batch_size: int = 256
    seed: int = 0
    output_dir: str = "outputs/default"
```

```python title="argparse_main.py"
--8<-- "code/dummy_train/argparse_main.py"
```

The flow is:

- parse a few flags
- build an `ExperimentConfig`
- pass that dataclass into `train(config)`

If you run it, the script just prints the resolved config and a message
that the training function is intentionally a placeholder.

Run it like this:

```console
$ uv run --group examples python code/dummy_train/argparse_main.py --n-epochs 50 --lr 1e-3 --width 128
```

This approach is a good fit when:

- there are only a handful of parameters worth changing often
- you want `--help` output immediately
- each run is mostly a one-off command

The tradeoff is that long commands can get noisy. Once you find yourself
copying and pasting a large command and changing only one value each
time, a config file often becomes easier to manage.

## Stage 2: a TOML config

The next step is to put your default experiment settings in a config
file instead of hard-coding them in the script.

Here, the defaults live in `code/dummy_train/config.toml`, and the
loader is in `code/dummy_train/toml_main.py`. The script reads the TOML
file and constructs the same `ExperimentConfig` dataclass used in the
`argparse` example.

```toml title="config.toml"
--8<-- "code/dummy_train/config.toml"
```

```python title="toml_main.py"
--8<-- "code/dummy_train/toml_main.py"
```

Typical usage:

```console
$ uv run --group examples python code/dummy_train/toml_main.py
$ uv run --group examples python code/dummy_train/toml_main.py --config code/dummy_train/config.toml
```

This buys you two things at once:

- the defaults are written down in a stable, readable file
- the script stays very simple

The mental model is still simple: load a file and pass the resulting
dataclass into `train(config)`.

!!! note

    Something that you might was is the ability to pass a configuration
    file, along with CLI overrides. [I've implemented this in a gist,
    that you can drop
    in](https://gist.github.com/abhijit-c/b18acfce4b86c2e5e11983463550fd6f).

## Stage 3: Hydra

[Hydra](https://hydra.cc/) is the industry standard tooling for this type of
thing.
It's particularly useful when you have many related experiments and you want
configuration to become a first-class part of the workflow.

For this example, the Hydra setup is intentionally small:

- `code/dummy_train/hydra_main.py` is the entry point
- `code/dummy_train/conf/config.yaml` is the root config
- `code/dummy_train/conf/experiment/default.yaml` holds the experiment values

```python title="hydra_main.py"
--8<-- "code/dummy_train/hydra_main.py"
```

```yaml title="conf/config.yaml"
--8<-- "code/dummy_train/conf/config.yaml"
```

```yaml title="conf/experiment/default.yaml"
--8<-- "code/dummy_train/conf/experiment/default.yaml"
```

Hydra still ends in the same place as the other stages: a populated
`ExperimentConfig` dataclass passed into the placeholder training
function.

Typical usage:

```console
$ uv run --group examples python code/dummy_train/hydra_main.py
$ uv run --group examples python code/dummy_train/hydra_main.py experiment.lr=1e-3 experiment.width=128
```

Hydra starts to pay off when:

- experiment configurations are numerous enough that you want them in a directory tree
- different groups of settings naturally belong together
- command-line overrides should work on nested config values

But Hydra is not automatically better than TOML or `argparse`.
It adds concepts: config groups, composition, and a framework-specific
override syntax. That extra power is useful only if the project is large
enough to benefit from it. For a single local script, Hydra may be more
structure than you need.

## Looking ahead: Optuna

The three approaches above are all about **manual** parametrization:
they make it easier for a human to choose hyper-parameters and feed them
into the same script in a controlled way.

Hyperparameter tuning, through something like [Optuna](https://optuna.org/),
addresses a different problem. Instead of choosing each learning rate or width
by hand, Optuna can search over candidate values automatically and track which
settings worked best.

## Comparing the approaches

| Approach | Best for | Main benefit | Main cost |
| --- | --- | --- | --- |
| Edit the file | Very small, local experiments | Almost no abstraction | Hard to reproduce and compare runs |
| `argparse` | A few parameters changed often | Simple path from flags to a dataclass | Long commands become repetitive |
| TOML | Stable defaults written down in a file | File-based defaults with a simple loader | Another config format to maintain |
| Hydra | Larger experiment trees | Structured configs and powerful overrides | More concepts and more complexity |

The important point is not to "graduate" to the most sophisticated tool
as quickly as possible. The right level of parametrization is the one
that reduces friction for your current experiments without making the
code harder to understand than it needs to be.
