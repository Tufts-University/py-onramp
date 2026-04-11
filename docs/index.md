# An onramp for scientific computing and data analysis in Modern Python

This tutorial covers topics that are essential for research in scientific
computing and data analysis using Python, but are typically *not* covered in an
introductory course. 
These are the things you *need* to know if you are writing software that
meets any of the following criteria:

- You expect to be working on it for more than a couple of weeks.
- You expect that it will be composed of more than a hundred or so lines
  of code.
- You expect to return to it, later in time.
- You want it to produce results that can be trusted, for example if you
  are publishing a research paper based on those results.
- You expect that it will be used by one or more other people.
- You are contributing to another project, for example an open-source
  software package.

## What you will learn

1. How to get started with a modern scientific Python workflow using
   `uv`: install Python, create and sync a project environment, and run
   code reproducibly on your laptop.
2. How to organize your code as an installable project using
   `pyproject.toml`, rather than a loose collection of scripts.
3. How to test scientific code, including numerical correctness,
   floating-point tolerances, regression tests, and other checks that
   help you trust your results.
4. How to document your code and build usable command-line workflows so
   that you and others can run, understand, and extend your work.
5. How to make your work reproducible with pinned dependencies,
   lockfiles, recorded parameters, and a clear separation between
   notebooks, scripts, packages, inputs, and outputs.
6. How to use basic collaboration and quality practices for research
   code, including Git, formatting, linting, and lightweight
   automation.
7. How to improve performance, choose appropriate parallel or JIT tools,
   and move from laptop workflows to shared HPC and SLURM-based systems
   when needed.

## What you need to know

This tutorial assumes you know the very basics of programming with Python.
If you can write a loop and a function in Python, and if you know how to run a
`.py` script, you should be able to follow this tutorial easily.

Moreover, it is written for the student and researcher in STEM.
The examples, code design, and tools are primarily designed with that field in
mind.
While many of the ideas are transferable, we cannot guarantee that.

## Contents

- [Getting Started](getting-started.md)
- [Packaging](packaging.md)
- [Testing](testing.md)
