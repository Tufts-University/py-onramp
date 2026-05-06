# The Command Line

Almost everything in this tutorial happens in a terminal, a text-based window
where you type commands and the computer executes them.
If you are used to clicking around in a graphical interface, the terminal can
feel strange at first, but it quickly becomes the fastest way to get things
done in scientific computing.
This page gets you up and running with the handful of commands you actually
need.

## Opening your terminal

**macOS** - Open the built-in *Terminal* app.
The easiest way is Spotlight: press `⌘ Space`, type `terminal`, and press
`Enter`.

**Linux** - Most desktop environments have a keyboard shortcut such as
`Ctrl+Alt+T`, or you can find a *Terminal* entry in your applications menu.

**Windows** - Open *PowerShell* or *Git Bash*.
For PowerShell, press `Win`, type `powershell`, and press `Enter`.
Git Bash is installed alongside Git (see below) and provides a Unix-style
shell that behaves identically to macOS and Linux terminals.

## Navigating with `pwd`, `ls`, and `cd`

The terminal always has a *working directory*, the folder it is currently
"inside".
Three commands let you find your bearings and move around:

- `pwd` - **p**rint **w**orking **d**irectory: shows the full path of where you are.
- `ls` - **l**i**s**t the files and folders in the current directory.
- `cd` - **c**hange **d**irectory.

Here is a typical session.
Start by checking where you are and what is there:

```console
$ pwd
/home/user
$ ls
Desktop  Documents  Downloads  Pictures
```

Move into a folder:

```console
$ cd Documents
$ ls
research  notes.txt
```

Go one level back up with `cd ..`:

```console
$ cd ..
$ ls
Desktop  Documents  Downloads  Pictures
```

You can also jump straight to a path deeper in the tree:

```console
$ cd Documents/research
$ ls
data  analysis.py
```

Those three commands are genuinely most of what you need to navigate the file
system from the command line.

## Getting this repository with `git`

`git` is a version-control tool that also lets you download a full copy of any
public repository with one command.
If you do not have `git` installed yet, get it from
[https://git-scm.com/install](https://git-scm.com/install). If on Windows, this
also installs Git Bash on Windows.
Once installed, to get this tutorial's code and materials, run:

```console
$ git clone https://github.com/Tufts-University/py-onramp
```

This creates a new folder called `py-onramp` in your current directory.
Step into it:

```console
$ cd py-onramp
$ ls
code  docs  README.md
```

You now have everything you need locally.
Head to [Getting Started](getting-started.md) to set up your Python environment
and run some code.
