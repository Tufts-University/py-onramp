# Improving the usability of Python programs

## Logging

It can be useful to print out either a message or the value of some
variable while your code is running. This is quite common and is usually
accomplished with a simple call to the `print` function.

```python
x = 1.234
print("The value of x is {0:0.4f}.".format(x))
```

```text
The value of x is 1.2340.
```

Doing this is a good idea to keep track of milestones in your code. That
way, both when you are developing your code and when other users are
running the code, they can be notified of an event, progress, or value.

Printing a message is also useful for notifying the user when something
is not going as expected. These are all different *levels* of
messaging.

*Logging* is simply engaging in this behavior of printing out messages,
with the added feature that you include metadata such as a timestamp or
message category, as well as a filter where only messages with a high
enough level of criticality are actually allowed to be printed.

### Logging basics

The general idea is that there are multiple levels of messages that can
be printed. Typically these include:

1. DEBUG, for diagnostic purposes
2. INFO, for basic information
3. WARNING, for non-normal behavior
4. ERROR, for errors where the operation cannot continue
5. CRITICAL, for errors where the program cannot continue

During the initialization portion of your code, you would configure a
*logger* object with a format, where to print messages such as console,
file, or both, and what level to use by default. Usually, you would set
the default log level to `INFO` and the debugging messages used for
diagnostics would not actually be printed. Then, allow the user to
override this with a [command line argument](#command-line-arguments),
for example `--debug`.

### Example setup

Python has a [logging](https://docs.python.org/3/library/logging.html)
module as part of the standard library. It is very comprehensive and
allows the user to heavily customize many parts of the behavior. It is
pretty straightforward to implement your own logging functionality, but
unless you are doing something special, why not use the standard
library?

```python
import logging

log = logging.getLogger("ProjectName")

file_handler = logging.FileHandler("path/for/output.log")
console_handler = logging.StreamHandler()

formatter = logging.Formatter("%(levelname)s %(asctime)s %(name)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

log.addHandler(file_handler)
log.addHandler(console_handler)
log.setLevel(logging.INFO)
```

Then, somewhere in the code:

```python
log.debug("report on some variable")
log.info("notification of milestone")
log.warn("non-standard behavior")
log.error("unrecoverable issue")
log.critical("panic!")
```

```text
INFO 2018-07-24 09:41:56,683 ProjectName - notification of milestone
WARNING 2018-07-24 09:41:56,835 ProjectName - non-standard behavior
ERROR 2018-07-24 09:41:57,103 ProjectName - unrecoverable issue
CRITICAL 2018-07-24 09:41:57,103 ProjectName - panic!
```

Notice that the debug message was not printed. This is because we set
the log level to `INFO`. Only messages with a level equal to or higher
than the assigned level will make it past the filter.

### Logging with color

Another common feature of logging is to add color as an indicator of the
message type. Obviously, this only applies to messages that are printed
to the console. If you have ever started up a *Jupyter* notebook server,
you might have noticed the logging messages it prints in a similar
format, where the metadata is in a bold color. The color codes are
generally as follows:

- DEBUG, blue
- INFO, green
- WARNING, orange or yellow
- ERROR, red
- CRITICAL, purple

## Command line arguments

In addition to packaging your code in a way that other users or projects
can import for use in their code, it often makes sense to also make
elements of the code executable from the command line as standalone
scripts. Python has everything you need to do this built right in.

As with logging, there are several Python packages available that handle
command line argument parsing for you, including a robust
implementation provided right in the standard library:
[argparse](https://docs.python.org/3/library/argparse.html).

The *argparse* module, as well as the others, rely on a universally
accepted convention for how command line arguments should be structured.
Nearly all of the standard utilities on Unix and Linux systems use this
same syntax. This convention covers both the command line argument
syntax and the structure of *usage* statements that your script prints
out, for example when supplying the `--help` option. The *argparse*
module actually takes care of all of this for you.

### Unix convention

There is a fair bit of complexity to the convention surrounding usage
statements, but the argument syntax is fairly simple.

*Positional arguments* are those that do not have names. These are
usually file paths in the context of analysis scripts. *Optional
arguments* are those that have defaults and may or may not accept a
value.

Optional arguments can be specified with short form or long form names,
usually both. The short form names are a single letter preceded by a
single dash, for example `-a`. Short form options that do not take an
argument can be stacked, for example `-abc`. Long form arguments are
whole words preceded by two dashes, for example `--debug`. Long form
arguments that are multiple words are usually joined with dashes, for
example `--output-directory`.

There is more, but these are the basics.

### Simple example

The best, most robust, and cross-platform way of providing a standalone
script with your package is to let your `setup.py` file handle it. Doing
the following will create the proper executable on both Windows and Unix
systems and put it in a place that is readily callable, that is, on the
user's `PATH`.

```python title="setup.py"
# use "entry_points" to point to function and setuptools
# will create executables on your behalf.
setup(
# ...
    # syntax: "{name}={package}.{module}:{function}"
    # "{name}" will be on your PATH in the same "/bin/"
    # alongside python/pip executables.
    entry_points={"console_scripts": [
        "do_science=my_package.do_science:main",
    ]},
# ...
)
```

This says that you have a file, `my_package/do_science.py`, with a
function called `main` that when called does the thing you want the
script to do. The function will not be given any arguments, but you can
get what you need from `sys.argv`. This has the effect of creating an
executable you can invoke with the name `do_science` that behaves
equivalently to the following:

```python
import sys
from my_package.do_science import main

sys.exit(main())
```

With this in mind, your function can and should return integer values
which will be used as the exit status of the command. This is another
Unix convention. Returning zero is for success, while returning a
non-zero status indicates some specific error has occurred.

The following shows a basic usage of `argparse` and how to define your
`main` function.

```python title="do_science.py"
import argparse

parser = argparse.ArgumentParser(
    prog="do_science",
    description="do cool science thing",
)

# positional argument
parser.add_argument("input_file", help="path to input data file")

# optional argument
parser.add_argument("-d", "--debug", action="store_true", help="enable debugging messages")

def main() -> int:
    """Main entry point for `do_science`.

    Returns:
        exit_status: int
            0 if success, non-zero otherwise.
    """

    # parse_args() automatically grabs sys.argv if you do not provide them.
    opts = parser.parse_args()
    # opts is a namespace
    # opts.input_file is a string with the value from the command line
    # opts.debug is True or False, defaulting to False with "store_true"
    return 0
```

After the package is installed, `pip install my_package ...`, you will
be able to call the script:

```text
> do_science
usage: do_science [-h] [-d] input_file
```

```text
> do_science --help
usage: do_science [-h] [-d] input_file

do cool science thing

positional arguments:
  input_file   path to input data file

optional arguments:
  -h, --help   show this help message and exit
  -d, --debug  enable debugging messages
```
