# Testing your code

!!! note

    This section is based heavily on Ned Batchelder's excellent article
    and PyCon 2014 talk [Getting Started
    Testing](https://nedbatchelder.com/text/test0.html).
    
    *Tests are the dental floss of development: everyone knows they
    should do it more, but they do not, and they feel guilty about it.*
    
    Ned Batchelder
    
    *Code without tests should be approached with a 10-foot pole.*
    
    me

How can you write modular, extensible, and reusable code?

After making changes to a program, how do you ensure that it will still
give the same answers as before?

How can we make finding and fixing bugs an easy, fun, and rewarding
experience?

These seemingly unrelated questions all have the same answer, and it is
**automated testing**.

## Testing by example: differentiating $f(x; t) = A(t)^{-1} x$

Suppose we have a vector-valued function
\[
f(x; t) = A(t)^{-1} x,
\]
where \(x \in \mathbb{R}^2\) is fixed and \(t\) is a scalar parameter.
If \(A(t)\) is differentiable and invertible, then differentiating the
identity \(A(t) A(t)^{-1} = I\) gives
\[
A'(t) A(t)^{-1} + A(t) \frac{d}{dt} A(t)^{-1} = 0,
\]
so
\[
\frac{d}{dt} A(t)^{-1} = -A(t)^{-1} A'(t) A(t)^{-1}.
\]
Therefore
\[
\frac{d}{dt} f(x; t)
= \frac{d}{dt}\!\left(A(t)^{-1} x\right)
= -A(t)^{-1} A'(t) A(t)^{-1} x.
\]

For a concrete example, let
\[
A(t) =
\begin{pmatrix}
1+t & 0 \\
0 & 1-t
\end{pmatrix}.
\]
Then
\[
f(x; t)
=
\begin{pmatrix}
\dfrac{x_1}{1+t} \\
\dfrac{x_2}{1-t}
\end{pmatrix},
\qquad
\frac{d}{dt} f(x; t)
=
\begin{pmatrix}
-\dfrac{x_1}{(1+t)^2} \\
\dfrac{x_2}{(1-t)^2}
\end{pmatrix}.
\]

Here is an original implementation based on a hand derivation. It has a
sign error in the derivative:

```python title="matrix_derivative.py"
--8<-- "code/matrix_derivative_v1.py"
```

- Which tests would you write first?
- Which values of \(x\) and \(t\) make the expected answer easy to
  compute by hand?
- If the derivative is wrong, can the tests tell you which sign is
  wrong?

### Testing interactively

The quickest way to test a mathematical claim is often to try a few
simple values in an interactive Python session:

```python
>>> import numpy as np
>>> from matrix_derivative_v1 import f, df_dt
>>> x = np.array([2.0, 3.0])
>>> f(x, 0.0)
array([2., 3.])
>>> df_dt(x, 0.0)
array([ 2., -3.])
```

The value of \(f(x; 0)\) is correct, since \(A(0) = I\). But the
derivative is already suspicious: from the formula above we expect
\((-2, 3)\), not \((2, -3)\).

Interactive testing is useful, but it has the same limitations here as
everywhere else. You have to remember what you tried, you have to rerun
everything manually after each change, and you still have to inspect the
results yourself.

### Writing a test script

A much better approach is to put the checks into a script:

```python title="test_matrix_derivative.py"
--8<-- "code/test_matrix_derivative_v1.py"
```

Now we can run the same checks whenever we want:

```console
$ python test_matrix_derivative.py
f(x; t) = [1.66666667 3.75      ]
df_dt(x, t) = [ 1.38888889 -4.6875    ]
```

This is more reproducible, but we still have to inspect the output and
decide for ourselves whether the numbers are correct.

### Testing with assertions

For mathematics-heavy code, assertions are especially useful because we
usually know what a small reference calculation should be.

The `assert` statement checks whether a condition is true. If it is
false, Python raises an `AssertionError`:

We can now replace printed output with a statement of the expected
answer:

```python title="test_matrix_derivative.py"
--8<-- "code/test_matrix_derivative_v2.py"
```

If we run this script,

```console
$ python test_matrix_derivative.py

Traceback (most recent call last):
  File "test_matrix_derivative.py", line 8, in <module>
    assert np.allclose(df_dt(x, t), np.array([-2.0, 3.0]))
AssertionError
```

we immediately learn that the claimed derivative is inconsistent with a
simple hand calculation at \(t = 0\).

### A brief aside: finite differences

When testing a derivative, it is often useful to have a second way to
compute it. One common option is a finite-difference approximation:
\[
\frac{d}{dt} f(x; t)
\approx
\frac{f(x; t+h) - f(x; t-h)}{2h}.
\]
This is not an exact formula, but for a small value of \(h\) it provides
an independent numerical check.

Here is a script that compares the analytic derivative with both a
hand-computed formula and a centered finite difference:

```python title="test_matrix_derivative.py"
--8<-- "code/test_matrix_derivative_v3.py"
```

```console
$ python test_matrix_derivative.py

Traceback (most recent call last):
  File "test_matrix_derivative.py", line 13, in <module>
    assert np.allclose(df_dt(x, t), np.array([-2.0 / 1.2**2, 3.0 / 0.8**2]))
AssertionError
```

Again, the script stops at the first failure. That is enough to tell us
there is a problem, but not enough to summarize all the checks we care
about.

### Using a test runner

A test runner takes a collection of tests, executes them all, and then
reports which passed and which failed. A very popular test runner for
Python is
[pytest](https://docs.pytest.org/en/latest/).

To use `pytest`, we rewrite each check as a separate test function:

```python title="test_matrix_derivative.py"
--8<-- "code/test_matrix_derivative_v4.py"
```

If you are already using SciPy, it also provides tools for derivative
checks. In particular,
[`scipy.optimize.check_grad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.check_grad.html)
can be useful when your problem can be phrased in terms of gradients of
scalar-valued functions. Here we will keep the test self-contained and
write the expected formula directly.

To run the tests, use the project environment:

```console
$ uv run pytest code/test_matrix_derivative_v4.py

collected 3 items

code/test_matrix_derivative_v4.py .FF                                 [100%]

=================================== FAILURES ===================================
______________________ test_df_dt_matches_formula_at_zero ______________________

    def test_df_dt_matches_formula_at_zero():
        x = np.array([2.0, 3.0])
>       assert np.allclose(df_dt(x, 0.0), np.array([-2.0, 3.0]))
E       assert False
E        +  where False = <function allclose at ...>(array([ 2., -3.]), array([-2.,  3.]))

code/test_matrix_derivative_v4.py:17: AssertionError

_____________________ test_df_dt_matches_finite_difference _____________________

    def test_df_dt_matches_finite_difference():
        x = np.array([2.0, 3.0])
        t = 0.2
>       assert np.allclose(df_dt(x, t), finite_difference_df_dt(x, t), atol=1.0e-8)
E       assert False
E        +  where False = <function allclose at ...>(array([ 1.38888889, -4.6875    ]), array([-1.38888889,  4.6875    ]), atol=1e-08)

code/test_matrix_derivative_v4.py:23: AssertionError
========================= 2 failed, 1 passed in 0.07s ==========================
```

The report is already informative. One test passed, so our function
evaluation is fine. Two derivative tests failed, and both failures point
to the same issue: the expected vector is the negative of what our code
returned.

### Useful tests

Now that we know how to run tests, which tests are actually useful for a
derivative like this?

Arbitrary choices of \(x\) and \(t\) are less helpful than cases that
make the structure of the mathematics visible. For this example, useful
tests include:

- \(t = 0\), where \(A(0) = I\) and the formula simplifies
- A value such as \(t = 0.2\), where the denominators are still simple
  but not equal to \(1\)
- A finite-difference comparison, which checks the derivative without
  reusing the same algebraic derivation
- Cases where one component of \(x\) is zero, to isolate each term of
  the derivative separately

These are the kinds of tests included in the `pytest` file above.

### Fixing the code

The tests suggest that the code used
\[
+A(t)^{-1} A'(t) A(t)^{-1} x
\]
instead of
\[
-A(t)^{-1} A'(t) A(t)^{-1} x.
\]
In other words, the hand derivation dropped the overall minus sign.

Here is the corrected implementation:

```python title="matrix_derivative.py"
--8<-- "code/matrix_derivative_v2.py"
```

If we update the tests to import the corrected implementation, we obtain

```python title="test_matrix_derivative.py"
--8<-- "code/test_matrix_derivative_v5.py"
```

and the test runner now reports

```console
$ uv run pytest code/test_matrix_derivative_v5.py

collected 3 items

code/test_matrix_derivative_v5.py ...                                 [100%]

============================== 3 passed in 0.10s ===============================
```

This is exactly why tests are useful: they give us a way to check that
the bug is fixed and that we did not break the rest of the calculation
while fixing it.

## Types of testing

Software testing is a vast topic and there are [many levels and
types](https://en.wikipedia.org/wiki/Software_testing) of software
testing.

For scientific and research software, the focus of testing efforts is
primarily:

1. **Unit tests**: Unit tests aim to test small, independent sections of
   code, a function or parts of a function, so that when a test fails,
   the failure can easily be associated with that section of code. This
   is the kind of testing that we have been doing so far.
2. **Regression tests**: Regression tests aim to check whether changes
   to the program result in it producing different results from before.
   Regression tests can test larger sections of code than unit tests. As
   an example, if you are writing a machine learning application, you
   may want to run your model on small data in an automated way each
   time your software undergoes changes, and make sure that the same or
   a better result is produced.

## Test-driven development

[Test-driven development
(TDD)](https://en.wikipedia.org/wiki/Test-driven_development) is the
practice of writing tests for a function or method *before* actually
writing any code for that function or method. The TDD process is to:

1. Write a test for a function or method
2. Write just enough code that the function or method passes that test
3. Ensure that all tests written so far pass
4. Repeat the above steps until you are satisfied with the code

Proponents of TDD suggest that this results in better code. Whether or
not TDD sounds appealing to you, writing tests should be *part* of your
development process and never an afterthought. In the process of writing
tests, you often come up with new corner cases for your code and realize
better ways to organize it. The result is usually code that is more
modular, more reusable, and of course more testable than if you did not
do any testing.

## Growing a useful test suite

More tests are always better than fewer, and your code should have as
many tests as you are willing to write. That being said, some tests are
more useful than others. Designing a useful suite of tests is a
challenge in itself, and it helps to keep the following in mind when
growing tests:

1. **Tests should run quickly**: testing is meant to be done as often as
   possible. Your entire test suite should complete in no more than a
   few seconds, otherwise you will not run your tests often enough for
   them to be useful. Always test your functions or algorithms on very
   small and simple data, even if in practice they will be dealing with
   more complex and large datasets.
2. **Tests should be focused**: each test should exercise a small part
   of your code. When a test fails, it should be easy for you to figure
   out which part of your program you need to focus debugging efforts
   on. This can be difficult if your code is not modular, that is, if
   different parts of your code depend heavily on each other. This is
   one of the reasons TDD is said to produce more modular code.
3. **Tests should cover all possible code paths**: if your function has
   multiple code paths, for example an if-else statement, write tests
   that execute both the if part and the else part. Otherwise, you might
   have bugs in your code and still have all tests pass.
4. **Test data should include difficult and edge cases**: it is easy to
   write code that only handles cases with well-defined inputs and
   outputs. In practice however, your code may have to deal with input
   data for which it is not clear what the behavior should be. For
   example, what should `flip_string("")` return? Make sure you write
   tests for such cases, so that you force your code to handle them.
