# Extending DADApy

Contributions to the library are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

There are various ways to contribute:

- If you need support, want to report a bug or ask for features, you can check the [Issues page](https://github.com/sissa-data-science/DADApy/issues) and raise an issue, if applicable.

- If you would like to contribute a bug fix of feature then [submit a Pull request](https://github.com/sissa-data-science/DADApy/pulls).

For other kinds of feedback, you can contact one of the
[authors](https://github.com/sissa_data_science/dadapy/main/AUTHORS.md).


## A few simple rules

- Before working on a feature, reach out to one of the core developers or discuss the feature in an issue. The library caters a diverse audience and new features require upfront coordination.

- Always include unit tests when you contribute new features, as they help to a) prove that your code works correctly, and b) guard against future breaking changes to lower the maintenance cost.

- Whenever possible, keep API compatibility in mind when you change code in the `dadapy` library. Reviewers of your pull request will comment on any API compatibility issues.

- Before committing and opening a PR, run all tests locally. This saves CI hours and ensures you only commit clean code.

## Step-by-step guide to Pull Requests (PRs)

Imagine you want to contribute a feature `X` to `DADApy`.
In this case, you would ideally follow these steps:

1. Download (or fork) the repository
2. Open a new branch named `feature_X`
3. Add the feature to this branch locally, with a corresponding test
4. Check that all tests pass locally (more on this in the next section)
5. Push your changes to `origin/feature_X`
6. Open a pull request from `origin/feature_X` to `origin/main`
7. Check again that all tests of the PR pass 
8. Wait for another developer to review your code, and modify the code if needed
9. After the PR is approved, the code can be merged!

## A guide to run tests locally

### Linting tests

To quickly check code linting you can run the following three commands
from the main `DADApy` folder.

```
make black
make isort
make flake8
```

These will automatically format your code using [`black`](https://black.readthedocs.io/en/stable/), will 
automatically sort your import statements using [`isort`](https://pycqa.github.io/isort/index.html), and
will suggest code style improvements using [`flake8`](https://flake8.pycqa.org/en/latest/).

The above commands depend on the specific package versions installed on your local machine.
To be sure that the tests will also pass on the GitHub machine, 
you can use ['tox'](https://github.com/tox-dev/tox/tree/master) with the following commands.

```
tox -e black
tox -e isort
tox -e flake8
```

### Code tests

To quickly check code correctness you can run either

```
make test
```

or

```
make coverage 
```

where the second command will also provide a coverage report.

Once again, to be sure that the tests will also pass on GitHub you can use tox via

```
tox -e py3.x
```

where `3.x` should be set to your local Python version. 
