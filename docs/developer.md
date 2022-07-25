# Developer information

## Doctests
JAX uses pytest in doctest mode to test the code examples within the documentation.
You can run this using
```
pytest docs
```
Additionally, JAX runs pytest in `doctest-modules` mode to ensure code examples in
function docstrings will run correctly. You can run this locally using, for example:
```
pytest --doctest-modules jax/_src/numpy/lax_numpy.py
```
Keep in mind that there are several files that are marked to be skipped when the
doctest command is run on the full package; you can see the details in
[`ci-build.yaml`](https://github.com/google/jax/blob/main/.github/workflows/ci-build.yaml)

# Type checking

We use `mypy` to check the type hints. To check types locally the same way
as the CI checks them:

```
pip install mypy
mypy --config=mypy.ini --show-error-codes jax
```

Alternatively, you can use the [pre-commit](https://pre-commit.com/) framework to run this
on all staged files in your git repository, automatically using the same mypy version as
in the GitHub CI:

```
pre-commit run mypy
```

# Linting

JAX uses the [flake8](https://flake8.pycqa.org/) linter to ensure code quality. You can check
your local changes by running:

```
pip install flake8
flake8 jax
```

Alternatively, you can use the [pre-commit](https://pre-commit.com/) framework to run this
on all staged files in your git repository, automatically using the same flake8 version as
the GitHub tests:

```
pre-commit run flake8
```

# Update documentation

To rebuild the documentation, install several packages:
```
pip install -r docs/requirements.txt
```
And then run:
```
sphinx-build -b html docs docs/build/html -j auto
```
This can take a long time because it executes many of the notebooks in the documentation source;
if you'd prefer to build the docs without executing the notebooks, you can run:
```
sphinx-build -b html -D nb_execution_mode=off docs docs/build/html -j auto
```
You can then see the generated documentation in `docs/build/html/index.html`.

The `-j auto` option controls the parallelism of the build. You can use a number
in place of `auto` to control how many CPU cores to use.

(update-notebooks)=

## Update notebooks

We use [jupytext](https://jupytext.readthedocs.io/) to maintain two synced copies of the notebooks
in `docs/notebooks`: one in `ipynb` format, and one in `md` format. The advantage of the former
is that it can be opened and executed directly in Colab; the advantage of the latter is that
it makes it much easier to track diffs within version control.

### Editing ipynb

For making large changes that substantially modify code and outputs, it is easiest to
edit the notebooks in Jupyter or in Colab. To edit notebooks in the Colab interface,
open <http://colab.research.google.com> and `Upload` from your local repo.
Update it as needed, `Run all cells` then `Download ipynb`.
You may want to test that it executes properly, using `sphinx-build` as explained above.

### Editing md

For making smaller changes to the text content of the notebooks, it is easiest to edit the
`.md` versions using a text editor.

### Syncing notebooks

After editing either the ipynb or md versions of the notebooks, you can sync the two versions
using [jupytext](https://jupytext.readthedocs.io/) by running `jupytext --sync` on the updated
notebooks; for example:

```
pip install jupytext==1.13.8
jupytext --sync docs/notebooks/quickstart.ipynb
```

The jupytext version should match that specified in
[.pre-commit-config.yaml](https://github.com/google/jax/blob/main/.pre-commit-config.yaml).

To check that the markdown and ipynb files are properly synced, you may use the
[pre-commit](https://pre-commit.com/) framework to perform the same check used
by the github CI:

```
git add docs -u  # pre-commit runs on files in git staging.
pre-commit run jupytext
```

### Creating new notebooks

If you are adding a new notebook to the documentation and would like to use the `jupytext --sync`
command discussed here, you can set up your notebook for jupytext by using the following command:

```
jupytext --set-formats ipynb,md:myst path/to/the/notebook.ipynb
```

This works by adding a `"jupytext"` metadata field to the notebook file which specifies the
desired formats, and which the `jupytext --sync` command recognizes when invoked.

### Notebooks within the sphinx build

Some of the notebooks are built automatically as part of the pre-submit checks and
as part of the [Read the docs](https://jax.readthedocs.io/en/latest) build.
The build will fail if cells raise errors. If the errors are intentional, you can either catch them,
or tag the cell with `raises-exceptions` metadata ([example PR](https://github.com/google/jax/pull/2402/files)).
You have to add this metadata by hand in the `.ipynb` file. It will be preserved when somebody else
re-saves the notebook.

We exclude some notebooks from the build, e.g., because they contain long computations.
See `exclude_patterns` in [conf.py](https://github.com/google/jax/blob/main/docs/conf.py).

## Documentation building on readthedocs.io

JAX's auto-generated documentation is at <https://jax.readthedocs.io/>.

The documentation building is controlled for the entire project by the
[readthedocs JAX settings](https://readthedocs.org/dashboard/jax). The current settings
trigger a documentation build as soon as code is pushed to the GitHub `main` branch.
For each code version, the building process is driven by the
`.readthedocs.yml` and the `docs/conf.py` configuration files.

For each automated documentation build you can see the
[documentation build logs](https://readthedocs.org/projects/jax/builds/).

If you want to test the documentation generation on Readthedocs, you can push code to the `test-docs`
branch. That branch is also built automatically, and you can
see the generated documentation [here](https://jax.readthedocs.io/en/test-docs/). If the documentation build
fails you may want to [wipe the build environment for test-docs](https://docs.readthedocs.io/en/stable/guides/wipe-environment.html).

For a local test, I was able to do it in a fresh directory by replaying the commands
I saw in the Readthedocs logs:

```
mkvirtualenv jax-docs  # A new virtualenv
mkdir jax-docs  # A new directory
cd jax-docs
git clone --no-single-branch --depth 50 https://github.com/google/jax
cd jax
git checkout --force origin/test-docs
git clean -d -f -f
workon jax-docs

python -m pip install --upgrade --no-cache-dir pip
python -m pip install --upgrade --no-cache-dir -I Pygments==2.3.1 setuptools==41.0.1 docutils==0.14 mock==1.0.1 pillow==5.4.1 alabaster>=0.7,<0.8,!=0.7.5 commonmark==0.8.1 recommonmark==0.5.0 'sphinx<2' 'sphinx-rtd-theme<0.5' 'readthedocs-sphinx-ext<1.1'
python -m pip install --exists-action=w --no-cache-dir -r docs/requirements.txt
cd docs
python `which sphinx-build` -T -E -b html -d _build/doctrees-readthedocs -D language=en . _build/html
```

