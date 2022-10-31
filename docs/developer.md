# For developers

## Running the tests

pytest

```sh
poetry run task tests
```

### Linting

flake, mypy

```sh
poetry run task check
```

## Updating the documentation

To rebuild the documentation, install several packages:
```
pip install -r docs/requirements.txt
```
And then run:
```
cd docs; make html
```
You can then see the generated documentation in `docs/build/html/index.html`.

The `-j auto` option controls the parallelism of the build. You can use a number
in place of `auto` to control how many CPU cores to use.


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
[.pre-commit-config.yaml](https://github.com/NicolasDurrande/guepard/blob/main/.pre-commit-config.yaml).

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

Some of the notebooks are built automatically by the CI.
The build will fail if cells raise errors. If the errors are intentional, you can either catch them,
or tag the cell with `raises-exceptions` metadata ([example PR](https://github.com/google/jax/pull/2402/files)).
You have to add this metadata by hand in the `.ipynb` file. It will be preserved when somebody else
re-saves the notebook.

We exclude some notebooks from the build, e.g., because they contain long computations.
See `exclude_patterns` in [conf.py](https://github.com/google/jax/blob/main/docs/conf.py).

The auto-generated documentation is hosted by GitHub pages and configured at `.github/workflows/deploy.yml`.
