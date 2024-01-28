# Learning Classifier Systems from Scratch

> Workable learning classifier systems from scratch in python


This repository is a collection of Learning Classifier Systems development and experimentation. The goal is to create workable learning classifier systems from scratch in python. This will pull info from multiple sources and classifier systems. Everything from Urbanowicz's video description, to Wilson's ZCS and XCS, and But'z XCSTS and APS2. The main goal for these custom learning classifier systems is not to simply just copy existing LCS designs, but also to try and simplifiy them while maintaining performance. One of the end goals of this LCS research is to create a functional LCS that can be run on a quantum computer. In order to facilitate this, as much math and statistics as possible will be stripped away. For example, Butz's well performing XCSTS uses gradient descent for reward propication, Q-Learning, and Bayesian decision trees to try and push performance to higher levels in terms of speed and accuracy. These advances significantly reduce learning time and computional resources. However, the thought here is that even basic LCS that would traditionally be underwhelming when run on a conventional computer will be exceedingly fast when ran on a quantum computer. If the basic LCS design can be distilled down into memory lists, simple logic gates, and random number generators, it should be able to be translated to run on a quantum computer.

A Learning Classifier System is a type of machine learning algorithm originally concieved by Dr. John Holland in 1975 with his Cognitive System 1 (CS1). Whereas neural networks are designed to mimic the physical architecture of the brain, CS1 was designed to mimic how the brain thinks and learns. In its implest description, LCS is a large "if, then" list where the condition-action pairs are evolved over time through a genetic algorithm, another invention of Dr. Holland. Through the evolution of many simple rules, complex interactions can form and the LCS starts to become a complex adaptive system, similar to large scale neural networks. In theory, an intricate enough LCS could become a conscious machine learner that processes information and thinks the way the human mind does. To this end, it is believed by many that consiousness is an inherently quantum property. Additionally, basic LCS use the ternary alphabet of 0, 1, and # where the hash simble is the "don't care" symbol. This alphabet directly corresponds to the posibilities of quantum states i.e., 0, 1, or in superposition. If data and LCS operations can be borken down to solely rely on the ternary alphabet (this should be possible as conventional computers only run on binary), then LCS should perform very well when run on a quantum computer.

## Installation

In order to set up the necessary environment:

1. review and uncomment what you need in `environment.yml` and create an environment `lcs` with the help of [conda]:
   ```
   conda env create -f environment.yml
   ```
2. activate the new environment with:
   ```
   conda activate lcs
   ```

> **_NOTE:_**  The conda environment will have lcs installed in editable mode.
> Some changes, e.g. in `setup.cfg`, might require you to run `pip install -e .` again.


Optional and needed only once after `git clone`:

3. install several [pre-commit] git hooks with:
   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

4. install [nbstripout] git hooks to remove the output cells of committed notebooks with:
   ```bash
   nbstripout --install --attributes notebooks/.gitattributes
   ```
   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.


Then take a look into the `scripts` and `notebooks` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yml` for the exact reproduction of your
   environment with:
   ```bash
   conda env export -n lcs -f environment.lock.yml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yml` using:
   ```bash
   conda env update -f environment.lock.yml --prune
   ```
## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── Dockerfile              <- Build a docker container with `docker build .`.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- [DEPRECATED] Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── src
│   └── lcs                 <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
