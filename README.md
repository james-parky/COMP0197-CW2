# COMP0197-CW2

## Install Dependencies
Outside of the packages included in the comp0197-xx-cw2 conda environment several packages are used to ensure good code and repo quality. To install dependencies, run:
```bash
pip install -r dev-requirements.txt
```
The required development dependencies can be found in `dev-requirements.txt`.

## Set up `pre-commit`
This repository uses the `pre-commit` library to enable pre-commit hooks within the project. These hooks allow for various code content and quality checks to be completed when you run `git commit`, ensuring than no un-linted and un-formatted code is persisted to the remote repository. There are two hooks set up:
- Black (formatter)
- Pylint (linter)
You only need to set up `pre-commit` **once** when you clone this repo, by running:
```bash
pre-commit install
```
If you get an error similar to `command not found: pre-commit` then it needs to be added to your `$PATH`. A tutorial for windows or unix-based systems can be found [here](https://graycode.ie/blog/how-to-add-python-pip-to-path/).

## Contribution
Please only push code to a non-main branch. A branch can be created using
```bash
git checkout -b <branch_name>
```
A pull request can then be made into main (`main <- <your branch>`) which will require approval from at least one other team member. The pylint pre-commit hook explained above will also be run again when the merge is attempted.
