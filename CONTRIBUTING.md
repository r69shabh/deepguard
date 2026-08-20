# Contributing to Adaptive Cyber-Physical Security (Anomaly-Based IDS)

First off, thank you for considering contributing to this project! It's people like you that make the open-source community such a great place to learn, inspire, and create.

## Where do I go from here?

If you've noticed a bug or have a feature request, make one! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

## Fork & create a branch

If this is something you think you can fix, then fork the project and create a branch with a descriptive name.

A good branch name would be (where issue #325 is the ticket you're working on):

```sh
git checkout -b 325-add-xgboost-metalearner
```

## Get the test suite running

Make sure you have installed the project in editable mode with development dependencies:

```sh
pip install -e .
pip install pytest
```

Run the test suite to ensure everything is working:

```sh
pytest tests/
```

## Implement your fix or feature

At this point, you're ready to make your changes. Feel free to ask for help; everyone is a beginner at first.

- **Add tests!** Your patch won't be accepted if it doesn't have tests.
- **Keep it focused.** Don't try to change too much in one PR.
- **Update documentation.** If you change the CLI or configuration, update `README.md` and `configs/default.yaml`.

## Make a Pull Request

At this point, you should switch back to your master branch and make sure it's up to date with the main repository:

```sh
git remote add upstream https://github.com/r69shabh/anomaly-based-ids.git
git checkout main
git pull upstream main
```

Then update your feature branch from your local copy of master, and push it!

```sh
git checkout 325-add-xgboost-metalearner
git rebase main
git push --set-upstream origin 325-add-xgboost-metalearner
```

Finally, go to GitHub and make a Pull Request.

## Code Style

- We follow standard PEP-8 conventions. 
- Please type-hint all new functions and methods.
- Include docstrings in NumPy format for any new classes or public functions.
