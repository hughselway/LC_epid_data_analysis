# Epidemiological analysis of smoking and histological subtypes of lung cancer

## Setup

To set up the Poetry environment, follow these steps:

1. **Install Poetry**: If you don't have Poetry installed, you can install it by following the instructions on the [Poetry website](https://python-poetry.org/docs/#installation).

2. **Install Dependencies**: Navigate to the project directory and install the dependencies using Poetry.

```sh
poetry install
```

## Running the Scripts

The code is set up as a Python module. For example, to generate plots of smoking surveys
over time, open a terminal and run

```sh
poetry run python -m epid_analysis.plot.smoking_surveys
```
