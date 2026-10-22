# Fretting FEM Regime Classification ML API

A finite-element simulation and machine-learning framework for classifying fretting slip regimes in a ball-on-flat contact configuration.

This repository contains the machine-learning classification and API deployment workflow associated with research on integrating finite-element modelling with machine learning for rapid prediction of fretting contact behaviour.

## Overview

High-fidelity finite-element simulations can reproduce complex fretting contact behaviour but require considerable computational time. This project uses FEM-generated data to train machine-learning classifiers for rapidly identifying the fretting slip regime under previously unseen operating conditions.

The classifier predicts one of three regimes:

- **Gross Slip Regime (GSR)**
- **Mixed Slip Regime (MSR)**
- **Partial Slip Regime (PSR)**

The selected trained model is saved using Joblib and deployed through a FastAPI REST API.

## Research Workflow

Abaqus FEM simulations
→ FEM-generated dataset
→ Data exploration
→ Physics-informed feature engineering
→ Train-test split
→ Repeated stratified cross-validation
→ Hyperparameter optimization
→ Model comparison
→ Best classifier selection
→ Joblib serialization
→ FastAPI deployment
→ Fretting regime prediction

## Machine-Learning Models

The following classification algorithms are evaluated:

- Logistic Regression
- Support Vector Classifier (SVC)
- Random Forest Classifier
- XGBoost Classifier

Hyperparameter optimization is performed using GridSearchCV and RandomizedSearchCV.

Macro-F1 is used as the main model-selection metric so that the three fretting regimes contribute equally to model evaluation.

Repeated Stratified K-Fold cross-validation is used during model development.
## Input Features

The primary FEM-derived input variables are:

- `P` - Applied normal load (N)
- `D` - Displacement amplitude
- `µ` - Coefficient of friction
- `a` - Hertzian contact radius

Additional physics-informed features are calculated:

`µP = µ × P`

`µPD = µ × P × D`

`D_over_a = D / a`

The final feature order used by the classifier is:

- `P`
- `D`
- `µ`
- `a`
- `µP`
- `µPD`
- `D_over_a`

## Output Classes

The trained model classifies the contact condition as:

`0` → Gross Slip

`1` → Mixed Slip

`2` → Partial Slip

## Project Structure

    fretting-fem-regimeclassification-ml-api/
    |
    |-- data/
    |   `-- ML_fretting_screening.csv
    |
    |-- src/
    |   |-- api.py
    |   |-- explore.py
    |   |-- train_classification.py
    |   |-- streamlit_app.py
    |   |
    |   `-- fretting_fem_regimeclassification_ml_api/
    |       `-- __init__.py
    |
    |-- fretting_regime_classifier.joblib
    |-- .python-version
    |-- pyproject.toml
    |-- uv.lock
    `-- README.md

### Main Files

`src/explore.py`

Used for dataset loading, inspection, and exploratory data analysis.

`src/train_classification.py`

Used for feature engineering, train-test splitting, cross-validation, hyperparameter optimization, model evaluation, model selection, and Joblib serialization.

`src/api.py`

Contains the FastAPI application used to obtain predictions from the trained classifier.

`fretting_regime_classifier.joblib`

Contains the fitted classification pipeline used for API inference.

`src/streamlit_app.py`

Reserved for the Streamlit user interface.
## Environment Management

This project uses `uv` for Python environment and dependency management.

A separate `requirements.txt` file is therefore not required.

The project dependencies are managed through:

- `pyproject.toml`
- `uv.lock`

## Installation

Clone the repository:

    git clone https://github.com/avneeshDS/fretting-fem-regimeclassification-ml-api.git

Enter the repository:

    cd fretting-fem-regimeclassification-ml-api

Create/synchronize the environment:

    uv sync

## Running the Project

### Data Exploration

    uv run python src/explore.py

### Model Training

    uv run python src/train_classification.py

After training and model selection, the selected machine-learning pipeline is saved as:

    fretting_regime_classifier.joblib

## Running the FastAPI Application

From the repository root, run:

    uv run python -m uvicorn src.api:app --reload

The API runs locally at:

    http://127.0.0.1:8000

Swagger documentation:

    http://127.0.0.1:8000/docs

ReDoc documentation:

    http://127.0.0.1:8000/redoc

## API Endpoints

### Health Check

    GET /health

Used to verify that the API and classification model are available.

### Prediction

    POST /predict

Example request:

    {
      "P": 50.0,
      "D": 0.025,
      "µ": 0.5,
      "a": 0.11804
    }

The API automatically calculates:

- `µP`
- `µPD`
- `D_over_a`

and returns the predicted fretting regime.

Example response structure:

    {
      "predicted_regime": "Gross Slip",
      "derived_features": {
        "µP": 25.0,
        "µPD": 0.625,
        "D_over_a": 0.21179
      }
    }

Note: The example response illustrates the API response format. The actual regime is determined by the trained model.

## Model Serialization

The selected machine-learning pipeline is serialized using Joblib.

Example:

    joblib.dump(best_model, "fretting_regime_classifier.joblib")

The FastAPI application loads the trained model for inference using Joblib.

## Code Quality

The project can use Ruff for code formatting and linting.

Format the project:

    uv run ruff format .

Check the code:

    uv run ruff check .

Automatically fix supported issues:

    uv run ruff check . --fix

## Unit Testing

Pytest can be used for automated testing of feature engineering and API functionality.

Run all tests:

    uv run pytest

Run tests with coverage:

    uv run pytest --cov=. --cov-report=term-missing

## Reproducibility

The project follows the following practices:

- Python dependencies are managed using `uv`.
- Dependency versions are recorded in `uv.lock`.
- Fixed random states are used where applicable.
- A separate test dataset is retained for independent model evaluation.
- Repeated stratified cross-validation is used during model development.
- Hyperparameter optimization is performed using only the training dataset.
- Physics-informed features are calculated consistently during training and API inference.
- The selected fitted model pipeline is serialized using Joblib.
- The independent test dataset is not used for hyperparameter selection.

## Technologies

The project uses:

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Joblib
- FastAPI
- Pydantic
- Uvicorn
- uv
- Ruff
- pytest
- Abaqus FEM
## Associated Research

This repository supports research integrating finite-element modelling and machine learning for rapid prediction of fretting contact behaviour.

The broader research framework investigates:

1. Dissipated-energy prediction
2. Fretting slip-regime classification

This repository specifically focuses on the fretting slip-regime classification and API deployment workflow.

## Publication

**Finite-element and machine-learning framework for dissipated-energy prediction and slip-regime classification in fretting contacts**

Avneesh Kumar, Marek Vostrak, and Sarka Houdkova

*Results in Engineering*, Volume 32, 2026, Article 1113296

DOI:

https://doi.org/10.1016/j.rineng.2026.1113296

## Citation

If you use this repository or methodology in academic work, please cite the associated publication:

    @article{Kumar2026FEMMLFretting,
      title   = {Finite-element and machine-learning framework for dissipated-energy prediction and slip-regime classification in fretting contacts},
      author  = {Kumar, Avneesh and Vostrak, Marek and Houdkova, Sarka},
      journal = {Results in Engineering},
      volume  = {32},
      pages   = {1113296},
      year    = {2026},
      doi     = {10.1016/j.rineng.2026.1113296},
      url     = {https://doi.org/10.1016/j.rineng.2026.1113296}
    }

## Data Availability

The machine-learning dataset was generated from finite-element simulations performed as part of the associated research.

Please refer to the published article for the complete data-availability statement.

## License

The license of the published article and the software contained in this repository should be treated separately.

Refer to the repository `LICENSE` file for the software license when available.

## Acknowledgements

The associated research was carried out as part of the OP JAK Inter-sectoral Cooperation for ITI project:

**European Laboratory of Applied Research of Additive Technologies for Thermal Spraying**

Project number:

`CZ.02.01.01/00/23_021/0008594`

Supported by the European Union.

