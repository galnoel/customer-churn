# 📊 Customer Churn Prediction Pipeline

A production-ready, end-to-end machine learning pipeline for predicting customer churn in a telecommunications company. The pipeline automates data preprocessing, feature engineering, baseline evaluation, hyperparameter tuning, model ensembling, and submission generation — all driven by a single YAML configuration file.

---

## 🏗️ Project Structure

```
customer-churn-prediction/
├── config.yaml                          # Central configuration file
├── run_pipeline.py                      # Entry point — orchestrates the full workflow
├── preprocess.py                        # Feature engineering & preprocessing transformers
├── modeler_script.py                    # Modeling pipeline (baseline, tuning, ensemble)
├── eda-customer-churn-datasets.ipynb    # Exploratory Data Analysis notebook
├── .env                                 # Kaggle API credentials (gitignored)
├── .gitignore
├── data/
│   ├── train.csv                        # Training dataset
│   ├── test.csv                         # Test dataset (for submission)
│   └── sample_submission.csv            # Expected submission format
└── outputs/                             # Auto-generated run folders (gitignored)
    └── {mode}_{preprocess}_{timestamp}/
        ├── pipeline_run.log
        ├── submission.csv
        ├── model.joblib
        ├── label_encoder.joblib
        ├── report.pkl
        ├── feature_importance.csv
        ├── feature_importance.png
        ├── confusion_matrix.png
        └── oof_*.csv                    # Out-of-fold prediction files
```

---

## ✨ Key Features

- **Configuration-Driven** — All paths, model parameters, and workflow modes are controlled via `config.yaml`.
- **Declarative Feature Engineering** — Define derived features, capping rules, and binary mappings in a schema dictionary. No boilerplate code required.
- **Custom Scikit-learn Transformers** — Includes `DomainClipper`, `QuantileCapper`, `DerivedFeatures`, and `BinaryMapTransformer` for robust, reproducible preprocessing.
- **Multi-Model Baseline Evaluation** — Automatically trains and evaluates 8 classifiers (Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost) using stratified K-fold cross-validation.
- **Hyperparameter Tuning** — Supports both `RandomizedSearchCV` and `Optuna` for the top-performing models.
- **Soft-Voting Ensemble** — Combines the top 3 tuned models into a `VotingClassifier` with soft voting.
- **Out-of-Fold (OOF) Predictions** — Generates OOF probability outputs at every stage (baseline, tuned, ensemble) for stacking and analysis.
- **Automated Artifacts** — Each run saves the trained model, label encoder, performance report, feature importance chart, and confusion matrix into a timestamped folder.
- **Dual Run Modes** — Supports `"full"` (tune + ensemble) and `"baseline_only"` (quick evaluation) workflows.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/galnoel/customer-churn.git
   cd customer-churn-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna pyyaml joblib tqdm matplotlib seaborn
   ```

3. **Prepare the data:**
   Place `train.csv`, `test.csv`, and `sample_submission.csv` into the `data/` directory.

4. **(Optional) Set Kaggle credentials:**
   Create a `.env` file with your Kaggle API keys if downloading data programmatically:
   ```
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_api_key
   ```

---

## ⚙️ Configuration

All pipeline behavior is controlled through [`config.yaml`](config.yaml):

```yaml
workflow:
  run_mode: "full"          # "full" (tune + ensemble) or "baseline_only"

data:
  train_path: "data/train.csv"
  test_path: "data/test.csv"
  target_column: "Churn"
  id_column: "id"

pipeline_params:
  n_folds: 5               # Number of CV folds
  n_iter_search: 15         # Iterations for RandomizedSearchCV
  tuner: "random_search"    # "random_search" or "optuna"
  n_trials_optuna: 3        # Trials for Optuna tuner
  scoring_metric: "f1_macro" # Metric to optimize

outputs:
  base_dir: "outputs"
  # All paths use {run_folder} placeholder — resolved at runtime
  submission_file: "{run_folder}/submission.csv"
  model_path: "{run_folder}/model.joblib"
  # ... (see config.yaml for full list)
```

---

## 🏃 Usage

### Run the full pipeline

```bash
python run_pipeline.py
```

This will:

1. Load the configuration and create a timestamped output folder.
2. Read and prepare the training and test datasets.
3. Apply the feature schema (binary mapping, capping, derived features, encoding).
4. Evaluate 8 baseline models with stratified K-fold CV.
5. Tune the top 3 models using the configured tuner.
6. Build a soft-voting ensemble from the tuned models.
7. Generate the submission CSV and save all artifacts.

### Run baseline only (quick mode)

Set `run_mode: "baseline_only"` in `config.yaml`, then:

```bash
python run_pipeline.py
```

This skips tuning and ensembling — it selects the best baseline model, fits it on the full training set, and generates a submission.

---

## 🔧 Pipeline Architecture

### Preprocessing (`preprocess.py`)

The preprocessing pipeline is built as a composable scikit-learn `Pipeline` with a `ColumnTransformer`:

```
Input DataFrame
  │
  ├── clean_values         → Lowercase string values for consistency
  ├── derive_features      → Compute derived columns (e.g., tenure_years, monthly_vs_average)
  │
  └── ColumnTransformer
        ├── num  → DomainClipper → QuantileCapper → (optional: Imputer, Scaler)
        ├── bin  → BinaryMapTransformer (maps Yes/No, Male/Female → 1/0)
        ├── ord  → OrdinalEncoder (if ordinal features are defined)
        └── nom  → OneHotEncoder (for multi-category features)
```

#### Feature Schema

The feature schema in `get_feature_schema()` declaratively defines:

| Key                    | Purpose                                                |
|------------------------|--------------------------------------------------------|
| `numeric`              | Continuous columns to keep as numbers                  |
| `binary_map`           | Columns with binary text values → integer mapping      |
| `categorical_nominal`  | Multi-class columns for one-hot encoding               |
| `categorical_ordinal`  | Ordered categories for ordinal encoding                |
| `derived`              | Computed features using expressions or functions       |
| `capping`              | Outlier handling via domain bounds or quantile caps    |
| `drop`                 | Columns to exclude (e.g., `id`)                        |
| `target`               | Name of the target column                              |

### Modeling (`modeler_script.py`)

The `ModelerPipeline` class manages the complete modeling lifecycle:

| Stage | Method                       | Description                                       |
|-------|------------------------------|---------------------------------------------------|
| 1     | `evaluate_baseline_models()` | CV evaluation of 8 classifiers with OOF outputs   |
| 2     | `tune_top_models()`          | Hyperparameter search on top 3 models              |
| 3     | `evaluate_ensemble()`        | Soft-voting ensemble of tuned models               |
| 4     | `generate_submission()`      | Predict on test set and export CSV                 |
| 5     | `analyze_feature_importance()`| Extract and plot averaged feature importances     |
| 6     | `_save_artifacts()`          | Persist model, encoder, and report to disk         |
| 7     | `visualize_feature_importance()` | Bar chart of top 20 features                  |
| 8     | `visualize_confusion_matrix()` | Confusion matrix from OOF predictions            |

#### Supported Models

| Model                | Library    | Scaling Required |
|----------------------|------------|:----------------:|
| Logistic Regression  | sklearn    | ✅               |
| K-Nearest Neighbors  | sklearn    | ✅               |
| Gaussian Naive Bayes | sklearn    | ✅               |
| Decision Tree        | sklearn    | ❌               |
| Random Forest        | sklearn    | ❌               |
| XGBoost              | xgboost    | ❌               |
| LightGBM             | lightgbm   | ❌               |
| CatBoost             | catboost   | ❌               |

---

## 📈 Outputs

Each run generates a self-contained folder under `outputs/` named with the pattern:

```
{run_mode}_{preprocess_file}_{YYYYMMDD_HHMMSS}/
```

**Example:** `full_preprocess_20260324_172715/`

| File                      | Description                                       |
|---------------------------|---------------------------------------------------|
| `pipeline_run.log`        | Full execution log with timestamps                |
| `submission.csv`          | Final predictions for the test set                |
| `model.joblib`            | Serialized final model (ensemble or single)       |
| `label_encoder.joblib`    | Fitted LabelEncoder for target classes            |
| `report.pkl`              | Dictionary with baseline scores, tuning results   |
| `feature_importance.csv`  | Averaged feature importances                      |
| `feature_importance.png`  | Bar chart visualization of top features           |
| `confusion_matrix.png`    | Confusion matrix from out-of-fold predictions     |
| `oof_baseline_*.csv`      | OOF probabilities for each baseline model         |
| `oof_tuned_*.csv`         | OOF probabilities for each tuned model            |
| `oof_ensemble_*.csv`      | OOF probabilities for the ensemble                |

---

## 📓 Exploratory Data Analysis

The [`eda-customer-churn-datasets.ipynb`](eda-customer-churn-datasets.ipynb) notebook contains:

- Distribution analysis of features and target variable
- Correlation analysis between features
- Visualization of churn patterns across customer segments
- Insights that informed the feature engineering decisions in `preprocess.py`

---

## 🛠️ Extending the Pipeline

### Add a new derived feature

Edit `get_feature_schema()` in `preprocess.py` and add an entry to the `"derived"` list:

```python
{
    "name": "charges_per_month",
    "expr": "`TotalCharges` / (`tenure` + 1)",
    "requires": ["TotalCharges", "tenure"],
    "type": "numeric",
    "on_missing": "skip"
}
```

### Add a new model

1. Import the model in `modeler_script.py`.
2. Add it to `_get_models()`.
3. Add its hyperparameter grid to `_get_hyperparameter_grids()` and/or `_get_optuna_grids()`.

### Switch tuning strategy

In `config.yaml`, change `tuner` to `"optuna"` and adjust `n_trials_optuna`:

```yaml
pipeline_params:
  tuner: "optuna"
  n_trials_optuna: 50
```

---

## 📦 Dependencies

| Package       | Purpose                               |
|---------------|---------------------------------------|
| pandas        | Data manipulation                     |
| numpy         | Numerical operations                  |
| scikit-learn  | ML models, preprocessing, evaluation |
| xgboost       | Gradient boosting                     |
| lightgbm      | Gradient boosting                     |
| catboost      | Gradient boosting                     |
| optuna        | Bayesian hyperparameter optimization  |
| pyyaml        | Configuration file parsing            |
| joblib        | Model serialization                   |
| matplotlib    | Plotting                              |
| seaborn       | Statistical visualizations            |
| tqdm          | Progress bars                         |

---

## 📝 License

This project is for educational and competition purposes.
