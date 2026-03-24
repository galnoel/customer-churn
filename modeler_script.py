# modeler_script.py
import numpy as np
import pandas as pd
import pickle
import logging
import joblib
from pathlib import Path
import warnings
from sklearn.calibration import cross_val_predict
from tqdm import tqdm # <--- IMPORT TQDM

# Scikit-learn utilities
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import optuna
from optuna.integration import OptunaSearchCV

from sklearn.base import clone



# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelerPipeline:
    """
    An object-oriented pipeline for a complete machine learning workflow.
    Handles feature preprocessing, baseline evaluation, tuning, ensembling, and prediction.
    Now driven by a configuration file.
    """
    def __init__(self, cfg, schema, make_preprocessor_func):
        self.cfg = cfg
        self.schema = schema
        self.make_preprocessor = make_preprocessor_func
        self.report = {}
        self.best_estimators_ = {}
        self.final_model = None
        self.label_encoder = None # Will be passed during run
        self.train_ids = None

    def run(self, X_train, y_train_encoded, X_test, train_ids, test_ids, label_encoder, mode="full"):
        """Executes the entire modeling workflow."""
        self.X_train = X_train
        self.y_train_encoded = y_train_encoded
        self.X_test = X_test
        self.test_ids = test_ids
        self.label_encoder = label_encoder
        self.train_ids = train_ids

        # --- Step 1 is now common to both modes ---
        baseline_report = self.evaluate_baseline_models()
        best_baseline_name = baseline_report.index[0]
        logging.info(f"   Best baseline model found: {best_baseline_name} with Score: {baseline_report.iloc[0]:.4f}")

        model_name_for_submission = ""

        # --- Conditional workflow based on mode ---
        if mode == 'full':
            logging.info("===== Running FULL workflow (Tuning and Ensembling) =====")
            top_3_models = baseline_report.head(3).index.tolist()
            self.tune_top_models(top_3_models)
            self.evaluate_ensemble()
            model_name_for_submission = "Ensemble"

        elif mode == 'baseline_only':
            logging.info(f"===== Running BASELINE_ONLY workflow (using best model: {best_baseline_name}) =====")
            # Create a pipeline for the single best model
            should_scale = best_baseline_name in ['LogisticRegression', 'KNeighbors', 'SVC', 'GaussianNB']
            best_model_pipeline = Pipeline(steps=[
                ('preprocessor', self.make_preprocessor(self.schema, scale_numeric=should_scale)),
                ('classifier', self._get_models()[best_baseline_name])
            ])
            self.final_model = best_model_pipeline
            model_name_for_submission = best_baseline_name
        
        else:
            raise ValueError(f"Invalid run mode: '{mode}'. Choose from ['full', 'baseline_only'].")

        # top_3_models = self.evaluate_baseline_models()
        # self.tune_top_models(top_3_models)
        # self.evaluate_ensemble()
        self.generate_submission(model_name=model_name_for_submission)
        self.analyze_feature_importance()
        self._save_artifacts()
        self.visualize_confusion_matrix()
        
        logging.info("===== Pipeline execution finished successfully. =====")

    def _get_models(self):
        """Returns a dictionary of all models to evaluate as baselines."""
        return {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=2000),
            'KNeighbors': KNeighborsClassifier(),
            # 'SVC': SVC(probability=True, random_state=42),
            'GaussianNB': GaussianNB(),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbosity=-1, n_jobs=-1),
            'CatBoost': cb.CatBoostClassifier(random_state=42, verbose=0)
        }

    def _get_hyperparameter_grids(self):
        """Returns hyperparameter search spaces for tunable models."""
        return {
            'LogisticRegression': {
                'classifier__C': [0.01, 0.1, 1, 10, 100]
            },
            'GaussianNB': {
                'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            },
            'RandomForest': {
                'classifier__n_estimators': [100, 200, 300, 500],
                'classifier__max_depth': [10, 20, 30, None],
                'classifier__min_samples_split': [2, 5, 10],
            },
            'XGBoost': {
                'classifier__n_estimators': [100, 200, 500],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__max_depth': [3, 5, 7],
            },
            'LightGBM': {
                'classifier__n_estimators': [100, 200, 500],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__num_leaves': [20, 31, 50],
            },
            'CatBoost': {
                'classifier__iterations': [100, 200, 500],
                'classifier__learning_rate': [0.01, 0.05, 0.1],
                'classifier__depth': [4, 6, 8],
            },
             'SVC': {
                 'classifier__C': [0.1, 1, 10],
                 'classifier__gamma': [0.1, 0.01, 0.001],
                 'classifier__kernel': ['rbf']
            }
        }
    
    def _get_optuna_grids(self):
        """Returns Optuna-specific hyperparameter search spaces."""
        return {
            'LogisticRegression': {
                'classifier__C': optuna.distributions.FloatDistribution(0.01, 100, log=True),
            },
            'GaussianNB': {
                'classifier__var_smoothing': optuna.distributions.FloatDistribution(1e-9, 1e-5, log=True),
            },
            'RandomForest': {
                'classifier__n_estimators': optuna.distributions.IntDistribution(100, 1000, step=100),
                'classifier__max_depth': optuna.distributions.IntDistribution(10, 50, log=True),
                'classifier__min_samples_split': optuna.distributions.IntDistribution(2, 20),
            },
            'XGBoost': {
                'classifier__n_estimators': optuna.distributions.IntDistribution(100, 1000, step=100),
                'classifier__learning_rate': optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
                'classifier__max_depth': optuna.distributions.IntDistribution(3, 10),
            },
            'LightGBM': {
                'classifier__n_estimators': optuna.distributions.IntDistribution(100, 1000, step=100),
                'classifier__learning_rate': optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
                'classifier__num_leaves': optuna.distributions.IntDistribution(20, 50),
            },
            'CatBoost': {
                'classifier__iterations': optuna.distributions.IntDistribution(100, 1000, step=100),
                'classifier__learning_rate': optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
                'classifier__depth': optuna.distributions.IntDistribution(4, 10),
            },
        }
    def _generate_and_save_oof(self, model_pipeline, model_name: str, stage: str):
        """
        A helper method to perform a CV loop, generate OOF predictions,
        save them, and return the mean F1 score.
        """
        logging.info(f"   Generating OOF for '{model_name}' ({stage} stage)...")
        
        oof_predictions = []
        oof_indices = []
        fold_scores = []
        fold_map = np.zeros(len(self.X_train))

        cv_strategy = StratifiedKFold(n_splits=self.cfg['pipeline_params']['n_folds'], shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(self.X_train, self.y_train_encoded)):
            X_train_fold, X_val_fold = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_fold, y_val_fold = self.y_train_encoded[train_idx], self.y_train_encoded[val_idx]
            
            model_clone = clone(model_pipeline)
            model_clone.fit(X_train_fold, y_train_fold)
            
            preds_proba = model_clone.predict_proba(X_val_fold)
            preds_class = np.argmax(preds_proba, axis=1) # Get class predictions for scoring

            # Calculate score for the fold based on config metric
            scoring_metric = self.cfg['pipeline_params'].get('scoring_metric', 'f1_macro')
            if scoring_metric == 'accuracy':
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_val_fold, preds_class)
            else:
                from sklearn.metrics import f1_score
                score = f1_score(y_val_fold, preds_class, average='macro')
                
            fold_scores.append(score)

            oof_predictions.append(preds_proba)
            oof_indices.extend(val_idx)
            fold_map[val_idx] = fold

        # --- Assemble the OOF DataFrame ---
        oof_preds_full = np.concatenate(oof_predictions, axis=0)
        oof_df = pd.DataFrame(oof_preds_full, columns=[f"proba_{c}" for c in self.label_encoder.classes_])
        
        # Reorder to match original dataframe index
        oof_df = oof_df.iloc[np.argsort(oof_indices)]
        oof_df.index = self.X_train.index

        oof_df['id'] = self.train_ids
        oof_df['y_true'] = self.label_encoder.inverse_transform(self.y_train_encoded)
        oof_df['fold'] = fold_map.astype(int)

        final_cols = ['id', 'fold', 'y_true'] + [col for col in oof_df if col.startswith('proba_')]
        oof_df = oof_df[final_cols]
        
        # Save to a uniquely named CSV
        oof_filename = f"oof_{stage}_{model_name}.csv"
        run_folder = Path(self.cfg['outputs']['model_path']).parent
        oof_path = run_folder / oof_filename
        oof_df.to_csv(oof_path, index=False)
        logging.info(f"     OOF predictions saved to '{oof_path}'")
        
        # Save latest OOF for confusion matrix
        aligned_preds_proba = oof_df[[col for col in oof_df if col.startswith('proba_')]].values
        self.latest_oof_preds_class = np.argmax(aligned_preds_proba, axis=1)

        return fold_scores   

    def evaluate_baseline_models(self):
        """Evaluates all base models using cross-validation."""
        logging.info("--- 1. Evaluating Baseline Models ---")
        models = self._get_models()
        results = {}
        cv_strategy = StratifiedKFold(n_splits=self.cfg['pipeline_params']['n_folds'], shuffle=True, random_state=42)

        # --- CHANGE HERE: WRAP THE LOOP WITH TQDM ---
        for name, model in tqdm(models.items(), desc="Evaluating Baselines"):
            should_scale = name in ['LogisticRegression', 'KNeighbors', 'SVC', 'GaussianNB']
            pipeline = Pipeline(steps=[
                ('preprocessor', self.make_preprocessor(self.schema, scale_numeric=should_scale)),
                ('classifier', model)
            ])
            
            # scores = cross_val_score(pipeline, self.X_train, self.y_train_encoded, cv=cv_strategy, scoring='f1_macro', n_jobs=-1)
            fold_scores = self._generate_and_save_oof(pipeline, name, stage="baseline")
            mean_f1 = np.mean(fold_scores)
            results[name] = mean_f1
            logging.info(f"   {name}: Mean Score = {results[name]:.4f}")

        self.report['baseline_performance'] = pd.Series(results).sort_values(ascending=False)
        logging.info("   Baseline evaluation complete.\n")
        # return self.report['baseline_performance'].head(3).index.tolist()
        return self.report['baseline_performance']

    def tune_top_models(self, top_model_names):
        """Tunes hyperparameters for the best-performing models."""
        logging.info("--- 2. Tuning Top 3 Models ---")
        
        tuner_type = self.cfg['pipeline_params'].get('tuner', 'random_search')
        logging.info(f"   Using tuner: {tuner_type}")   
        if tuner_type == 'optuna':
            param_grids = self._get_optuna_grids()
        else:
            param_grids = self._get_hyperparameter_grids()
        cv_strategy = StratifiedKFold(n_splits=self.cfg['pipeline_params']['n_folds'], shuffle=True, random_state=42)
        self.report['tuned_models'] = {}

        # --- CHANGE HERE: WRAP THE LOOP WITH TQDM ---
        for model_name in tqdm(top_model_names, desc="Tuning Top Models"):
            if model_name not in param_grids:
                logging.warning(f"   Skipping {model_name}: No hyperparameter grid defined.")
                should_scale = model_name in ['LogisticRegression', 'KNeighbors', 'SVC', 'GaussianNB']
                self.best_estimators_[model_name] = Pipeline(steps=[
                    ('preprocessor', self.make_preprocessor(self.schema, scale_numeric=should_scale)),
                    ('classifier', self._get_models()[model_name])
                ]).fit(self.X_train, self.y_train_encoded)
                continue

            logging.info(f"   -> Tuning {model_name}...")
            model = self._get_models()[model_name]
            param_grid = param_grids[model_name]
            should_scale = model_name in ['LogisticRegression', 'KNeighbors', 'SVC', 'GaussianNB']

            pipeline = Pipeline(steps=[
                ('preprocessor', self.make_preprocessor(self.schema, scale_numeric=should_scale)),
                ('classifier', model)
            ])

            scoring_metric = self.cfg['pipeline_params'].get('scoring_metric', 'f1_macro')

            if tuner_type == 'optuna':
                search_cv = OptunaSearchCV(
                    estimator=pipeline,
                    param_distributions=param_grid,
                    n_trials=self.cfg['pipeline_params']['n_trials_optuna'],
                    scoring=scoring_metric,
                    cv=cv_strategy,
                    random_state=42,
                    n_jobs=-1
                )
            else: # Default to RandomizedSearchCV
                search_cv = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=param_grid,
                    n_iter=self.cfg['pipeline_params']['n_iter_search'],
                    scoring=scoring_metric,
                    cv=cv_strategy,
                    random_state=42,
                    n_jobs=-1,
                    verbose=0
                )

            search_cv.fit(self.X_train, self.y_train_encoded)
            # ------------------------------------

            self.report['tuned_models'][model_name] = {
                'best_score': search_cv.best_score_,
                'best_params': search_cv.best_params_
            }
            self.best_estimators_[model_name] = search_cv.best_estimator_
            logging.info(f"     Best Score: {search_cv.best_score_:.4f}")

            self._generate_and_save_oof(search_cv.best_estimator_, model_name, stage="tuned")

        logging.info("   Tuning complete.\n")

    def evaluate_ensemble(self):
        """Creates and evaluates an ensemble of the best-tuned models."""
        logging.info("--- 3. Evaluating Ensemble Performance ---")
        if not self.best_estimators_:
            logging.error("   Skipping ensemble: No models were successfully tuned.")
            return

        estimator_list = list(self.best_estimators_.items())
        
        ensemble_model = VotingClassifier(estimators=estimator_list, voting='soft', n_jobs=-1)
        # cv_strategy = StratifiedKFold(n_splits=self.cfg['pipeline_params']['n_folds'], shuffle=True, random_state=42)
        # ensemble_scores = cross_val_score(ensemble_model, self.X_train, self.y_train_encoded, cv=cv_strategy, scoring='f1_macro', n_jobs=-1)
        ensemble_scores = self._generate_and_save_oof(ensemble_model, "Ensemble", stage="ensemble")
        mean_score, std_score = np.mean(ensemble_scores), np.std(ensemble_scores)
        self.report['ensemble_score'] = mean_score
        logging.info(f"   Ensemble Mean CV Score: {mean_score:.4f} +/- {std_score:.4f}")
        logging.info("   Ensemble evaluation complete.\n")
        self.final_model = ensemble_model

    def generate_submission(self, model_name="Ensemble"):
        """Fits the final model and creates the submission file."""
        logging.info(f"--- 4. Generating Submission File using {model_name} ---")
        if self.final_model is None:
            logging.error("   Cannot generate submission: Final model not available.")
            return

        self.final_model.fit(self.X_train, self.y_train_encoded)
        predictions_encoded = self.final_model.predict(self.X_test)

        submission_df = pd.DataFrame({'id': self.test_ids, self.cfg['data']['target_column']: predictions_encoded.astype(int)})
        sub_path = self.cfg['outputs']['submission_file']
        submission_df.to_csv(sub_path, index=False)
        
        logging.info(f"   Submission file '{sub_path}' created successfully.")
        logging.info(f"   Submission Head:\n{submission_df.head().to_string()}")

    def analyze_feature_importance(self):
        """Extracts and saves feature importances from tree-based models in the ensemble."""
        logging.info("--- 5. Analyzing Feature Importance ---")
        if self.final_model is None:
            logging.error("   Cannot analyze features: Final model is not available.")
            return

        all_importances = []
        
        # Check if the final model is an ensemble (VotingClassifier)
        if isinstance(self.final_model, VotingClassifier):
            if not hasattr(self.final_model, 'estimators_'):
                self.final_model.fit(self.X_train, self.y_train_encoded)
            
            for name, model_pipeline in self.final_model.named_estimators_.items():
                if hasattr(model_pipeline.named_steps['classifier'], 'feature_importances_'):
                    try:
                        feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
                        importances = model_pipeline.named_steps['classifier'].feature_importances_
                        importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances, 'model': name})
                        all_importances.append(importance_df)
                    except Exception as e:
                        logging.warning(f"Could not get feature importance for {name}: {e}")

        # Handle a single pipeline model
        elif isinstance(self.final_model, Pipeline):
            model_pipeline = self.final_model
            if not hasattr(model_pipeline.named_steps['classifier'], 'feature_importances_'):
                 logging.warning("   Selected model does not support feature importance.")
                 return
            try:
                # The model must be fitted to get importances
                if not hasattr(model_pipeline.named_steps['classifier'], 'n_features_in_'):
                    model_pipeline.fit(self.X_train, self.y_train_encoded)

                feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
                importances = model_pipeline.named_steps['classifier'].feature_importances_
                importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances, 'model': 'Best Baseline'})
                all_importances.append(importance_df)
            except Exception as e:
                logging.warning(f"Could not get feature importance for the single model: {e}")

        if not all_importances:
            logging.warning("   No models with feature importance found.")
            return

        # Combine and average the importances (for ensembles, this averages; for single, it just processes)
        combined_df = pd.concat(all_importances)
        avg_importance_df = combined_df.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()

        fi_path = self.cfg['outputs']['feature_importance_path']
        avg_importance_df.to_csv(fi_path, index=False)
        logging.info(f"   Saved average feature importances to '{fi_path}'.")
        logging.info(f"   Top 10 Features:\n{avg_importance_df.head(10).to_string()}")

        self.visualize_feature_importance(avg_importance_df)


    def _save_artifacts(self):
        """Saves the final model, label encoder, and report to disk."""
        logging.info("--- 6. Saving Artifacts ---")
        try:
            joblib.dump(self.final_model, self.cfg['outputs']['model_path'])
            logging.info(f"   Final model saved to: {self.cfg['outputs']['model_path']}")
            
            joblib.dump(self.label_encoder, self.cfg['outputs']['label_encoder_path'])
            logging.info(f"   Label encoder saved to: {self.cfg['outputs']['label_encoder_path']}")

            with open(self.cfg['outputs']['report_path'], 'wb') as f:
                pickle.dump(self.report, f)
            logging.info(f"   Performance report saved to: {self.cfg['outputs']['report_path']}")

        except Exception as e:
            logging.error(f"   Error saving artifacts: {e}")

    def visualize_feature_importance(self, avg_importance_df):
        """Creates and saves a bar plot of the top N most important features."""
        logging.info("--- 7. Visualizing Feature Importance ---")
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot top 20 features
            top_features = avg_importance_df.head(20)
            
            sns.barplot(
                x="importance",
                y="feature",
                data=top_features,
                palette="viridis"
            )
            
            plt.title('Top 20 Most Important Features (Averaged Across Models)')
            plt.xlabel('Average Importance Score')
            plt.ylabel('Feature Name')
            plt.tight_layout() # Adjust layout to make room for feature names
            
            plot_path = self.cfg['outputs']['feature_importance_plot_path']
            plt.savefig(plot_path)
            plt.close() # Close the plot to free up memory
            logging.info(f"   Feature importance plot saved to '{plot_path}'")
        
        except Exception as e:
            logging.error(f"   Could not create feature importance plot: {e}")

    def visualize_confusion_matrix(self):
        """Creates and saves a confusion matrix plot for the final model's training predictions."""
        logging.info("--- 8. Visualizing Confusion Matrix ---")
        if self.final_model is None:
            logging.error("   Cannot create confusion matrix: Final model not available.")
            return
            
        try:
            logging.info("   Reusing out-of-fold predictions for the confusion matrix...")
            
            y_pred_cv = getattr(self, 'latest_oof_preds_class', None)
            if y_pred_cv is None:
                # Fallback if OOF not found
                cv_strategy = StratifiedKFold(n_splits=self.cfg['pipeline_params']['n_folds'], shuffle=True, random_state=42)
                y_pred_cv = cross_val_predict(
                    self.final_model, 
                    self.X_train, 
                    self.y_train_encoded, 
                    cv=cv_strategy, 
                    n_jobs=-1
                )
                
            class_names = self.label_encoder.classes_
            
            # Create the confusion matrix
            cm = confusion_matrix(self.y_train_encoded, y_pred_cv)

            # Plot using ConfusionMatrixDisplay
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            
            fig, ax = plt.subplots(figsize=(10, 10))
            disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
            
            plt.title('Confusion Matrix on Training Data')
            plt.tight_layout()

            plot_path = self.cfg['outputs']['confusion_matrix_plot_path']
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"   Confusion matrix plot saved to '{plot_path}'")
        
        except Exception as e:
            logging.error(f"   Could not create confusion matrix plot: {e}")