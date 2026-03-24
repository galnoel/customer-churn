# run_pipeline.py
import pandas as pd
import yaml
import logging
from sklearn.preprocessing import LabelEncoder

# Import the main pipeline class and helper functions
from modeler_script import ModelerPipeline
from preprocess import make_preprocessor, get_feature_schema

from datetime import datetime
from pathlib import Path
import inspect

def setup_logging(log_file):
    """Sets up the logging configuration."""
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() # Also print to console
        ]
    )

def main():
    """Main function to run the entire ML pipeline."""
    # 1. Load configuration and setup logging
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_mode = cfg.get('workflow', {}).get('run_mode', 'full')
    preprocess_filename = Path(inspect.getsourcefile(make_preprocessor)).name.replace(".py", "")

    run_folder_name = f"{run_mode}_{preprocess_filename}_{run_timestamp}"


    # 2. Create the main run folder
    base_dir = cfg['outputs']['base_dir']
    run_folder_path = Path(base_dir) / run_folder_name
    run_folder_path.mkdir(parents=True, exist_ok=True)
    
    # 3. Update all output paths in the config to use the new run folder
    for key, path_template in cfg['outputs'].items():
        if "{run_folder}" in str(path_template): # Check if the path needs formatting
             cfg['outputs'][key] = path_template.format(run_folder=run_folder_path)
    # ---------------------------------------------------------
    
    setup_logging(cfg['outputs']['log_file'])
    logging.info("Configuration loaded and logging is set up.")
    
    # --- This will now log the full path to the run folder ---
    logging.info(f"All outputs for this run will be saved in: {run_folder_path}")
    logging.info(f"Using preprocess file: {preprocess_filename}.py")
    
    # try:
    #     # Inspect the imported function to find its source file
    #     preprocess_file_path = inspect.getsourcefile(make_preprocessor)
    #     preprocess_filename = Path(preprocess_file_path).name
    #     logging.info(f"Using preprocess file: {preprocess_filename}")
    # except (TypeError, OSError):
    #     # Fallback if inspect fails
    #     logging.warning("Could not determine preprocess file name automatically.")

    logging.info(f"Run Timestamp: {run_timestamp}")

    # 4. Load data
    logging.info("Loading data...")
    train_df_raw = pd.read_csv(cfg['data']['train_path'])
    test_df_raw = pd.read_csv(cfg['data']['test_path'])
    train_ids = train_df_raw[cfg['data']['id_column']] 
    test_ids = test_df_raw[cfg['data']['id_column']]

    # 5. Prepare dataframes for the pipeline
    logging.info("Preparing dataframes...")
    train_df = train_df_raw.copy()
    test_df = test_df_raw.copy()

    schema = get_feature_schema()
    target_col = schema['target']

    # Use LabelEncoder to convert string targets to integers
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(train_df[target_col])

    X_train = train_df.drop(columns=[target_col, cfg['data']['id_column']], errors='ignore')
    X_test = test_df.drop(columns=[cfg['data']['id_column']], errors='ignore')
    
    # Align columns to prevent errors during prediction
    X_test = X_test[X_train.columns]

    # 6. Initialize and run the pipeline
    logging.info("Initializing and running the ModelerPipeline.")
    schema = get_feature_schema()
    pipeline = ModelerPipeline(cfg, schema=schema, make_preprocessor_func=make_preprocessor)

    run_mode = cfg.get('workflow', {}).get('run_mode', 'full')
    logging.info(f"Running pipeline in '{run_mode}' mode.")
    pipeline.run(X_train, y_train_encoded, X_test, train_ids, test_ids, le, mode=run_mode)

if __name__ == '__main__':
    main()