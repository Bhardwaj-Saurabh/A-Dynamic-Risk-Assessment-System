"""
This module automates the ML model scoring and monitoring process
"""

# import libraries
import os
import re
import logging
import json
import pandas as pd
from sklearn.metrics import f1_score

import scoring
import training
import ingestion
import reporting
import deployment
import diagnostics

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

input_path = os.path.join(config['input_folder_path'])
prod_directory = os.path.join(config['prod_deployment_path'])
output_folder_path = config['output_folder_path']
artifacts_path = os.path.join(config['output_model_path'])

def main():
    # Check and read new data
    logging.info("Checking for new data...")

    # First, read ingestedfiles.txt
    with open(os.path.join(prod_directory, "ingestedfiles.txt")) as file:
        ingested_files = {line.strip('\n') for line in file.readlines()[1:]}

    # Second, determine whether the source data folder has files that aren't
    # listed in ingestedfiles.txt
    source_files = set(os.listdir(input_path))

    # Deciding whether to proceed, part 1
    # If you found new data, you should proceed. otherwise, do end the process
    # here
    if len(source_files.difference(ingested_files)) == 0:
        logging.info("No new data found")
        return None

    # Ingesting new data
    logging.info("Ingesting new data...")
    ingestion.merge_multiple_dataframe()

    # Checking for model drift
    logging.info("Checking for model drift...")

    # Check whether the score from the deployed model is different from the
    # score from the model that uses the newest ingested data
    with open(os.path.join(prod_directory, "latestscore.txt")) as file:
        deployed_score = float(file.read()[0])

    data_df = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    y = data_df['exited']
    X = data_df.drop(columns=['corporation','exited'], axis=1)

    y_pred = diagnostics.model_predictions(X)
    new_score = f1_score(y.values, y_pred)

    # Deciding whether to proceed, part 2
    logging.info(f"The previous Deployed score is {deployed_score}")
    logging.info(f"The New score is {new_score}")

    # If you found model drift, you should proceed. otherwise, do end the
    # process here
    if(new_score >= deployed_score):
        logging.info("No model drift occurred")
        return None

    # Re-training
    logging.info("Re-training model...")
    training.train_model()
    logging.info("Re-scoring model...")
    scoring.score_model()

    # Re-deployment
    logging.info("Re-deploying model...")

    # If you found evidence for model drift, re-run the deployment.py script
    deployment.store_model_into_pickle(artifacts_path)

    # Diagnostics and reporting
    logging.info("Running diagnostics and reporting...")

    # Run diagnostics.py and reporting.py for the re-deployed model
    reporting.score_model()
    os.system("python apicalls.py")

if __name__ == "__main__":
    main()