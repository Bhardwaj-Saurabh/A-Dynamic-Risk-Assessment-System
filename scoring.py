"""
This module score the train model on test data
"""

# import libraries
import pandas as pd
import pickle
import os
from sklearn import metrics
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

### Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

logging.info("Loading data and model config ...")
dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_artifacts = os.path.join(config['output_model_path'])

### Function for model scoring
def score_model():
    """
    Score the trained model on test data and save the result to latestscore.txt file. 
    
    The latestscore.txt file is saved in the model_artifacts directory.
    """
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    with open(os.path.join(model_artifacts, 'trainedmodel.pkl'), "rb") as file:
        model = pickle.load(file)

    logging.info("Loading test data...")
    testdata = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    logging.info("Preparing test data as feature and target...")
    X_test = testdata.drop(['corporation','exited'], axis=1)
    y_test = testdata['exited'].values.reshape(-1, 1).ravel()

    logging.info("Predicting test data...")
    pred = model.predict(X_test.values)

    logging.info("Calculating F1 score...")
    f1_score = metrics.f1_score(pred, y_test)

    # Save metrics
    logging.info("Saving metrics...")
    with open(os.path.join(model_artifacts, "latestscore.txt"), 'w') as file:
        file.write(str(f1_score))

    logging.info(f"Scoring: F1={f1_score:.2f}")

    return f1_score

if __name__ == '__main__':
    score_model()

