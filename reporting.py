"""
This module contains functions for reporting
"""

# import libraries
import matplotlib.pyplot as plt
import json
import os
import logging
from diagnostics import model_predictions
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

logging.info("Loading config...")
dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])


############## Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    logging.info("Loading test data...")
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))  
    y_test = test_data['exited'].values.reshape(-1, 1).ravel()  
    X_test = test_data.drop(columns=['corporation','exited'], axis=1)

    logging.info("Generating predictions...")
    y_pred = model_predictions(X_test)
    
    # Create a heatmap using Seaborn
    logging.info("Creating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True) 
    plt.title('Model Confusion Matrix')

    logging.info("Saving confusion matrix...")
    plt.savefig(os.path.join(output_model_path, "confusionmatrix.png"))

    logging.info("Confusion matrix saved")

if __name__ == '__main__':
    score_model()
