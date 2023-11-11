"""
This module contains functions for diagnostics
"""

# import libraries
import pandas as pd
import timeit
import os
import sys
import json
import pickle
import logging  
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

### Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

logging.info("Loading config...")
dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_folder_path = os.path.join(config['output_folder_path'])

logging.info("Reading final data...")
finaldata = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))


### Function to get model predictions
def model_predictions(X_test):
    #read the deployed model and a test dataset, 
    # calculate predictions
    # Load model
    logging.info("Loading model...")    
    with open(os.path.join(prod_deployment_path, 
                           "trainedmodel.pkl"), 'rb') as model:
        model = pickle.load(model)

    logging.info("Predicting test data...")
    pred = model.predict(X_test.values)

    return pred #return value should be a list containing all predictions

### Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    logging.info("Calculating summary statistics...")   
    numeric = finaldata.select_dtypes(include='int64')
    stats = numeric.drop(['exited'], axis=1).agg(['mean', 'median', 'std'])
    logging.info("Summary Statistics calculated...")

    return stats #return value should be a list containing all summary statistics


### Function to get missing data
def missing_data():
    # calculate missing data here
    logging.info("Calculating missing data...")
    missing_values_percent = list(finaldata.isnull().sum()/len(finaldata))  
    return missing_values_percent #return a list of missing data and percentage


### Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    total_time_lst = []
    scripts = ['ingestion.py', 'training.py']

    logging.info("Calculating timing...")   
    for process in scripts:
        start_time = timeit.default_timer()

        logging.info(f"Running {process}...")   
        os.system(f'python3 {process}')
        timing = timeit.default_timer() - start_time
        logging.info(f"Finished {process}...")

        total_time_lst.append(timing)
    logging.info("Timing calculated...")    
    return total_time_lst #return a list of 2 timing values in seconds

### Function to check dependencies
def outdated_packages_list():
    logging.info("Checking for outdated packages...")
    outdated = subprocess.check_output(
        ['pip', 'list', '--outdated']).decode(sys.stdout.encoding)
    logging.info("Outdated packages found...")
    return str(outdated)







    
