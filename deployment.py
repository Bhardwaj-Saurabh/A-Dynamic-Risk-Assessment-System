"""
This module will deploy the final model
"""
import os
import json
import logging
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])

### function for deployment
def store_model_into_pickle(model_path):
    """ 
    Copy the latest pickle file to prod deployment directory 
    """
    #copy the latest pickle file, the latestscore.txt value,
    # and the ingestfiles.txt file into the deployment directory
    logging.info("Creating deployment directory if not exists...")
    if not os.path.exists(prod_deployment_path):
        os.makedirs(prod_deployment_path)

    logging.info("Saving artifacts...")
    for file_name in os.listdir(model_path):
        source_model = os.path.join(model_path, file_name)
        prod_model = os.path.join(prod_deployment_path, file_name)
        shutil.copy(source_model, prod_model)   

    logging.info("Saving text file detail...")
    shutil.copy(
        os.path.join(dataset_csv_path, 'ingestedfiles.txt'),
        os.path.join(prod_deployment_path, 'ingestedfiles.txt'))

    logging.info("Artifacts has been saved to deployment directory")

if __name__ == '__main__':
    store_model_into_pickle(model_path)
        
        

