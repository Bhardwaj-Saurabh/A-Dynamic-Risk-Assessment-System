import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")    


### Load config.json and get input and output paths ###
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

### Function for data ingestion ###
def merge_multiple_dataframe():
    # Check for datasets, compile them together, and write to an output file
    # Get the current directory
    current_path = os.getcwd()

    file_lst = []
    file_names = os.listdir(input_folder_path)
    for file_name in file_names:

        if file_name.endswith(".csv"):
            file_lst.append(os.path.join(current_path, 
                                         input_folder_path, 
                                         file_name))

    df = pd.DataFrame(
        columns=[
            "corporation",
            "lastmonth_activity",
            "lastyear_activity",
            "number_of_employees",
            "exited",
        ]
    )
    logging.info("Reading datasets...")
    for file in file_lst:
        df_temp = pd.read_csv(file)
        df = pd.concat([df, df_temp])

    # Remove duplicates
    logging.info("Removing duplicates...") 
    clean_df = df.drop_duplicates()

    # Save to CSV
    logging.info("Saving to CSV...")    
    clean_df.to_csv(os.path.join(output_folder_path, "finaldata.csv"), index=False)

    # Save the record
    with open(f"{output_folder_path}/ingestedfiles.txt", "w") as f:
        for file_name in file_names:
            if file_name.endswith(".csv"):
                f.write(file_name)
                f.write("\n")


if __name__ == '__main__':
    merge_multiple_dataframe()
