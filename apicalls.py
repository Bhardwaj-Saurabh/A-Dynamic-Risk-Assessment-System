import requests
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# load the config file
with open('config.json', 'r') as file:
    config = json.load(file)

test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path'])

# Call each API endpoint and store the responses
logging.info("Calling each API endpoint...")
response_pred = requests.post(f"{URL}/prediction",
                          json={"filepath": os.path.join(test_data_path, 'testdata.csv')}).text
response_score = requests.get(f"{URL}/scoring").text
response_summary = requests.get(f"{URL}/summarystats").text
response_diagnostics = requests.get(f"{URL}/diagnostics").text

# Combine all API responses
responses = response_pred + "\n" + response_score + "\n" + response_summary + "\n" + response_diagnostics

# save the responses to the text file
logging.info("Saving responses to text file...")
with open(os.path.join(model_path, "apireturns.txt"), "w") as file:
    file.write(responses)



