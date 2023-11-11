from flask import Flask, request
import pickle
from diagnostics import model_predictions, dataframe_summary, execution_time, outdated_packages_list, missing_data  
import json
import os
import pandas as pd
from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

# logging configs
dataset_csv_path = os.path.join(config['output_folder_path']) 
output_model_path = os.path.join(config['output_model_path'])

# load the model
with open(os.path.join(output_model_path, 'trainedmodel.pkl'), "rb") as model:
    prediction_model = pickle.load(model)

@app.route('/')
def index():
    return "Use this app to run a full ML Pipeline"

####################### Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3 
    filepath = request.get_json()['filepath']

    df = pd.read_csv(filepath)
    df = df.drop(['corporation', 'exited'], axis=1)

    pred = model_predictions(df)
    return str(pred) #add return value for prediction outputs

####################### Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    score = score_model()   
    return str(score) #add return value (a single F1 score number)

####################### Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    stats = dataframe_summary()
    return str(stats) #return a list of all calculated summary statistics

####################### Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    time = execution_time()
    na_values = missing_data()
    outdated = outdated_packages_list()

    return str(time) + '\n' + str(na_values) + '\n' + str(outdated) #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)
