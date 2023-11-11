# Dynamic Risk Assessment System
[ML DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) by Udacity.

## Description
This project is part of Udacity ML DevOps Engineer Nanodegree with end-to-end Machine Learning Model Scoring and Monitoring. The objective is to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's clients. 

in addition, we are also setting up processes to re-train, re-deploy, monitor and report on the ML model.

## Prerequisites
- Python 3 required

## Dependencies
This project dependencies is available in the ```requirements.txt``` file.

## Installation
```
conda create -n venv python=3.8
```

```
conda activate venv
```

```
pip install -r requirements.txt
```

## Project Structure
```
📦Dynamic-Risk-Assessment-System
 ┣
 ┣ 📂data
 ┃ ┣ 📂ingesteddata                 # Contains csv and metadata of the ingested data
 ┃ ┃ ┣ 📜finaldata.csv
 ┃ ┃ ┗ 📜ingestedfiles.txt
 ┃ ┣ 📂practicedata                 # Data used for practice mode initially
 ┃ ┃ ┣ 📜dataset1.csv
 ┃ ┃ ┗ 📜dataset2.csv
 ┃ ┣ 📂sourcedata                   # Data used for production mode
 ┃ ┃ ┣ 📜dataset3.csv
 ┃ ┃ ┗ 📜dataset4.csv
 ┃ ┗ 📂testdata                     # Test data
 ┃ ┃ ┗ 📜testdata.csv
 ┣ 📂model
 ┃ ┣ 📂models                       # Models pickle, score, and reports for production mode
 ┃ ┃ ┣ 📜apireturns.txt
 ┃ ┃ ┣ 📜confusionmatrix.png
 ┃ ┃ ┣ 📜latestscore.txt
 ┃ ┃ ┣ 📜summary_report.pdf
 ┃ ┃ ┗ 📜trainedmodel.pkl
 ┃ ┣ 📂practicemodels               # Models pickle, score, and reports for practice mode
 ┃ ┃ ┣ 📜apireturns.txt
 ┃ ┃ ┣ 📜confusionmatrix.png
 ┃ ┃ ┣ 📜latestscore.txt
 ┃ ┃ ┣ 📜summary_report.pdf
 ┃ ┃ ┗ 📜trainedmodel.pkl
 ┃ ┗ 📂production_deployment        # Deployed models and model metadata needed
 ┃ ┃ ┣ 📜ingestedfiles.txt
 ┃ ┃ ┣ 📜latestscore.txt
 ┃ ┃ ┗ 📜trainedmodel.pkl
 ┣ 📂src
 ┃ ┣ 📜apicalls.py                  # Runs app endpoints
 ┃ ┣ 📜app.py                       # Flask app
 ┃ ┣ 📜config.py                    # Config file for the project which depends on config.json
 ┃ ┣ 📜deployment.py                # Model deployment script
 ┃ ┣ 📜diagnostics.py               # Model diagnostics script
 ┃ ┣ 📜fullprocess.py               # Process automation
 ┃ ┣ 📜ingestion.py                 # Data ingestion script
 ┃ ┣ 📜pretty_confusion_matrix.py   # Plots confusion matrix
 ┃ ┣ 📜reporting.py                 # Generates confusion matrix and PDF report
 ┃ ┣ 📜scoring.py                   # Scores trained model
 ┃ ┣ 📜training.py                  # Model training
 ┃ ┗ 📜wsgi.py
 ┣ 📜config.json                    # Config json file
 ┣ 📜cronjob.txt                    # Holds cronjob created for automation
 ┣ 📜README.md
 ┗ 📜requirements.txt               # Projects required dependencies
```

## Steps Overview

<img src="image/fullprocess.jpg" width=550 height=300>

## Usage

### 1- Run data ingestion
```
cd src
python ingestion.py
```
**Artifacts output:**
```
data/ingesteddata/finaldata.csv
data/ingesteddata/ingestedfiles.txt
```

### 2- Model training
```
python training.py
```
**Artifacts output:**
```
models/practicemodels/trainedmodel.pkl
```

###  3- Model scoring 
```
python scoring.py
```
**Artifacts output:**
```
models/practicemodels/latestscore.txt
``` 

### 4- Model deployment
```
python deployment.py
```
**Artifacts output:**
```
models/prod_deployment_path/ingestedfiles.txt
models/prod_deployment_path/trainedmodel.pkl
models/prod_deployment_path/latestscore.txt
``` 

### 5- Run diagnostics
```
python diagnostics.py
```

### 6- Run reporting
```
python reporting.py
```
**Artifacts output:**
```
models/practicemodels/confusionmatrix.png
models/practicemodels/summary_report.pdf
```

### 7- Run Flask App
```
python app.py
```

### 8- Run API endpoints
```
python apicalls.py
```
Artifacts output:
```
models/practicemodels/apireturns.txt
```

### 9- Full process automation
```
python fullprocess.py
```
### 10- Cron job

Start cron service
```
sudo service cron start
```

Edit crontab file
```
sudo crontab -e
```
   - Select **option 3** to edit file using vim text editor
   - Press **i** to insert a cron job
   - Write the cron job in ```cronjob.txt``` which runs ```fullprocces.py``` every 10 mins
   - Save after editing, press **esc key**, then type **:wq** and press enter
  
View crontab file
```
sudo crontab -l
```

## License
Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See ```LICENSE``` for more information.


