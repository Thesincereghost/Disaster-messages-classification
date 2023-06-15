# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Running the project](#Running_the_project)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

+ Python version - 3.6
+ Libraries used - pandas, plotly, nltk, flask, sqlalchemy, sys, numpy, scikit-learn

## Project Motivation<a name="motivation"></a>

Just as a disaster hits, the organizations responding are bombarded with a high volume of messages, especialy when they have the least amount of time to pick the most important ones and categorize them. Since, different  organizations handles differnet parts of the effects of the disaster,effective categorization of messages will be of great value.

The aim of this project is to provide a way to easily classify a message recieved during a disaster event, based on a classifier pre-trained on past disaster data from Appen. The input message is taken through a Flask web app, which also has some useful visualizations based on past data. The Web app passes the message to a pre-trained classifier and gives out which categories the message falls into.


## File Descriptions <a name="files"></a>

+ data/disaster_categories.csv data/disaster_messages.csv: Disater messages and categorzation datasets
+ data/process_data.py: Pipeline to extract, transform and load the message and category datasets into a SQL DB
+ model/train_classifier.py: Pipeline to train a model based on the DB created by the ETL pipeline, and save it down inot a pickle file
+ model/classifier.pkl: Pickle file storing the pre trained model
+ app/templates/go.html: Webpage to recive user input and present predictions
+ app/templates/master.html: Webpage to show useful visualizations
+ app/run.py: Main code of the Flask app
+ notebooks/ETL_Pipeline_Preparation.ipynb: Notebook used to prepare the ETL pipeline
+ notebooks/ML_Pipeline_Preparation.ipynb: Notebook used to prepare the ML pipeline

## Running the project <a name="Running_the_project"></a>

 + To run ETL pipeline that cleans the data and stores it in an SQL database -
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
 + To run the ML pipeline that trains a classifier and saves it as a pickle file
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
 + To run the Flask App -
        `cd app` and then `python run.py`


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The datasets were provided by Appen(formally Figure 8). Some of the functions used in the code were from the lessons in the Udacity Data Scientist Nanodegree program.
