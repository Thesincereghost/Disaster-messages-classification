# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''Load data from SQL database and break into target and feature dataframes
    
    Args:
    database_filepath: path to SQL database from root directory 
    
    Returns:
    X: Dataframe containing features
    Y: Dataframe with target columns
    category_names: Series of all unique categories
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('message_categories', con=engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''
    To convert a string of text into list of tokens by:
    1. Capitalization removal, punctuation removeal 
    2. Split into a list of words and remove stop words
    3. Reduce the words to their root forms by lemmitization 
    
    Args:
    text: String to be tokenized
    
    Returns:
    clean_tokens: Tokenized data
    '''
    #Replace anything that isn't A-Z or )-9 with a space. 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())
    # Split text into words using NLTK
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    # Lemmatize verbs by specifying pos
    cleaned_tokens = [WordNetLemmatizer().lemmatize(t, pos='v') for t in lemmed]

    return cleaned_tokens 


def build_model():
    '''
    To create the machine learning pipeline object with setup for parameter tuning using GridSearch
    
    Returns:
    model : Pipeline object
    '''
    
    #Initialize pipeline object with transformers and classifier
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
            ])
    #Initialize parametrs to be fed into GridSearch
    parameters = {
    'vect__ngram_range' : ((1,1), (1,2)),
    'tfidf__smooth_idf' : [True, False],
    'clf__estimator__max_depth' : [40, 50]
    }
    #Initialize GridSearch object    
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model
    


def evaluate_model(model, X_test, Y_test, category_names):
    ''' 
    Generates the classification report for a given model and test sets
    Args:
    model : Machine learning model object fitted on the training set
    X_test : Dataframe of test features
    Y_test : Dataframe of test targets
    category_names : Series holding all target categories
    
    Returns:
    report : Classification report showing precision, recll and F-1 score for each category
    '''
    y_pred = model.predict(X_test)
    report = classification_report(Y_test, y_pred, target_names=category_names)
    return report


def save_model(model, model_filepath):
    ''' 
    Save model as a pickle file
    
    Args:
    model: Machine learning model to be saved
    model_filepath: file path to where the pickle file is to be stored
    '''
    pickle.dump(cv, open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        report=evaluate_model(model, X_test, Y_test, category_names)
        print(report)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()