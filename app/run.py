import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''
    To convert a string of text into list of tokens by:
    1. Capitalization removal, punctuation removeal and stop word removal
    2. Split into a list of words
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Number of messages for each Genre (Code provided by Udacity)
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Number of messages for each category
    category_names=list(df.iloc[:,4:].columns)
    category_message_counts = list(df[list(df.iloc[:,4:].columns)].sum().values)
    
    #Average number of categories across different Genres of message
    df['Total Categories'] = df[list(df.iloc[:,4:].columns)].sum(axis=1)
    genre_average_categories = df.groupby('genre').mean()['Total Categories']
    

    graphs = [
        # Graph of number of messages for each Genre (Code provided by Udacity)
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Number of messages for each Genre',
                'yaxis': {
                    'title': "Number of messages"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # Graph of number of messages for each category
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_message_counts
                )
            ],

            'layout': {
                'title': 'Number of messages for each category',
                'yaxis': {
                    'title': "Number of messages"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        # Graph of average number of categories across different Genres of message
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_average_categories
                )
            ],

            'layout': {
                'title': 'Average number of categories across Genre',
                'yaxis': {
                    'title': "Average number of Categories"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()