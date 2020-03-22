import json
import plotly
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Scatter
import joblib
from sqlalchemy import create_engine
import random

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Message_Category', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    #====================
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    #====================
    words_cloud = df.loc[:, ['genre','message']]
    stop_words = stopwords.words('english')
    # Direct word cloud
    txt_direct = words_cloud[words_cloud['genre'].str.contains('direct')]['message'].str.replace(r'[^a-zA-Z0-9]', ' ').str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
    tokens_direct = word_tokenize(txt_direct)
    lemmed_direct = [WordNetLemmatizer().lemmatize(w) for w in tokens_direct if w not in stop_words]
    words_direct = nltk.FreqDist(lemmed_direct)
    results_direct = pd.DataFrame(words_direct.most_common(30), columns=['Word','Count']).set_index('Word')
    results_direct['weights_direct'] = results_direct['Count'].rank()+5
    weights_direct = results_direct['weights_direct']
    direct_names = list(results_direct.index)
    # Social word cloud
    txt_social = words_cloud[words_cloud['genre'].str.contains('social')]['message'].str.replace(r'[^a-zA-Z0-9]', ' ').str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
    tokens_social = word_tokenize(txt_social)
    lemmed_social = [WordNetLemmatizer().lemmatize(w) for w in tokens_social if w not in stop_words]
    words_social = nltk.FreqDist(lemmed_social)
    results_social = pd.DataFrame(words_social.most_common(30), columns=['Word','Count']).set_index('Word')
    results_social['weights_social'] = results_social['Count'].rank()+5
    weights_social = results_social['weights_social']
    social_names = list(results_social.index)
    # News word cloud
    txt_news = words_cloud[words_cloud['genre'].str.contains('news')]['message'].str.replace(r'[^a-zA-Z0-9]', ' ').str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
    tokens_news = word_tokenize(txt_news)
    lemmed_news = [WordNetLemmatizer().lemmatize(w) for w in tokens_news if w not in stop_words]
    words_news = nltk.FreqDist(lemmed_news)
    results_news = pd.DataFrame(words_news.most_common(30), columns=['Word','Count']).set_index('Word')
    results_news['weights_news'] = results_news['Count'].rank()+5
    weights_news = results_news['weights_news']
    news_names = list(results_news.index)
    #====================
    colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i in range(30)]

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # Direct word cloud chart
        {
            'data': [
                Scatter(
                    x = random.choices(range(5,45), k=30),
                    y = random.choices(range(5,45), k=30),
                    mode = 'text',
                    text = direct_names,
                    hoverinfo = 'none',
                    marker={'opacity': 0.3},
                    textfont={'size': weights_direct
                             ,'color': colors}
                )
            ],

            'layout': {
                'title': 'Word Frequency in Direct Category',
                'xaxis': {'showgrid': False
                         ,'linecolor' : 'grey'
                         ,'showticklabels': False
                         ,'zeroline': False
                         ,'range': [0,50]
                         ,'mirror':True
                         ,'showline': True},
                'yaxis': {'showgrid': False
                         ,'linecolor' : 'grey'
                         ,'showticklabels': False
                         ,'range': [0,50]
                         ,'zeroline': False
                         ,'mirror':True
                         ,'showline': True}
            }
        },
        # Social word cloud chart
        {
            'data': [
                Scatter(
                    x = random.choices(range(5,45), k = 30),
                    y = random.choices(range(5,45), k = 30),
                    mode = 'text',
                    text = social_names,
                    hoverinfo = 'none',
                    marker={'opacity': 0.3},
                    textfont={'size': weights_social
                             ,'color': colors}
                )
            ],

            'layout': {
                'title': 'Word Frequency in Social Category',
                'xaxis': {'showgrid': False
                         ,'linecolor' : 'grey'
                         ,'showticklabels': False
                         ,'range': [0,50]
                         ,'zeroline': False
                         ,'mirror':True
                         ,'showline': True},
                'yaxis': {'showgrid': False
                         ,'linecolor' : 'grey'
                         ,'range': [0,50]
                         ,'showticklabels': False
                         ,'zeroline': False
                         ,'mirror':True
                         ,'showline': True}
            }
        },
        # News word cloud chart
        {
            'data': [
                Scatter(
                    x = random.choices(range(5,45), k = 30),
                    y = random.choices(range(5,45), k = 30),
                    mode = 'text',
                    text = news_names,
                    hoverinfo = 'none',
                    marker={'opacity': 0.3},
                    textfont={'size': weights_news
                             ,'color': colors}
                )
            ],

            'layout': {
                'title': 'Word Frequency in News Category',
                'xaxis': {'showgrid': False
                         ,'linecolor' : 'grey'
                         ,'range': [0,50]
                         ,'showticklabels': False
                         ,'zeroline': False
                         ,'mirror':True
                         ,'showline': True},
                'yaxis': {'showgrid': False
                         ,'range': [0,50]
                         ,'linecolor' : 'grey'
                         ,'showticklabels': False
                         ,'zeroline': False
                         ,'mirror':True
                         ,'showline': True}
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
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()