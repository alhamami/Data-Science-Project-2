import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


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
df = pd.read_sql_table('disaster_response_df', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    The function will make graphs 
    
    Returns:
    Render web page with plotly graphs
    """

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    tokens = []
    all = []
    vis1 = []
    vis2 = []
    vis3 = []
    title1 ='DISTRIBUTION OF MESSAGE GENRES'
    title2 = 'TOTAL NUMBER'
    title3 = 'MOST FREQUENT WORDS'
    x1 = dict(title = 'GENRE')
    x2 = dict(title = 'TAG')
    y = dict(title = 'COUNT')
    y3 = dict(title = 'TOKEN')
    lentgth = len(df['message'])

    for i in range(lentgth):
        tok = tokenize(df['message'][i])
        tokens = tokens + tok
    

    bar1 = Bar(x = genre_names,y = genre_counts)
    bar2 = Bar(x = list((df.iloc[:, 4:].sum().sort_values(ascending=False)).index),y = df.iloc[:, 4:].sum().sort_values(ascending=False))
    bar3 = Bar(x = list(((pd.Series([token for token in tokens if (len(token)>2) and (token not in ['have', 'with', 'you', "n't", 'are', 'not','the', 'and', 'for', 'this'])])).value_counts().sort_values(ascending=False)[:20]).index),
                    y = (pd.Series([token for token in tokens if (len(token)>2) and (token not in ['have', 'with', 'you', "n't", 'are', 'not','the', 'and', 'for', 'this'])])).value_counts().sort_values(ascending=False)[:20])
    
    vis1.append(bar1)
    vis2.append(bar2)
    vis3.append(bar3)
    all.append(dict(data=vis1, layout = dict(title = title1,xaxis = x1,yaxis = y)))
    all.append(dict(data=vis2, layout = dict(title = title2,xaxis = x2,yaxis = y)))
    all.append(dict(data=vis3, layout = dict(title = title3,xaxis = y3,yaxis = y)))

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(all)]
    graphJSON = json.dumps(all, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    The function will save user input in query and then use model to predict classification for query
    
    Returns:
    Render the go.html Please see that file
    """
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
