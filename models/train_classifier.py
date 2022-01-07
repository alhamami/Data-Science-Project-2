import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def getCategs():
    allCategs = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people',
                 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools',
                  'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire',
                   'earthquake', 'cold', 'other_weather', 'direct_report']
    return allCategs               

def load_data(database_filepath):

    data = pd.read_sql_table('disaster_response_df', create_engine('sqlite:///{}'.format(database_filepath)))
    
    all = getCategs()
    return data['message'], data[all], all

def tokenizer(text):
    return word_tokenize(text)

def tokenize(text):
    data = []
    word = WordNetLemmatizer()
    tokenz = tokenizer(text)
    for token in tokenz:
        data.append(word.lemmatize(token).lower().strip())
    return data

def getPip():
    return Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),('tfidf',
     TfidfTransformer()),('clf', RandomForestClassifier())])


def build_model():
    pip = getPip()
    return pip
    

def evaluate_model(model, X_test, Y_test, category_names):
    print("Evaluate Report:")
    print(classification_report(Y_test, model.predict(X_test).astype(int), target_names=category_names))

def save_model(model, model_filepath):
    picOpen = open(model_filepath, "wb")
    pickle.dump(model, picOpen)


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
        evaluate_model(model, X_test, Y_test, category_names)

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
