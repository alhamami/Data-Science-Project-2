import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def getCategs():
    """
    the function will create a list contains all the categories that will be worked on.
    
    Returns:
    allCategs (list): It will return a list containing all the categories
    """
    allCategs = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
                'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people',
                 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools',
                  'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire',
                   'earthquake', 'cold', 'other_weather', 'direct_report']
    return allCategs               

def load_data(database_filepath):
    """
    The function will load a data which is a disaster_response_df Table and read it
    
    Parameter:
    database_filepath (string): name of database where come from command line input
    
    Returns:
    data['message'] (Dataframe): Selecting column form disaster_response_df Table based on their name which is message
    data[all] (Dataframe): The disaster_response_df Table with all categories
    all (list): all categories
    """

    data = pd.read_sql_table('disaster_response_df', create_engine('sqlite:///{}'.format(database_filepath)))
    
    all = getCategs()
    return data['message'], data[all], all

def tokenizer(text):
    """
    The function will extract the tokens from string of characters
    
    Parameter:
    text (string): the text that we will tokenize
    
    Returns:
    word_tokenize(text): return a tokenized copy of text
    """
    return word_tokenize(text)

def tokenize(text):
    """
    Carries out a series of transforms on the text provided in order to break it down to a format suitable for vectorization.
    
    Parameter:
    text (string): A disaster relief message to be tokenized.
    
    Returns:
    data (list): A list of tokens representing a version of the text that is suitable for vectorization.
    """
    data = []
    word = WordNetLemmatizer()
    tokenz = tokenizer(text)
    for token in tokenz:
        data.append(word.lemmatize(token).lower().strip())
    return data

def getPip():
    """
    The function will make NLP pipeline and choosing parameters for GridSearch to prepare the model
    
    Returns:
    (sklearn.model_selection._search.GridSearchCV): is GridSearchCV object characterizing the NLP pipeline and it also defining the parameters of the grid search
    """
    pipeline  = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),('tfidf',
          TfidfTransformer()),('clf',MultiOutputClassifier(RandomForestClassifier()))])

    parameters  = {'clf__estimator__n_estimators': [10, 25],
                  'clf__estimator__min_samples_split': [2, 4]}


    return GridSearchCV(pipeline ,param_grid=parameters)


def build_model():
    """
    The function it will call getPip function in order to make NLP pipeline and choosing parameters for GridSearch to prepare the model
    
    Returns:
    (sklearn.model_selection._search.GridSearchCV): which is the result that come from getPip function after choosing parameters
    """
    pip = getPip()
    return pip
    

def evaluate_model(model, X_test, Y_test, category_names):
    """
    The function will show the results of the model
    
    Parameter:
    model: This is the model we worked on
    X_test: The test dataset that we will use it to predict
    Y_test: The target that we will use to compare the production result with it 
    """
    print("Evaluate Report:")
    print(classification_report(Y_test, model.predict(X_test).astype(int), target_names=category_names))

def save_model(model, model_filepath):
    """
    The function will save the trained NLP model
    
    Parameter:
    model: The trained model that is to be saved
    model_filepath (str): Filepath for the model with the name for the model - should end with .pkl.
    
    Returns:
    Dumps the model into a .pkl file in the location specified.
    """
    picOpen = open(model_filepath, "wb")
    pickle.dump(model, picOpen)


def main():
    """
    Stacks up all the previous functions in the right order, runs them and creates a model
    """
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