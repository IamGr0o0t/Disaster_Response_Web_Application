import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import pickle
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download(['punkt','wordnet','stopwords'])
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

def load_data(database_filepath):
    '''
    Title: Function to load data from sqlite database and define X - features and y response dimensions 
    Input: database_filepath(str) - link to the filepath of the sqlite database
    Output: X(pandas DataFrame) - Features matrix
            Y(pandas DataFrame) - Multi Output labels
            category_names(list) - List of category names 
    '''
    # load data from database
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql('select * from Message_Category'
                    ,engine)
    # create feature matrix
    X = df['message']
    # response matrix
    Y = df.iloc[:, 4:]
    # create list of category names
    category_names = Y.columns
    return X, Y, category_names 

def tokenize(text):
    '''
    Title: Text pre-processing function.
    Input: Raw text data
    Output: lemmed - Normalized, stop words removed, tokenized, stemmed and lemmatized text data.
    '''
    # Normalize text.
    text = re.sub(r'[^a-zA-Z0-9]', ' '
                 ,text.lower())
    # Instantiate stop words             
    stop_words = stopwords.words('english')
    # Tokenize words
    words = word_tokenize(text)
    # Stemming
    stemmed = [PorterStemmer().stem(w) for w in words]
    # Lemmatize
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]
    return lemmed

def build_model():
    '''
    Title: Grid Search model with pipeline.
    Input: None
    Output: grid - grid search model with pipeline and classifier
    '''
    # build a pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize))
                        ,('tfidf', TfidfTransformer())
                        ,('clf', MultiOutputClassifier(RandomForestClassifier()))])
    # create parameters for grid
    param_grid = {'clf__estimator__n_estimators' : [int(x) for x in np.linspace(start = 100, stop = 300, num = 2)] # number of trees in RF
                 ,'clf__estimator__max_depth': [int(x) for x in np.linspace(1, 100, num = 2)] # maximum number of levels in tree
                 ,'clf__estimator__min_samples_leaf': [1, 3, 5]} # minimum samples at each leaf node

    # create grid
    grid = GridSearchCV(pipeline
                       ,param_grid
                       ,cv = 3
                       ,return_train_score = False)
    return grid

def evaluate_model(model, X_test, Y_test):
    '''
    Title: Model evaluation results.
    Input: model - estimator-object
           X_test, y_test - test set
           category_names - list of category strings
    Output: None
    '''
    # predict
    Y_pred = model.predict(X_test)

    # create empty dataframe with columns of interest
    results = pd.DataFrame(columns = ['Category', 'f_score', 'precision', 'recall'])
    num = 0

    # looping through Y_test columns
    for cat in Y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(Y_test[cat], Y_pred[:, num], average = 'weighted')
        results.at[num + 1, 'Category'] = cat
        results.at[num + 1, 'f_score'] = f_score
        results.at[num + 1, 'precision'] = precision
        results.at[num + 1, 'recall'] = recall
        num += 1
    # get results results    
    print('Results:', results)
    print('Average f_score:', results['f_score'].mean())
    print('Average precision:', results['precision'].mean())
    print('Average recall:', results['recall'].mean())

def save_model(model, model_filepath):
    '''
    Title: Save model to a pickle.
    Input: model - estimator-object
           model_filepath(str) - model pickle
    Output: None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    '''
    Title: Check the results.
    Input: None
    Output: None
    '''
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
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'models/train_classifier.py data/DisasterResponse.db models/classifier.pkl')

if __name__ == '__main__':
    main()