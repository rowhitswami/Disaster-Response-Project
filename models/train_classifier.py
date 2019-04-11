# import libraries
import sys
import pandas as pd
import sqlalchemy as DB
import nltk
import pickle
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_data(database_filepath):
    # load data from database
    engine = DB.create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('select * from myTable', con=engine)
    X = df['message'].values
    Y = df.drop(['id', 'message', 'genre'], axis=1)
    return X, Y


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    # Creating pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),

        # Experimented with different algorithms and
        # AdaBoostClassifier gave the best result
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)

    # Overall Accuracy
    overall_accuracy = (Y_pred == Y_test).mean().mean()
    print('OVERALL ACCURACY OF MODEL: ', overall_accuracy)

    Y_pred_df = pd.DataFrame(Y_pred, columns = Y_test.columns)
    for column in Y_test.columns:
        print('------------------------------------------------------\n')
        print('Feature: {}\n'.format(column))
        print('Accuracy: ', accuracy_score(Y_test[column], Y_pred_df[column]))
        print(classification_report(Y_test[column],Y_pred_df[column]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
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
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()