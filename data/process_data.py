import sys
import pandas as PD
import sqlalchemy as DB

def load_data(messages_filepath, categories_filepath):
    '''
    Loading the 'Messages' and 'Categories' datasets and
    merging on common feature 'id'

    INPUT: 
        messages_filepath - path of CSV file
        categories_filepath - path of CSV file
    OUTPUT:
        df - Merged dataframe
    '''
    # Loading Message dataset
    messages = PD.read_csv(messages_filepath)

    # Loading Categories dataset
    categories = PD.read_csv(categories_filepath)

    # Merge both datasets
    df = PD.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    '''
    Flattening, cleaning and concatenating the 'categories' features with df
    INPUT:
        df - Merged dataframe
    OUTPUT:
        df - Cleaned dataframe
    '''

    # Split categories into separate category columns.
    categories = df['categories'].str.split(';', expand=True)
    col_values = categories.iloc[0].str.replace('-','')
    col_values = col_values.str.replace('0','')
    col_values = col_values.str.replace('1','')
    categories.columns = list(col_values)

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Droppping the original categories column from `df`
    df = df.drop('categories', axis=1)
 
    if df.shape[0] == categories.shape[0]:
        print("Both dataframes can be concatenated.")
    else:
        print("Fix the shapes of the dataframe")
    # Concatenating the original dataframe with the new `categories` dataframe
    df = PD.concat([df, categories], axis=1)

    # As the 'message' column contains the translated messages,
    # it is safe to remove the 'original' column from dataset 
    # considering its count of null values
    df = df.drop('original', axis=1)

    return df


def save_data(df, database_filename):
    '''
    Save the clean dataset into an sqlite database

    INPUT:
        df - Cleaned dataframe
        database_filename - Name of database
    OUTPUT:
        SQL Database
    '''

    engine = DB.create_engine('sqlite:///' + database_filename)
    df.to_sql('myTable', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()