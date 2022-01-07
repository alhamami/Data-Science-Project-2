import sys
import pandas as pd
from sqlalchemy import create_engine


def split(categ):
    cates = (categ['categories'].str.split(pat=';', expand=True)).iloc[0, :]
    return cates.map(lambda x: x.split('-')[0])
    
def concat2F(messages_filepath, file1):
    file2 = pd.read_csv(messages_filepath)
    return pd.concat([file2, file1], axis=1)


def load_data(messages_filepath, categories_filepath):
    
    file1 = pd.read_csv(categories_filepath)
    file1.columns = split(file1)

    for x in file1:
        file1[x] = file1[x].map(lambda x: x[-1])
        file1[x] = file1[x].astype(int)

    data = concat2F(messages_filepath, file1.replace(2, 1, inplace=True))
    return data
    

def clean_data(df):     
    return df.drop_duplicates()
          
          
def save_data(df, database_filename):
    df.to_sql('disaster_response_df',
     create_engine('sqlite:///{}'.format(database_filename)), index=False)


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
