import sys
import pandas as pd
from sqlalchemy import create_engine



def concat2F(file2, file1):
    """
    The function will concat DataFrames
    
    Parameters:
    file1 (DataFrame): the categories DataFrame that will be splitting
    
    Returns:
    df (dataframe): the categories DataFrame after split it
    """
    return pd.concat([file2, file1], axis=1)


def load_data(messages_filepath, categories_filepath):
    """
    The function will loads data from two files and then it will merges the two files in one dataframe after convert categories to 0, 1
    
    Parameters:
    messages_filepath (str): filepath to the csv containing disaster relief messages
    categories_filepath (str):  filepath to the csv containing the multi-classifications of the above messages
    
    Returns:
    df (dataframe): The dataset that contains the messages and the classifications
    """

    file1 = pd.read_csv(categories_filepath)
	
	
	
    file1 = file1['categories'].str.split(pat=';', expand=True)
    gets = file1.iloc[0, :]
    col = gets.map(lambda x: x.split('-')[0])
    file1.columns = col



    for x in file1:
        file1[x] = file1[x].map(lambda x: x[-1])
        file1[x] = file1[x].astype(int)

    file1.replace(2, 1, inplace=True)
    file2 = pd.read_csv(messages_filepath)
    data = concat2F(file2, file1)
    return data
    

def clean_data(df): 
    """
    The function will remove the duplicates in the dataframe
    
    Parameters:
    df (dataframe): The dataset that contains the messages and the classifications
    
    Returns:
    df (dataframe): The cleaned dataset
    """    
    return df.drop_duplicates()
          
          
def save_data(df, database_filename):
    """
    The function will saves the data in SQLite database
    
    Parameters:
    df (dataframe): A cleaned dataset
    database_filename (str): The name of the database file to be created - should end in .db.  
    
    Returns:
    The database file with the above specified name
    """
    df.to_sql('disaster_response_df',
     create_engine('sqlite:///{}'.format(database_filename)), index=False)


def main():
    """
    The function will grabs the required variables from the command line,
    stacks the above functions in the right order and generates a clean .db file.
    """
    if len(sys.argv) == 5:

        messages_filepath, categories_filepath, database_filepath = sys.argv[2:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print(len(sys.argv))
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()