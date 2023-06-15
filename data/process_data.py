import sys
import pandas as pd
from sqlalchemy.engine import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    To load the disaster messages and categories datasets and merge them based on the id.
    
    Args:
    messages_filepath: messages dataset file path(saved as a .csv)
    categories_filepath: categories dataset file path (saved as a .csv)
    
    Returns:
    df: the merged dataframe
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages,categories,how='left',on='id')
    # Return dataframes
    return df

def clean_data(df):
    '''
    To clean the merged dataset
    1. The categories column in the merged dataset is semicolon seperated list of differnet categories with a 1 or 0 used to identify which of the categories the message falls into. The first step splits the categories column into individual columns and populates them with 1 or 0.
    2. Then replace the original categories column with the new split columns
    3. Finally remove duplicates and return the cleaned dataframe
    4. Fill NaN values in categories columns with 0
 
    Args: 
    df: Dataframe with messages and categories data
    
    Returns:
    df: Cleaned dataset
    '''
    # Split the values in the categories column on the ; character so that each         value becomes a separate column
    categories = df['categories'].str.split(';',expand=True)
    #Using the first row of categories dataframe to create column names for the categories data.
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x:x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    #Set each category value to 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype('float').astype('Int32')
        #Set anything greater than 1 to 1
        categories[column] = categories[column].apply(lambda x:1 if x>1 else x)
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df=df.drop_duplicates()
    return df



def save_data(df, database_filename):
    ''' 
    Save a dataframe to sqlite database at the given file_path
    
    Args:
    df: Dataframe to be saved
    database_filename: file path to where the Database is to be stored
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('message_categories', engine, index=False)  


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