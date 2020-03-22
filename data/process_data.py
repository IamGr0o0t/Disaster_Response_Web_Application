import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Title: Function to load and merge data from filepaths.
    Input: messages_filepath(str) - link to the filepath of the messages file
           categories_filepath(str) - link to the filepath of the categories file
    Output: df(pandas DataFrame) - merged dataframe containing messages and categories
    '''
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    df = messages_df.merge(categories_df
                          ,on = 'id')
    return df

def clean_data(df):
    '''
    Title: Data wrangling function.
    Input: df(pandas DataFrame) - Dataframe containing messages and categories
    Output: df(pandas DataFrame) - Cleaned and preprocess dataframe.
    '''

    # 1. split categories on ';' and expand to make column for each category
    categories = df['categories'].str.split(';'
                                           ,expand = True)             
    first_row = categories.loc[0] # get first row of the categories_df
    category_colnames = [] # empty list for column names
    # loop through first_row and append column names with row value, but last two characters
    for col_name in first_row:
        category_colnames.append(col_name[: -2])
    print('Category Column Names:', category_colnames)
    categories.columns = category_colnames
    # get 1's and 0's for category columns
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1] # set each value to be the last character of the string  
        categories[column] = pd.to_numeric(categories[column]) # convert column from string to numeric
    # concatenate category columns to the dataframe
    df = pd.concat([df, categories]
                  , axis = 1)
    # drop old categories column and child_alone column because it has only 0's (see ETL_Pipeline_Preparation.ipymb for more info)
    df.drop(['categories', 'child_alone']
           , axis = 1
           , inplace = True)

    # 2. Deal with duplicate and incorrect data 
    df = df[df['related'] != 2] # Exclude all non-binary rows in 'related' column 
    df.drop_duplicates(inplace = True) # Drop duplicates
    return df

def save_data(df, database_filepath):
    '''
    Title: Save data to sqlite database.
    Input: df(pandas DataFrame) - Dataframe containing messages and categories
           database_filename(str) - Name of the databse to be created
    Output: None
    '''
    engine = create_engine('sqlite:///'+ str (database_filepath))
    df.to_sql('Message_Category'
             ,engine
             ,index = False
             ,if_exists = 'replace')
             
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