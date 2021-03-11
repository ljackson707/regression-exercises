import pandas as pd
from sklearn.model_selection import train_test_split
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def acquire_telco():
    '''
    Grab our data from path and read as csv
    '''
    
    df = pd.read_csv('Cust_Churn_Telco.csv')
    return(df)
    
def clean_telco(df):
    '''
    Takes in a df of telco_data and cleans the data appropriatly by dropping nulls,
    creates dummy variables for Contract type,
    converts data to numerical, and bool data types, 
    and drops columsn that are not needed.
    
    
    return: df, a cleaned pandas data frame.
    '''
    
    df = df
    dummy_df = pd.get_dummies(df[["Contract"]], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    df['Contract_Two year'] = df['Contract_Two year'].astype(bool)
    df = df.loc[df['Contract_Two year'], :]
    df = df.dropna()
    df['SeniorCitizen'] = df['SeniorCitizen'].astype('O')
    df = df.drop(columns = ['gender','SeniorCitizen',
                            'Partner','Dependents',
                            'PhoneService','MultipleLines',
                            'InternetService','OnlineSecurity',
                            'OnlineBackup','DeviceProtection',
                            'TechSupport','StreamingTV',
                            'StreamingMovies','PaperlessBilling',
                            'PaymentMethod','Contract_One year',
                            'Contract','Churn'])
             
    return df
    
def split_telco(df):
    '''
    Takes in a cleaned df of telco data and splits the data appropriatly into train, validate, and test.
    '''
    
    train_val, test = train_test_split(df, train_size =  0.8, random_state = 123)
    train, validate = train_test_split(df, train_size =  0.8, random_state = 123)
    return train, validate, test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def wrangle_telco():
    '''
    wrangle_grades will read in our telco data as a pandas dataframe,
    clean the data,
    split the data,
    return: train, validate, test sets of pandas dataframes for tleco data, strat on total charges
    '''
    df = clean_telco(acquire_telco())
    return split_telco(df)