import pandas as pd
import numpy as np
import env
from sklearn.model_selection import train_test_split
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For Telco Data

def acquire_telco():
    '''
    Grab our data from path and read as csv
    '''
    
    df = pd.read_csv('Cust_Churn_Telco.csv')
    return(df)
    
def clean_telco(df):
    '''
    Takes in a df of telco_data and cleans the data appropriatly by dropping nulls,
    removing white space,
    creates dummy variables for Contract type,
    converts data to numerical, and bool data types, 
    and drops columsn that are not needed.
    
    
    return: df, a cleaned pandas data frame.
    '''
    
    # Instead of using dummies to seperate contracts use, 
    # df[['Contract']].value_counts()
    # Use a SQL querry
    
    df = df
    df.TotalCharges = df.TotalCharges.replace(r'^\s*$', np.nan, regex = True)
    df = df.fillna(0)
    dummy_df = pd.get_dummies(df[["Contract"]], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    df['Contract_Two year'] = df['Contract_Two year'].astype(bool)
    df = df.loc[df['Contract_Two year'], :]
    df = df.drop(columns = ['gender','SeniorCitizen',
                            'Partner','Dependents',
                            'PhoneService','MultipleLines',
                            'InternetService','OnlineSecurity',
                            'OnlineBackup','DeviceProtection',
                            'TechSupport','StreamingTV',
                            'StreamingMovies','PaperlessBilling',
                            'PaymentMethod','Contract_One year',
                            'Contract','Churn', 'Contract_Two year'])
    df = df.set_index("customerID")         
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

# Use train.index = train.customerID to make index the id and then you can drop the customerId column when running regressions.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For Zillow Data

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def acquire_zillow():
    '''
    Grab our data from path and read as csv
    '''
    
    df = pd.read_sql('SELECT * FROM properties_2017 join predictions_2017 using(parcelid) where transactiondate between "2017-05-01" and "2017-06-30" and unitcnt = 1', get_connection('zillow'))

    return(df)
    
def clean_zillow(df):
    '''
    Takes in a df of zillow_data and cleans the data appropriatly by dropping nulls,
    removing white space,
    creates dummy variables for Contract type,
    converts data to numerical, and bool data types, 
    and drops columsn that are not needed.
    
    return: df, a cleaned pandas data frame.
    '''
    
    # Instead of using dummies to seperate contracts use, 
    # df[['Contract']].value_counts()
    # Use a SQL querry
    
    df = df
    df = df.loc[:, df.isnull().mean() < .10]        
    return df
    
def split_zillow(df, stratify_by=""):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    return train, validate, test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def wrangle_zillow():
    '''
    wrangle_zillow will read in our zillow data as a pandas dataframe,
    clean the data,
    split the data,
    return: train, validate, test sets of pandas dataframes for tleco data
    '''
    df = clean_zillow(acquire_zillow())
    return split_zillow(df)