import pandas as pd
import numpy as np
import env
import sklearn.preprocessing
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
    b
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
    df = df.fillna(0)
    df = df.dropna()
    df = df.drop(columns = ['transactiondate', 'propertyzoningdesc', 'propertycountylandusecode', 'id', 'censustractandblock', 'logerror','assessmentyear','taxvaluedollarcnt', 'structuretaxvaluedollarcnt','regionidcounty','regionidcity', 'rawcensustractandblock', 'longitude', 'latitude', 'heatingorsystemtypeid','regionidzip', 'finishedsquarefeet12', 'id','parcelid', 'roomcnt', 'unitcnt'])
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def wrangle_student_math(path):
    df = pd.read_csv(path, sep=";")
    
    # drop any nulls
    df = df[~df.isnull()]

    # get object column names
    object_cols = get_object_cols(df)
    
    # create dummy vars
    df = create_dummies(df, object_cols)
      
    # split data 
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(df, 'G3')
    
    # get numeric column names
    numeric_cols = get_numeric_X_cols(X_train, object_cols)

    # scale data 
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(X_train, X_validate, X_test, numeric_cols)
    
    return df, X_train, X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test


def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")


    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols
    
    
def create_dummies(df, object_cols):
    '''
    This function takes in a dataframe and list of object column names,
    and creates dummy variables of each of those columns. 
    It then appends the dummy variables to the original dataframe. 
    It returns the original df with the appended dummy variables. 
    '''
    
    # run pd.get_dummies() to create dummy vars for the object columns. 
    # we will drop the column representing the first unique value of each variable
    # we will opt to not create na columns for each variable with missing values 
    # (all missing values have been removed.)
    dummy_df = pd.get_dummies(object_cols, dummy_na=False, drop_first=True)
    
    # concatenate the dataframe with dummies to our original dataframe
    # via column (axis=1)
    df = pd.concat([df, dummy_df], axis=1)

    return df


def train_validate_test_split(df, target):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

        
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test


def get_numeric_X_cols(X_train, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]
    
    return numeric_cols


def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    '''
    this function takes in 3 dataframes with the same columns, 
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler. 
    it returns 3 dataframes with the same column names and scaled values. 
    '''
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).


    scaler = sklearn.preprocessing.MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    #scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train. 
    # 
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, 
                                  columns=numeric_cols).\
                                  set_index([X_train.index.values])

    X_validate_scaled = pd.DataFrame(X_validate_scaled_array, 
                                     columns=numeric_cols).\
                                     set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, 
                                 columns=numeric_cols).\
                                 set_index([X_test.index.values])

    
    return X_train_scaled, X_validate_scaled, X_test_scaled

