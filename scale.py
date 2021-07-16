import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from wrangle import split_telco
import pandas as pd
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def minmax_scale(train, validate, test):
    
    # Make the thing
    scaler = sklearn.preprocessing.MinMaxScaler()

    # We fit on the training data
    # in a way, we treat our scalers like our ML models
    # we only .fit on the training data
    scaler.fit(train)
    
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)
    
    # turn the numpy arrays into dataframes
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)
    
    fig = plt.figure(figsize=(12, 6))
    gs = plt.GridSpec(2,2)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    ax1.title.set_text('Tenure')
    ax2.title.set_text('MonthlyCharges')
    ax3.title.set_text('TotalCharges')
    
    ax1.hist(train_scaled.tenure)
    
    ax2.hist(train_scaled.MonthlyCharges)

    ax3.hist(train_scaled.TotalCharges)
    
    return train_scaled, validate_scaled, test_scaled

def standard_scale(train, validate, test):
    
    # Make the thing
    scaler = sklearn.preprocessing.StandardScaler()

    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(train)

    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)

    # turn the numpy arrays into dataframes
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)
    
    fig = plt.figure(figsize=(12, 6))
    gs = plt.GridSpec(2,2)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    ax1.title.set_text('Tenure')
    ax2.title.set_text('MonthlyCharges')
    ax3.title.set_text('TotalCharges')
    
    ax1.hist(train_scaled.tenure)
    
    ax2.hist(train_scaled.MonthlyCharges)

    ax3.hist(train_scaled.TotalCharges)
    
    return train_scaled, validate_scaled, test_scaled


def robust_scale(train, validate, test):
    
    # Make the thing
    scaler = sklearn.preprocessing.RobustScaler()

    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(train)

    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)

    # turn the numpy arrays into dataframes
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)
    
    fig = plt.figure(figsize=(12, 6))
    gs = plt.GridSpec(2,2)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    ax1.title.set_text('Tenure')
    ax2.title.set_text('MonthlyCharges')
    ax3.title.set_text('TotalCharges')
    
    ax1.hist(train_scaled.tenure)
    
    ax2.hist(train_scaled.MonthlyCharges)

    ax3.hist(train_scaled.TotalCharges)
    
    return train_scaled, validate_scaled, test_scaled

def quantile_transformer(train, validate, test):
    
    # Make the thing
    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution='normal')

    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(train)

    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)

    # turn the numpy arrays into dataframes
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)
    
    fig = plt.figure(figsize=(12, 6))
    gs = plt.GridSpec(2,2)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    ax1.title.set_text('Tenure')
    ax2.title.set_text('MonthlyCharges')
    ax3.title.set_text('TotalCharges')
    
    ax1.hist(train_scaled.tenure)
    
    ax2.hist(train_scaled.MonthlyCharges)

    ax3.hist(train_scaled.TotalCharges)
    
    return train_scaled, validate_scaled, test_scaled

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fucntions that show set scaler, original, and scaled
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_scaled_date(scaler, scaler_name, feature):
    train_scaled = scaler.fit_transform(train[[feature]])
    fig = plt.figure(figsize = (12,6))

    gs = plt.GridSpec(2,2)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    ax1.scatter(train[[feature]], train_scaled)
    ax1.set(xlabel = feature, ylabel = 'Scaled_' + feature, title = scaler_name)

    ax2.hist(train[[feature]])
    ax2.set(title = 'Original')

    ax3.hist(scaled)
    ax3.set(title = 'Scaled')
    plt.tight_layout();

def scale_fit_transform(X_train, X_validate, X_test):
    # Define the thing
    scaler = sklearn.preprocessing.MinMaxScaler()

    # Fit the thing
    scaler.fit(X_train[['lotsizesquarefeet']])
    scaler.fit(X_train[['calculatedfinishedsquarefeet']])
    scaler.fit(X_train[['bedroomcnt']])
    scaler.fit(X_train[['bathroomcnt']])
    
    scaler.fit(X_validate[['lotsizesquarefeet']])
    scaler.fit(X_validate[['calculatedfinishedsquarefeet']])
    scaler.fit(X_validate[['bedroomcnt']])
    scaler.fit(X_validate[['bathroomcnt']])
    
    scaler.fit(X_test[['lotsizesquarefeet']])
    scaler.fit(X_test[['calculatedfinishedsquarefeet']])
    scaler.fit(X_test[['bedroomcnt']])
    scaler.fit(X_test[['bathroomcnt']])

    #transform
    scaled1 = scaler.transform(X_train[['lotsizesquarefeet']])
    scaled2 = scaler.transform(X_train[['calculatedfinishedsquarefeet']])
    scaled3 = scaler.transform(X_train[['bedroomcnt']])
    scaled4 = scaler.transform(X_train[['bathroomcnt']])

    scaled11 = scaler.transform(X_validate[['lotsizesquarefeet']])
    scaled22 = scaler.transform(X_validate[['calculatedfinishedsquarefeet']])
    scaled33 = scaler.transform(X_validate[['bedroomcnt']])
    scaled44 = scaler.transform(X_validate[['bathroomcnt']])
    
    scaled111 = scaler.transform(X_test[['lotsizesquarefeet']])
    scaled222 = scaler.transform(X_test[['calculatedfinishedsquarefeet']])
    scaled333 = scaler.transform(X_test[['bedroomcnt']])
    scaled444 = scaler.transform(X_test[['bathroomcnt']])
    
    # single step to fit and transform
    scaled1 = scaler.fit_transform(X_train[['lotsizesquarefeet']])
    scaled2 = scaler.fit_transform(X_train[['calculatedfinishedsquarefeet']])
    scaled3 = scaler.fit_transform(X_train[['bedroomcnt']])
    scaled4 = scaler.fit_transform(X_train[['bathroomcnt']])
    
    scaled11 = scaler.fit_transform(X_validate[['lotsizesquarefeet']])
    scaled22 = scaler.fit_transform(X_validate[['calculatedfinishedsquarefeet']])
    scaled33 = scaler.fit_transform(X_validate[['bedroomcnt']])
    scaled44 = scaler.fit_transform(X_validate[['bathroomcnt']])
    
    scaled111 = scaler.fit_transform(X_test[['lotsizesquarefeet']])
    scaled222 = scaler.fit_transform(X_test[['calculatedfinishedsquarefeet']])
    scaled333 = scaler.fit_transform(X_test[['bedroomcnt']])
    scaled444 = scaler.fit_transform(X_test[['bathroomcnt']])

    #you can make a new 'scaled' column in original dataframe if you wish
    X_train['lotsizesquarefeet_scaled']  = scaled1
    X_train['calculatedfinishedsquarefeet_scaled'] = scaled2
    X_train['bedroomcnt_scaled'] = scaled3
    X_train['bathroomcnt_scaled'] = scaled4
    
    X_validate['lotsizesquarefeet_scaled']  = scaled11
    X_validate['calculatedfinishedsquarefeet_scaled'] = scaled22
    X_validate['bedroomcnt_scaled'] = scaled33
    X_validate['bathroomcnt_scaled'] = scaled44
    
    X_test['lotsizesquarefeet_scaled']  = scaled111
    X_test['calculatedfinishedsquarefeet_scaled'] = scaled222
    X_test['bedroomcnt_scaled'] = scaled333
    X_test['bathroomcnt_scaled'] = scaled444

    return scaled1, scaled2, scaled3, scaled4 