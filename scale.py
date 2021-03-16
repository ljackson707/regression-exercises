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

