import pandas as pd

csv_path = 'data_clean.csv'

usecols = (
  'loan_amnt',
  'term',
  'int_rate',
  'emp_length',
  'home_ownership',
  'annual_inc',
  'verification_status',
  'loan_status',
  'purpose',
  'zip_code',
  'addr_state',
  'dti',
  'delinq_2yrs',
  'fico_range_low',
  'fico_range_high',
  'inq_last_6mths',
  'open_acc',
  'pub_rec',
  'revol_bal',
  'revol_util',
  'total_acc',
  'application_type'
)

dtype = {
  'loan_amnt': 'int64',
  'term': 'category',
  'int_rate': 'object',
  'emp_length': 'category',
  'home_ownership': 'category',
  'annual_inc': 'float64',
  'verification_status': 'category',
  'loan_status': 'category',
  'purpose': 'category',
  'zip_code': 'category',
  'addr_state': 'category',
  'dti': 'float64',
  'delinq_2yrs': 'int64',
  'fico_range_low': 'int64',
  'fico_range_high': 'int64',
  'inq_last_6mths': 'int64',
  'open_acc': 'int64',
  'pub_rec': 'int64',
  'revol_bal': 'int64',
  'revol_util': 'object',
  'total_acc': 'int64',
  'application_type': 'category'
}

def prepare_data(produce_full_df=False, train_split=0.7):
    '''
    Read dataset.
    '''
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtype, low_memory=False)

    '''
    Transform dataset.
    '''
    # Drop rows with nan values in any columns.
    df = df.dropna()

    # Parse percentage features.
    def parse_percentage(percentage):
        return float(percentage.replace('%', '')) / 100.0

    df['revol_util'] = df['revol_util'].apply(parse_percentage)
    df['int_rate'] = df['int_rate'].apply(parse_percentage)

    # Change loan_status value 'Default' to 'Charged Off'.
    df['loan_status'] = df['loan_status'].map({'Charged Off': 0, 'Default': 0, 'Fully Paid': 1})

    if produce_full_df:
        return df

    df = pd.get_dummies(df)


    '''
    Split dataset into 'train' and 'test'.
    '''
    import numpy as np
    train, test = np.split(df.sample(frac=1), [int(train_split*len(df))])

    '''
    Split train/test into features/label numpy arrays.
    '''
    LABEL_COL = 'loan_status'
    train_x = train.loc[:, train.columns != LABEL_COL].values
    train_y = train.loc[:, train.columns == LABEL_COL].values.squeeze()
    test_x = test.loc[:, test.columns != LABEL_COL].values
    test_y = test.loc[:, test.columns == LABEL_COL].values.squeeze()

    return (train_x, train_y, test_x, test_y)

def print_report(test_y, test_predict):
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(test_y, test_predict))

    from sklearn.metrics import classification_report
    print(classification_report(test_y, test_predict))
