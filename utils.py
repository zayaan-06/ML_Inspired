#!/usr/bin/python3
import pandas as pd
import numpy as np

csv_path = './data_clean.csv'

def prepare_data(load_existing=False):
    if load_existing:
        import numpy as np
        dataset = np.load('dataset.npz')
        return (dataset['arr_0'], dataset['arr_1'], dataset['arr_2'], dataset['arr_3'])

    usecols = ['loan_amnt', 'funded_amnt',
           'term', 'int_rate', 'installment',
           'emp_length', 'home_ownership', 'annual_inc',
           'verification_status', 'loan_status',
           'purpose', 'zip_code', 'addr_state', 'dti',
           'delinq_2yrs', 'earliest_cr_line', 'fico_range_low',
           'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq',
           'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',
           'revol_util', 'total_acc', 'out_prncp',
           'out_prncp_inv', 'last_fico_range_high', 'last_fico_range_low',
           'collections_12_mths_ex_med', 'mths_since_last_major_derog',
           'policy_code', 'application_type', 'annual_inc_joint', 'dti_joint',
           'acc_now_delinq', 'tot_coll_amt',
           'tot_cur_bal', 'open_acc_6m', 'open_act_il', 'open_il_12m',
           'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',
           'open_rv_12m', 'open_rv_24m', 'all_util',
           'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
           'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util',
           'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct',
           'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
           'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
           'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
           'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
           'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
           'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
           'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
           'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75',
           'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim',
           'total_bal_ex_mort', 'total_bc_limit',
           'total_il_high_credit_limit']

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

    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtype, low_memory=False)

    # Parse percentage features.
    def parse_percentage(percentage):
      if str(percentage) == 'nan': return float('nan')
      return float(percentage.replace('%', '')) / 100.0

    df['revol_util'] = df['revol_util'].apply(parse_percentage)
    df['int_rate'] = df['int_rate'].apply(parse_percentage)

    nan_max_cols = [
        'mths_since_last_delinq',
        'mths_since_last_record',
        'mths_since_last_major_derog',
        'mths_since_rcnt_il',
        'mths_since_recent_bc',
        'mths_since_recent_bc_dlq',
        'mths_since_recent_inq',
        'mths_since_recent_revol_delinq',
        'mo_sin_old_il_acct',
        'mo_sin_old_rev_tl_op',
        'mo_sin_rcnt_rev_tl_op',
        'mo_sin_rcnt_tl',
    ]

    nan_zero_cols = [
        'tot_coll_amt',
        'tot_cur_bal',
        'open_acc_6m',
        'open_act_il',
        'open_il_12m',
        'open_il_24m',
        'total_bal_il',
        'open_rv_12m',
        'open_rv_24m',
        'inq_fi',
        'inq_last_12m',
        'num_accts_ever_120_pd',
        'num_actv_bc_tl',
        'num_actv_rev_tl',
        'num_bc_sats',
        'num_bc_tl',
        'num_il_tl',
        'num_op_rev_tl',
        'num_rev_accts',
        'num_rev_tl_bal_gt_0',
        'num_sats',
        'num_tl_120dpd_2m',
        'num_tl_30dpd',
        'num_tl_90g_dpd_24m',
        'num_tl_op_past_12m',
        'mort_acc',
        'percent_bc_gt_75',
        'tot_hi_cred_lim',
        'total_bal_ex_mort',
        'total_bc_limit',
        'total_il_high_credit_limit',
        'bc_open_to_buy',
        'acc_open_past_24mths',
        'total_cu_tl',
        'total_rev_hi_lim',
    ]

    nan_mean_cols = [
        'dti',
        'annual_inc_joint',
        'dti_joint',
        'il_util',
        'all_util',
        'bc_util',
        'pct_tl_nvr_dlq',
        'avg_cur_bal',
        'revol_util',
    ]

    import numpy as np
    for col in df:
      dt = df[col].dtype
      if col in nan_mean_cols:
        df[col] = df[col].fillna(np.nanmean(df[col].values))
      if col in nan_zero_cols:
        df[col] = df[col].fillna(0.0)
      if col in nan_max_cols:
        df[col] = df[col].fillna(np.nanmax(df[col].values) * 5)

    df['earliest_cr_line'] = pd.to_timedelta(pd.to_datetime(df['earliest_cr_line'])).dt.days
    df['loan_status'] = df['loan_status'].map({'Charged Off': 0, 'Default': 0, 'Fully Paid': 1})

    df = pd.get_dummies(df)


    '''
    Split dataset into 'train' and 'test'.
    '''
    import numpy as np
    train, test = np.split(df.sample(frac=1), [int(.7*len(df))])

    '''
    Split train/test into features/label numpy arrays.
    '''
    LABEL_COL = 'loan_status'
    train_x = train.loc[:, train.columns != LABEL_COL].values
    train_y = train.loc[:, train.columns == LABEL_COL].values.squeeze()
    test_x = test.loc[:, test.columns != LABEL_COL].values
    test_y = test.loc[:, test.columns == LABEL_COL].values.squeeze()

    # Standardize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    np.savez_compressed('dataset.npz', train_x, train_y, test_x, test_y)
    return (train_x, train_y, test_x, test_y)

def print_report(test_y, test_predict):
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(test_y, test_predict))

    from sklearn.metrics import classification_report
    print(classification_report(test_y, test_predict))

def prepare_regression_data(load_existing=False):
    if load_existing:
        import numpy as np
        dataset = np.load('regression_annual_dataset.npz')
        return (dataset['arr_0'], dataset['arr_1'], dataset['arr_2'], dataset['arr_3'])

    usecols = ['loan_amnt', 'funded_amnt', 'total_pymnt',
           'term', 'int_rate', 'installment', 'issue_d', 'last_pymnt_d',
           'emp_length', 'home_ownership', 'annual_inc',
           'verification_status',
           'purpose', 'zip_code', 'addr_state', 'dti',
           'delinq_2yrs', 'earliest_cr_line', 'fico_range_low',
           'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq',
           'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',
           'revol_util', 'total_acc', 'out_prncp',
           'out_prncp_inv', 'last_fico_range_high', 'last_fico_range_low',
           'collections_12_mths_ex_med', 'mths_since_last_major_derog',
           'policy_code', 'application_type', 'annual_inc_joint', 'dti_joint',
           'acc_now_delinq', 'tot_coll_amt',
           'tot_cur_bal', 'open_acc_6m', 'open_act_il', 'open_il_12m',
           'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',
           'open_rv_12m', 'open_rv_24m', 'all_util',
           'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
           'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util',
           'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct',
           'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
           'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
           'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
           'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
           'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
           'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
           'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
           'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75',
           'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim',
           'total_bal_ex_mort', 'total_bc_limit',
           'total_il_high_credit_limit']

    dtype = {
      'loan_amnt': 'int64',
      'term': 'category',
      'int_rate': 'object',
      'emp_length': 'category',
      'home_ownership': 'category',
      'annual_inc': 'float64',
      'verification_status': 'category',
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

    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtype, low_memory=False)

    # Parse percentage features.
    def parse_percentage(percentage):
      if str(percentage) == 'nan': return float('nan')
      return float(percentage.replace('%', '')) / 100.0

    df['revol_util'] = df['revol_util'].apply(parse_percentage)
    df['int_rate'] = df['int_rate'].apply(parse_percentage)

    nan_max_cols = [
        'mths_since_last_delinq',
        'mths_since_last_record',
        'mths_since_last_major_derog',
        'mths_since_rcnt_il',
        'mths_since_recent_bc',
        'mths_since_recent_bc_dlq',
        'mths_since_recent_inq',
        'mths_since_recent_revol_delinq',
        'mo_sin_old_il_acct',
        'mo_sin_old_rev_tl_op',
        'mo_sin_rcnt_rev_tl_op',
        'mo_sin_rcnt_tl',
    ]

    nan_zero_cols = [
        'tot_coll_amt',
        'tot_cur_bal',
        'open_acc_6m',
        'open_act_il',
        'open_il_12m',
        'open_il_24m',
        'total_bal_il',
        'open_rv_12m',
        'open_rv_24m',
        'inq_fi',
        'inq_last_12m',
        'num_accts_ever_120_pd',
        'num_actv_bc_tl',
        'num_actv_rev_tl',
        'num_bc_sats',
        'num_bc_tl',
        'num_il_tl',
        'num_op_rev_tl',
        'num_rev_accts',
        'num_rev_tl_bal_gt_0',
        'num_sats',
        'num_tl_120dpd_2m',
        'num_tl_30dpd',
        'num_tl_90g_dpd_24m',
        'num_tl_op_past_12m',
        'mort_acc',
        'percent_bc_gt_75',
        'tot_hi_cred_lim',
        'total_bal_ex_mort',
        'total_bc_limit',
        'total_il_high_credit_limit',
        'bc_open_to_buy',
        'acc_open_past_24mths',
        'total_cu_tl',
        'total_rev_hi_lim',
    ]

    nan_mean_cols = [
        'dti',
        'annual_inc_joint',
        'dti_joint',
        'il_util',
        'all_util',
        'bc_util',
        'pct_tl_nvr_dlq',
        'avg_cur_bal',
        'revol_util',
    ]

    import numpy as np
    for col in df:
      dt = df[col].dtype
      if col in nan_mean_cols:
        df[col] = df[col].fillna(np.nanmean(df[col].values))
      if col in nan_zero_cols:
        df[col] = df[col].fillna(0.0)
      if col in nan_max_cols:
        df[col] = df[col].fillna(np.nanmax(df[col].values) * 5)

    df['earliest_cr_line'] = pd.to_timedelta(pd.to_datetime(df['earliest_cr_line'])).dt.days

    df = df.dropna(subset=['last_pymnt_d'])
    df['label'] = (df['total_pymnt'] / df['funded_amnt']) ** (365.0 / ((pd.to_datetime(df['last_pymnt_d']) - pd.to_datetime(df['issue_d'])).dt.days + 30))

    df = df.drop(['total_pymnt', 'issue_d', 'last_pymnt_d'], axis=1)

    df = pd.get_dummies(df)


    '''
    Split dataset into 'train' and 'test'.
    '''
    import numpy as np
    train, test = np.split(df.sample(frac=1), [int(.7*len(df))])

    '''
    Split train/test into features/label numpy arrays.
    '''
    LABEL_COL = 'label'
    train_x = train.loc[:, train.columns != LABEL_COL].values
    train_y = train.loc[:, train.columns == LABEL_COL].values.squeeze()
    test_x = test.loc[:, test.columns != LABEL_COL].values
    test_y = test.loc[:, test.columns == LABEL_COL].values.squeeze()

    # Standardize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    np.savez_compressed('regression_annual_dataset.npz', train_x, train_y, test_x, test_y)
    return (train_x, train_y, test_x, test_y)
