#!/usr/bin/python3
import utils_old
import seaborn as sb
import matplotlib.pyplot as plt

print('Preparing dataset...')
df = utils_old.prepare_data(produce_full_df=True)

# Drop non-numerical columns and label column
df.drop(axis=1, labels=['term', 'emp_length', 'home_ownership', 'verification_status', 'loan_status', 'purpose', 'zip_code', 'addr_state', 'application_type'])

fig, ax = plt.subplots(figsize=(12,12))
sb.heatmap(df.corr()).get_figure().savefig('heatmap.svg')
