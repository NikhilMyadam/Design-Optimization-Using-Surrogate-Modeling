########################################################################################################################################################
# Created by Nikhil Myadam
########################################################################################################################################################

# File to create a single csv file containing all the information on models and objective functions
# concatenate individual models_database files i.e. csv files using pd.concat()

import pandas as pd

df = pd.concat(map(pd.read_csv, ['MODELS_DATABASE_95degrees.csv', 'MODELS_DATABASE_96degrees.csv', 'MODELS_DATABASE_97degrees.csv', 'MODELS_DATABASE_98degrees.csv', 'MODELS_DATABASE_99degrees.csv', 'MODELS_DATABASE_100degrees.csv']), ignore_index=True)

# Drop the index column if it exists
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Drop unnecessary columns
columns_to_drop = ['Impact_Model', 'Mesh_Part', 'Mesh_Impactor', 'Mass_Impactor', 'Dynamic_Job', 'Time_req_s']
df.drop(columns=columns_to_drop, inplace=True)

# Print column labels
print(df.keys())

# Reset index after dropping unnecessary columns
df.reset_index(drop=True, inplace=True)

# Create a new csv file i.e. concatenated csv file with dropped columns

df.to_excel("XGB_MODELS_DATABASE.xlsx")
df.to_csv("XGB_MODELS_DATABASE.csv")