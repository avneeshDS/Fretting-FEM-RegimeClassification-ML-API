import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Loading data from the csv file
df= pd.read_csv(r"V:\VZU\Python\Fretting_models\fretting_fem_regimeclassification_ml_api\data\ML_fretting_screening.csv", encoding="cp1252")

#Basic information about data
df.columns.to_list
df.info()
df.isnull().sum()

#Dropping rows with null values
df_ml = df.dropna().reset_index(drop=True)
df_ml.isnull().sum()

#Derive features for better relationship between data
df_ml['µP'] = df_ml['µ'] * df_ml['P']
df_ml['µPD'] = df_ml['µ'] * df_ml['P'] * df_ml['D']
df_ml['D_over_a'] = df_ml['D'] / df_ml['a']

#Seperating features and target
X = df_ml[['P', 'D', 'µ', 'a', 'µP', 'µPD', 'D_over_a']]
Y1 = df_ml[['Regime']]


# Heatmap for the feature correlation using Pearson's coefficient
plt.Figure(figsize=(8,5))
sns.heatmap(X.corr(), annot=True, cmap='plasma')
plt.title("Feature Correlation", weight = 'bold')
plt.show()