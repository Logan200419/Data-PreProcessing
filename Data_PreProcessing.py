import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

#importing the dataset
data_set = pd.read_csv('Data_set/Data.csv')

independent = data_set.iloc[:, :-1].values
dependent = data_set.iloc[:, -1].values
#print(independent)
#print(dependent)

#Missing Data
# Either average data will be filled or the data entry will be deleted
Imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
Imputer.fit(independent[:, 1:3])
independent[:, 1:3] = Imputer.transform(independent[:, 1:3])
#print(independent)

#Categorical Encoding of dataset (independent data)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
print(np.array(ct.fit_transform(independent)))

#Label Encoding for dependent data from dataset
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
print(le.fit_transform(dependent))






