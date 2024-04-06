import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

data = pd.read_csv('Data_set/test.csv')

categorical_data = ["Pclass", "Sex", "Embarked"]

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_data)], remainder='passthrough')
X = np.array(ct.fit_transform(data))
print(X)

le = LabelEncoder()
Y = le.fit_transform(data['Survived'])
print(Y)