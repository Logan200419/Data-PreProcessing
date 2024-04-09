#16 to end

import pandas as pd

data = pd.read_csv("train.csv",delimiter=',')
data = data.drop('instrumentalness', axis=1)
independent = data.iloc[:, 5:-1].values
#No missing data

#Categorical encoding
independent1 = independent
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
independent = ct.fit_transform(independent)
print(independent)

#Feature Scaling
sc = StandardScaler()
independent[:, 2:] = sc.fit_transform(independent[:, 2:])
print(independent)


