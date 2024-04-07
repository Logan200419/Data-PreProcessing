import pandas as pd
import numpy as np

data = pd.read_csv('Test1.csv')

independent = data.iloc[:, :-1].values
dependent = data.iloc[:, -1].values

from sklearn.impute import SimpleImputer
im = SimpleImputer(missing_values=np.nan,strategy='mean')
independent[:, 1:] = im.fit_transform(independent[:, 1:])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
independent = ct.fit_transform(independent)

le = LabelEncoder()
dependent = le.fit_transform(dependent)
#print(independent)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)
print(X_test)

