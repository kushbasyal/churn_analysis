import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from pipeline import extract, load, transform, run_pipeline

df = extract()
print(df.head())

print(df.shape)

print(df.info())
print(df.describe())

null_values = df.isna().sum()
print('null_values\n', null_values)

# We will hanlde null values later after separating it into training and testing set

df.drop(columns = ['customer_id'], inplace = True)
print(df.columns)

# lets separate the numerical and categorical columns
df_num = df.select_dtypes(include = 'number')
print(df_num)

df_cat = df.select_dtypes(include = 'object')
print(df_cat)

# lets plot a heatmap to see which columns are highly correlated
corr = df_num.corr()
print(corr)

# lets plot a heatmap to see the highly correlated features 
'''sns.heatmap(corr, cmap = 'coolwarm', annot = True,  fmt = '.2f')
plt.show()'''

X = df.iloc[:,:-1].values
print(X)

print(X[0])

for i in df_cat.columns:
    print(df_cat[i].value_counts())
sns.pairplot(df, hue = 'churn')
plt.show()

for i in df_num.columns:
    plt.figure(figsize = (10,6))
    sns.histplot(df_num[i], kde = True)
    plt.show()

y = df.iloc[:,-1].values
print(y[0])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print(X_train[0])
print(X_train.shape)
print(X_test.shape)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 'Unknown')
X_train[:,[5]] = imputer.fit_transform(X_train[:,[5]])
print(np.unique(X_train[:,[5]]))

X_test[:,[5]] = imputer.transform(X_test[:,[5]])
print(np.unique(X_test[:,[5]]))

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(handle_unknown= 'ignore'),[3,4,5,6,7])
], remainder= 'passthrough')
print(ct)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
pipeline = Pipeline(steps = [
    ('composer',ct),
    ('classifier', XGBClassifier())
])
#print(pipeline)
#pipeline.fit(X_train, y_train)

print(X_train[0])
print(y_train)
pipeline.fit(X_train, y_train)
print(pipeline)

y_pred = pipeline.predict(X_test)
print(y_pred)

import numpy as np

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)),1))

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(f'accuracy:\n {accuracy_score(y_test, y_pred)}')
print(f'confusion_matrix:\n {confusion_matrix(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(pipeline, X_train, y_train, cv  = 5)
print(accuracy)
print(accuracy.mean())
print(accuracy.std())

print(df.columns)

import warnings

# Ignore only this specific sklearn warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def churn_pred(tenure, monthly_charges, total_charges,
               contract, payment_method, internet_service,
               tech_support, online_security, support_calls):

    new_data = pd.DataFrame([{
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract': contract,
        'payment_method': payment_method,
        'internet_service': internet_service,
        'tech_support': tech_support,
        'online_security': online_security,
        'support_calls': support_calls
    }])

    pred = pipeline.predict(new_data)
    return "Churn" if pred[0] == 1 else "No Churn"

result = churn_pred(
    12, 75, 900, 'Month-to-month', 'Credit', 'Fiber', 'Yes', 'No', 2
)
print(result)  # Churn or No Churn