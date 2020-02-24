import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
X=pd.read_csv('train.csv', delimiter=',', usecols = ['qid','question_text'])
y=pd.read_csv('train.csv', delimiter=',', usecols = ['target'])

X_train, X_test,y_train,y_test=train_test_split(X, y, test_size=0.25, random_state=12,stratify=y)
with open('train_75.csv', 'w') as f:
     pd.concat([X_train, y_train], axis=1).to_csv(f)
with open('valid_25.csv', 'w') as f:
     pd.concat([X_test, y_test], axis=1).to_csv(f)

