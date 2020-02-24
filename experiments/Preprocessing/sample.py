import pandas as pd
train = pd.read_csv("train_75.csv")
print train['question_text'].count('girlfriend')

