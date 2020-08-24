# my first ever machine learning program

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train = pd.read_csv("/Users/alexdrozdz/Desktop/train.csv")

X, y = train.iloc[:, 1:], train.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1000)

randomForest=RandomForestClassifier()

# train basic random forest model
randomForest.fit(X_train, y_train)

y_pred = randomForest.predict(X_test)
print('Accuracy: ' + str(round(accuracy_score(y_test, y_pred), 3)*100) + '%')
# this basic model provides 96% of accuracy
