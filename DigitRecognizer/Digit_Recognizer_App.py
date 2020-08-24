import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv("path").values

rfc=RandomForestClassifier()

# train set:
x=data[0:33600, 1:]
y=data[0:33600, 0]

# train basic random forest model
rfc.fit(x, y)

# prediction
test=data[33600:,1:]
y_test=data[33600:, 0]

p=rfc.predict(test)
count=0
i=1
while i<8400:
    count+=1 if p[i]==actual_label[i] else 0
    i=i+1
print("Program provides with ", (count/8400)*100,"% accuracy")

# accuracy was 85% or sth like this 
