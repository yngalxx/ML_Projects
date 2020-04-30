import matplotlib.pyplot as pt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv("path").values

rfc=RandomForestClassifier()

# TRAINING DATA SET:
xtrain=data[0:33600, 1:]
train_label=data[0:33600, 0]

rfc.fit(xtrain, train_label)

# SAMPLE:
z=int(input("Enter an integer from the compartment: (0,33600) "))
d=xtrain[z]
d.shape=(28,28)
pt.imshow(255-d,cmap=None)
print("Predicted digit is: ",rfc.predict( [xtrain[z]] ))
pt.show()
print()

# ACCURACY:
xtest=data[33600:,1:]
actual_label=data[33600:, 0]

p=rfc.predict(xtest)
count=0
i=1
while i<8400:
    count+=1 if p[i]==actual_label[i] else 0
    i=i+1
print("Program provides with ", (count/8400)*100,"% accuracy")

