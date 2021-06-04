"""
iris
"""


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

csv = pd.read_csv("/home/andybae/study_bigdata/iris.csv")
csv_data = csv[ ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"] ]
csv_label = csv[ ["Name"] ]

csv_data.shape # (150, 4)

csv_label.shape # (150, 1)

train_data, test_data, train_label, test_label = train_test_split( csv_data, csv_label)
train_data.shape # (112,4)

test_data.shape # (38,4)

model = RandomForestClassifier()
model.fit(train_data, train_label) # weight와 bias 정하는 함수

pre = model.predict( test_data )
print(test_data.shape)
print(pre)

ac_score = metrics.accuracy_score(test_label, pre)
print("정답률 = ", ac_score)

test_data.loc[[64],:]

test_label.loc[[64]]

output = model.predict(test_data.loc[[64], :])
print(output)