import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

mr = pd.read_csv("/home/andybae/study_bigdata/mushroom.csv", header=None) # 파일 경로 때문에 colab에서는 이대로 실행 안될 수 있다

label = []
data = []
attr_list = []

for row_index, row in mr.iterrows(): # DataFrame 객체의 iterrows()메소드를 for문과 함께 # 행인덱스와, 행 데이터를 항 행식 반환
    label.append(row.iloc[0]) # 0번 컬럼에 독이 들어있는지 없는지 정보를 label 리스트에 담는다.
    row_data = []
    for v in row.iloc[1:]:
        row_data.append(ord(v))
    
    data.append(row_data)

ord('a') # 97

print(np.shape(data)) # (8124, 22)

print(np.shape(label)) # (8124, )

data_train, data_test, label_train, label_test = train_test_split(data, label)

clf = RandomForestClassifier()
clf.fit(data_train, label_train)

pre = clf.predict ( data_test ) # 데이터 예측

ac_score = metrics.accuracy_score( label_test, pre)
cl_report = metrics.classification_report( label_test, pre)

print("정답률 =", ac_score)
print("리포트 =\n", cl_report)