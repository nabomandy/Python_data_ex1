import mglearn
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager, rc

data = pd.read_csv("/home/andybae/study_bigdata/회귀분석/academy1.csv")
print( data )

print()
print("======================")
print()

# 군집 모델을 만듭니다.
kmeans = KMeans(n_clusters=3)
kmeans.fit( data.iloc[:,1:]) 

kmeans.fit_predict( data.iloc[:,1:])


print()
print("======================")
print()

rc('font', family="Malgun Gothic") # 한글가능한 폰트 설정
print("클러스터 레이블 ",kmeans.labels_)

mglearn.discrete_scatter(data.iloc[:,1], data.iloc[:,2], kmeans.labels_)
plt.legend(["클러스터 0", "클러스터 1", "클러스터 2"], loc='best')
plt.xlabel("국어점수")
plt.ylabel("영어점수")
plt.show()

print()
print("======================")
print()

# 국어점수 100, 영어줌수 80인 새로운 학생이 입학하였습니다
# 이 학생은 몇번 클러스터에 포함되어야  합니까?
pre=kmeans.predict( [ [100,80] ] )
print(pre)