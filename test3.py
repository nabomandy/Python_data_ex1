"""
ex02_xor_정확도
"""

import pandas as pd
from sklearn import svm, metrics

xor_input = [
    [0, 0, 0], # 앞에 0,0 x이고 뒤에 0이 y가 된다.
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
]

xor_df = pd.DataFrame( xor_input)
xor_data = xor_df.loc[ :, 0:1]
xor_label = xor_df.loc[ :, 2]
print(xor_data)
print()
print(xor_label)

clf = svm.SVC()
clf.fit(xor_data, xor_label)

pre = clf.predict( xor_data )

ac_score = metrics.accuracy_score(xor_label, pre)
print("정답률 = ", ac_score)