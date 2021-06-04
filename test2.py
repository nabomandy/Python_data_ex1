from sklearn import svm
import numpy as np

clf = svm.SVC()

x = [ [0,0], [1,0], [0,1], [1,1]]
y = [ 0, 1, 1, 0]
print(np.shape(x))
print(np.shape(y))
clf.fit( [ [0,0], [1,0], [0,1], [1,1] ], [0, 1, 1, 0] )

results = clf.predict( [ [0,0], [1,0] ])
print(results)