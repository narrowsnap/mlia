import kNN
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

# group, labels = kNN.create_data_set()
# input = [1, 0]
# result = kNN.classify0(input, group, labels, 3)
# print(result)

datingMat, labels = kNN.file2matrix('./../datingTestSet.txt')
ax = fig.add_subplot(111)
ax.scatter(datingMat[:, 1], datingMat[:, 2], 15.0*np.array(labels),
           15.0*np.array(labels))
plt.show()

