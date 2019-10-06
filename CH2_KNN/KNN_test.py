from CH2_KNN.KNN import *
import matplotlib.pyplot as plt
from numpy import *

group, labels = create_data_set()

print(group)


dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')

fig = plt.figure()

ax = fig.add_subplot(111)

ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2],
           15.0*array(dating_labels), 15.0*array(dating_labels))

plt.show()

norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
#
# print(norm_mat)
#
# print(ranges)
#
# print(min_vals)


# dating_calss_test()


# classify_person()

handwriting_class_test()