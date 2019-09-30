from CH2_KNN.KNN import *
import matplotlib.pyplot as plt

group, labels = create_data_set()

print(group)

print(labels)

plt.figure()

plt.scatter(group[:, 0], group[:, 1])

plt.show()

cl = classify_2([0, 0], group, labels, 3)

print(cl)
