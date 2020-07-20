#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
list_bins = list(range(0, 101, 10))
plt.hist(student_grades, bins=list_bins, edgecolor="black")
plt.title("Project A")
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.ylim(0, 30)
plt.xlim(0, 100)
plt.xticks(list_bins)

plt.show()
