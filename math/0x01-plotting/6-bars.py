#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

labels = ["Farrah", "felicia", "fred"]
apples = fruit[0]
bananas = fruit[1]
orange = fruit[2]
peach = fruit[3]

width = 0.5

fig, ax = plt.subplots()

ax.bar(labels, apples, width, label='Apples', color="red")
ax.bar(labels, bananas, width, bottom=apples, label='Bananas', color="yellow")
ax.bar(labels, orange, width, bottom=apples + bananas,
       label='Orange', color="#ff8000")
ax.bar(labels, peach, width, bottom=apples + bananas + orange,
       label='Peach', color="#ffe5b4")

list_bins = list(range(0, 81, 10))
ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
plt.yticks(list_bins)
ax.legend()

plt.show()
