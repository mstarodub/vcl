import csv
import matplotlib.pyplot as plt
import sys

with open(sys.argv[1], newline="") as f:
    reader = csv.reader(f, skipinitialspace=True)
    # skip header row
    next(reader)
    test_acc = [float(row[1].strip('"')) for row in reader]

tasks = range(1, len(test_acc) + 1)

plt.scatter(tasks, test_acc, marker="o")
plt.xticks(tasks)
plt.xlabel("task")
plt.ylabel("test accuracy")
plt.title("pmnist")
plt.grid(True)
plt.show()
