import matplotlib.pyplot as plt
import numpy as np

scores = []

with open("scores.txt", "r") as file:
    for line in file:
        scores.append(int(line))


maxscores = []
max = 0

for item in scores:
    if item > max:
        max = item
    maxscores.append(max)


avgs = []
scorenp = np.array(scores)
scorenp = np.reshape(scorenp, (int(scorenp.size/1000), 1000))
scorenp = np.average(scorenp, axis=1)
for item in scorenp:
    for i in range(1000):
        avgs.append(item)

plt.xlabel("numbers of games")
plt.ylabel("score")
plt.plot(maxscores, label="Max scores")
plt.plot(avgs, label="Average scores")
plt.legend()
plt.show()