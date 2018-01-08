weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25],
              [63, 28], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31], [65, 52], [57, 23],
              [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], [59, 46], [67, 23],
              [85, 37], [55, 40], [63, 30]]
blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395,
                               434, 220, 374, 308, 220, 311, 181, 274, 303, 244]

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for idx in range(25):
   ax.scatter(weight_age[idx][0],weight_age[idx][1], blood_fat_content[idx],'ro')

plt.show()
