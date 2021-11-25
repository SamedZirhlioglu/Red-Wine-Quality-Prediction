from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1234)
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
A = np.random.randint(5, size=(25, 10))
X = ["knn","decision tree","bagg", "svm", "naive","log","random"]
x = np.array([[i] * 10 for i in range(25)]).ravel() # x coordinates of each bar
y = np.array([i for i in range(10)] * 25) # y coordinates of each bar
z = np.zeros(25*10) # z coordinates of each bar

print(x)
print(y)
print(z)

dx = np.ones(25*10) # length along x-axis of each bar
dy = np.ones(25*10) # length along y-axis of each bar
dz = A.ravel() # length along z-axis of each bar (height)

print(A)
print(dz)

from matplotlib import cm
from matplotlib.colors import Normalize
cmap = cm.get_cmap('plasma')
norm = Normalize(vmin=min(dz), vmax=max(dz))
colors = cmap(norm(dz))


sc = cm.ScalarMappable(cmap=cmap,norm=norm)
sc.set_array([])
plt.colorbar(sc)
ax1.bar3d(x, y, z, dx, dy, dz, color=colors)
ax1.set_xlim(0, 100)

plt.show()