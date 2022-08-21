import matplotlib.pyplot as plt
import numpy as np

plt.xlabel('Inference Time(ms)')
plt.ylabel('Parameters(million)')


x = np.array([6.26, 9.13, 18.63])
y = np.array([6.3, 10.4, 25.3])
txt = ['mobileNetV2', 'hrNet_w18', 'resNet50']

plt.scatter(x, y, s=100, c='green', alpha=0.5)
for i in range(len(x)):
    plt.annotate(txt[i], xy=(x[i], y[i]), xytext=(x[i]-1.3, y[i]-1.8))

plt.xticks(np.linspace(0,20,3,endpoint=True))#设置横坐标间隔
plt.yticks(np.linspace(0,30,4,endpoint=True))#设置纵坐标间隔

plt.grid(True)
plt.show()