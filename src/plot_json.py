import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import json

# Load data
v_c = np.zeros((10,1000))
v_f = np.zeros((10,1000))

for i in range(10):
    cbp = json.load(open('/home/jacopo/Desktop/DataCheetah/cbp/seed{}.json'.format(i)))
    fisher = json.load(open('/home/jacopo/Desktop/DataCheetah/fisher/seed{}.json'.format(i)))
    for j in range(1000):
        v_c[i, j] = cbp[j][2]
        v_f[i, j] = fisher[j][2]


v_1_mean = np.mean(v_c, axis=0)
v_2_mean = np.mean(v_f, axis=0)

v_1_sem = stats.sem(v_c, axis=0)
v_2_sem = stats.sem(v_f, axis=0)


sns.set()
x = np.arange(1000)
plt.figure()
plt.plot(x, v_1_mean, 'b', label='Mean disc Return, ContinualB')
plt.fill_between(x, v_1_mean - v_1_sem, v_1_mean + v_1_sem, color='b', alpha=0.2)
plt.plot(x, v_2_mean, 'r', label='Mean disc Return, FisherB')
plt.fill_between(x, v_2_mean - v_2_sem, v_2_mean + v_2_sem, color='r', alpha=0.2)

plt.title('Half Cheetah')

plt.legend(title='Legend')
plt.show()

