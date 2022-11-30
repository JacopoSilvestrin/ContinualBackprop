import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
v_1 = np.load("/home/jacopo/Desktop/Test_NS_RL/BitFlip/ResetTest/FisherUnitTest/contErrors.npy")
v_2 = np.load("/home/jacopo/Desktop/Test_NS_RL/BitFlip/ForgettingTest/RandomTest/benchErrors.npy")
v_3 = np.load("/home/jacopo/Desktop/Test_NS_RL/BitFlip/ForgettingTest/RandomTest/randErrors.npy")
v_4 = np.load("/home/jacopo/Desktop/Test_NS_RL/BitFlip/ForgettingTest/RandomTest/fisherErrors.npy")
v_5 = np.load("/home/jacopo/Desktop/Test_NS_RL/BitFlip/ResetTest/FisherUnitTest/fisherUnitErrors.npy")

n_samples = v_1.shape[1]
bin_dim = 100
plot_dim = int(n_samples / bin_dim)

v_1_mean = np.zeros(plot_dim)
v_2_mean = np.zeros(plot_dim)
v_3_mean = np.zeros(plot_dim)
v_4_mean = np.zeros(plot_dim)
v_5_mean = np.zeros(plot_dim)
v_1_sem = np.zeros(plot_dim)
v_2_sem = np.zeros(plot_dim)
v_3_sem = np.zeros(plot_dim)
v_4_sem = np.zeros(plot_dim)
v_5_sem = np.zeros(plot_dim)

for i in range(plot_dim):
    v_1_mean[i] = np.mean(v_1[:,bin_dim*i:bin_dim*(i+1)])
    v_1_sem[i] = stats.sem(v_1[:,bin_dim*i:bin_dim*(i+1)], axis=None)

    v_2_mean[i] = np.mean(v_2[:, bin_dim * i:bin_dim * (i + 1)])
    v_2_sem[i] = stats.sem(v_2[:, bin_dim * i:bin_dim * (i + 1)], axis=None)

    v_3_mean[i] = np.mean(v_3[:, bin_dim * i:bin_dim * (i + 1)])
    v_3_sem[i] = stats.sem(v_3[:, bin_dim * i:bin_dim * (i + 1)], axis=None)

    v_4_mean[i] = np.mean(v_4[:, bin_dim * i:bin_dim * (i + 1)])
    v_4_sem[i] = stats.sem(v_4[:, bin_dim * i:bin_dim * (i + 1)], axis=None)

    v_5_mean[i] = np.mean(v_5[:, bin_dim * i:bin_dim * (i + 1)])
    v_5_sem[i] = stats.sem(v_5[:, bin_dim * i:bin_dim * (i + 1)], axis=None)

sns.set()
x = np.arange(plot_dim)
plt.figure()
plt.plot(x, v_1_mean, 'b', label='MSE,CBP')
plt.fill_between(x, v_1_mean - v_1_sem, v_1_mean + v_1_sem, color='b', alpha=0.2)
#plt.plot(x, v_2_mean, 'r', label='MSE, BP')
#plt.fill_between(x, v_2_mean - v_2_sem, v_2_mean + v_2_sem, color='r', alpha=0.2)
#plt.plot(x, v_3_mean, 'g', label='MSE, RBP')
#plt.fill_between(x, v_3_mean - v_3_sem, v_3_mean + v_3_sem, color='g', alpha=0.2)
#plt.plot(x, v_4_mean, 'y', label='MSE, FBP')
#plt.fill_between(x, v_4_mean - v_4_sem, v_4_mean + v_4_sem, color='y', alpha=0.2)
plt.plot(x, v_5_mean, 'c', label='MSE, FUBP')
plt.fill_between(x, v_5_mean - v_5_sem, v_5_mean + v_5_sem, color='c', alpha=0.2)

plt.title('BitFlippling regression problem, forgetting test m=20,f=15')

plt.legend(title='Legend')
plt.show()

plt.figure()
