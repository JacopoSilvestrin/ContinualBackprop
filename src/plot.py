import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
v_1 = np.load("/home/jacopo/CBP_Results/BitFlipping/ForgetAndLearn/FairComparison/benchErrors.npy")
v_2 = np.load("/home/jacopo/CBP_Results/BitFlipping/ForgetAndLearn/FairComparison/contErrors.npy")
v_3 = np.load("/home/jacopo/CBP_Results/BitFlipping/ForgetAndLearn/FairComparison/detErrors.npy")
v_4 = np.load("/home/jacopo/CBP_Results/BitFlipping/ForgetAndLearn/FairComparison/fisherUnitErrors.npy")
v_5 = np.load("/home/jacopo/CBP_Results/BitFlipping/ForgetAndLearn/FairComparison/bigContErrors.npy")
v_6 = np.load("/home/jacopo/CBP_Results/BitFlipping/ForgetAndLearn/FairComparison/growContErrors.npy")


n_samples = v_1.shape[1]
bin_dim = 100
plot_dim = int(n_samples / bin_dim)

v_1_mean = np.zeros(plot_dim)
v_2_mean = np.zeros(plot_dim)
v_3_mean = np.zeros(plot_dim)
v_4_mean = np.zeros(plot_dim)
v_5_mean = np.zeros(plot_dim)
v_6_mean = np.zeros(plot_dim)

v_1_sem = np.zeros(plot_dim)
v_2_sem = np.zeros(plot_dim)
v_3_sem = np.zeros(plot_dim)
v_4_sem = np.zeros(plot_dim)
v_5_sem = np.zeros(plot_dim)
v_6_sem = np.zeros(plot_dim)

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

    v_6_mean[i] = np.mean(v_6[:, bin_dim * i:bin_dim * (i + 1)])
    v_6_sem[i] = stats.sem(v_6[:, bin_dim * i:bin_dim * (i + 1)], axis=None)

sns.set()
x = np.arange(plot_dim)
plt.figure()
plt.plot(x, v_1_mean, 'b', label='MSE,Backprop')
plt.fill_between(x, v_1_mean - v_1_sem, v_1_mean + v_1_sem, color='b', alpha=0.2)
plt.plot(x, v_2_mean, 'r', label='MSE, ContBackprop')
plt.fill_between(x, v_2_mean - v_2_sem, v_2_mean + v_2_sem, color='r', alpha=0.2)
plt.plot(x, v_3_mean, 'g', label='MSE, GrowingFisherBackprop')
plt.fill_between(x, v_3_mean - v_3_sem, v_3_mean + v_3_sem, color='g', alpha=0.2)
plt.plot(x, v_4_mean, 'y', label='MSE, FisherBackprop')
plt.fill_between(x, v_4_mean - v_4_sem, v_4_mean + v_4_sem, color='y', alpha=0.2)
plt.plot(x, v_5_mean, 'c', label='MSE, BigContBackprop')
plt.fill_between(x, v_5_mean - v_5_sem, v_5_mean + v_5_sem, color='c', alpha=0.2)
plt.plot(x, v_6_mean, 'm', label='MSE, GrowingContBackprop')
plt.fill_between(x, v_6_mean - v_6_sem, v_6_mean + v_6_sem, color='m', alpha=0.2)

plt.title('BitFlippling regression problem, forgetting test m=20,f=15')

plt.legend(title='Legend')
plt.show()

plt.figure()
