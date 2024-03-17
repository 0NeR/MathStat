import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def get_bootstrap(x, n_sample):
    sample = np.random.choice(x, size = (x.size, n_sample), replace = True)
    return sample






a_list = "0.013 0.052 0.076 0.298 0.534 0.635 0.679 0.718 0.791 0.936 0.968 1.039 1.044 1.223 1.279 1.339 1.466 1.600 1.729 1.729 2.104 2.295 3.703 3.749 4.767"
x_list = list(map(float, a_list.split(' ')))
x_list = np.array(x_list)
N=1000
k = round(1 + np.log2(N))

x_boot = get_bootstrap(x_list, N)
x_boot_mean = np.mean(x_boot, axis = 0)
L = np.max(x_boot_mean) - np.min(x_boot_mean)
D = L/k
print('L, k, D = ', L, k, D)
sns.distplot(x_boot_mean, fit=stats.norm, color='b', bins = k)

plt.title('~_mean')
plt.xlabel('x', loc='right')
plt.ylabel('~_P(x)', loc='top')
plt.grid()
plt.show()
