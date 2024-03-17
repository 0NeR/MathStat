from scipy import stats
import numpy as np
import matplotlib.pyplot as plt



def get_bootstrap(x, n_sample):
    sample = np.random.choice(x, size = (x.size, n_sample), replace = True)
    return sample

def mu_3_i(arr_i):
    mu_3_i = 0
    x_mean_i = np.mean(arr_i)
    for jj in range(len(arr_i)):
        mu_3_i += (arr_i[jj] - x_mean_i)**3
    return mu_3_i/len(arr_i)






a_list = "0.013 0.052 0.076 0.298 0.534 0.635 0.679 0.718 0.791 0.936 0.968 1.039 1.044 1.223 1.279 1.339 1.466 1.600 1.729 1.729 2.104 2.295 3.703 3.749 4.767"
x_list = list(map(float, a_list.split(' ')))
x_list = np.array(x_list)
N = 1000
k = round(1+np.log2(N))

x_boot = get_bootstrap(x_list, N)
print(x_boot.shape)

alp1 = np.mean(x_boot, axis = 0)
alp2 = np.mean(x_boot**2, axis = 0)

mu2 = alp2 - alp1**2
mu3 = np.apply_along_axis(mu_3_i, 0, x_boot)

x_boot_assim = mu3 / (mu2 ** (3/2))
L = np.max(x_boot_assim) - np.min(x_boot_assim)
D = L/k

print('L, k, D = ', L, k, D)

plt.hist(x_boot_assim, bins = k, density=True, color='b')
plt.title('~_assimetration kf')
plt.xlabel('x', loc='right')
plt.ylabel('~_P(x)', loc='top')
plt.yticks(np.arange(0, 1.05, 0.05))
plt.ylim(0, 1.05)
plt.grid()
plt.show()
