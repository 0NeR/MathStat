import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps


Teta = 1
Betta = 0.95
N = 100

Xn = sps.uniform(loc = Teta, scale = Teta).rvs(size=N)
print("Xn:")
print(Xn)
print()

t1 = ((1 - Betta) / 2) ** (1 / N)
t2 = ((1 + Betta) / 2) ** (1 / N)
Xmax = np.max(Xn)

board1_acc = Xmax / (t2 + 1) 
board2_acc = Xmax / (t1 + 1)
l = board2_acc - board1_acc

print("1 : Точный доверительный интервал")
print("Доверительный интервал: (", board1_acc, ";", board2_acc, ")")
print("Длина доверительного интервала", l)
print()

#--------------------------------

teta_omm = 2/3 * np.mean(Xn)
t1 = sps.norm(loc = 0, scale = 1).ppf((1-Betta)/2)
t2 = sps.norm(loc = 0, scale = 1).ppf((1+Betta)/2)
alpha1 = np.mean(Xn)
alpha2 = np.mean(Xn**2)

board1_mm = teta_omm - 2/3 * (alpha2 - alpha1**2)**0.5 * t2 / N**0.5
board2_mm = teta_omm - 2/3 * (alpha2 - alpha1**2)**0.5 * t1 / N**0.5
l = board2_mm - board1_mm


print(" 2 : Асимптотический доверительный интервал (через метод моментов")
print("Доверительный интервал: (", board1_mm, ";", board2_mm, ")")
print("Длина доверительного интервала", l)
print()


#---------------------------------

def get_bootstrap(x, n_sample):
    sample = np.random.choice(x, size = (x.size, n_sample), replace = True)
    return sample

N = 50000
x_boot = get_bootstrap(Xn, N)
x_boot_omm = np.sort(2/3 * np.mean(x_boot, axis = 0)) # вариац ряд
k1 = int((1 - Betta) * N / 2)
k2 = int((1 + Betta) * N / 2)

board1_boot = x_boot_omm[k1]
board2_boot = x_boot_omm[k2]
l = board2_boot - board1_boot
print("3 : Бутстраповский параметрический доверительный интервал (используется ОММ)")
print(" Доверительный интервал: (", board1_boot, ";", board2_boot, ")")
print("Длина доверительного интервала", l)
print()


#-----------------------------------

print("4 : Сравнение доверительных интервалов")
plt.xlim(Teta*0.9, Teta*1.1)
plt.ylim(0, 3.5)

plt.text(Teta - 0.005, 3.1, round(board2_acc - board1_acc, 4))
plt.plot([board1_acc, board2_acc], [3,3], color='b', label = "Точный")

plt.text(Teta - 0.005, 2.1, round(board2_mm - board1_mm, 4))
plt.plot([board1_mm, board2_mm], [2,2], color='orange', label = "Асимпт.")

plt.text(Teta - 0.005, 1.1, round(board2_boot - board1_boot, 4))
plt.plot([board1_boot, board2_boot], [1,1], color='purple', label = "Бутст.")
plt.legend()
plt.show()
