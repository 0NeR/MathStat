import scipy.stats as sps
import matplotlib.pyplot as plt
import numpy as np




Tetta = 7
Betta = 0.95
n = 100

Xn = sps.pareto.rvs(b =Tetta-1, size = n)
print("Xn:")
print(Xn)
print()

print("med = ", np.median(Xn))
print()

# ---------------------------------------------

print("1 : Асимптотический доверительный интервал для med (через ММП)")
print()

Tetta_mmp = n/np.sum(np.log(Xn)) + 1
t1 = sps.norm(loc = 0, scale = 1).ppf((1-Betta)/2)
t2 = sps.norm(loc = 0, scale = 1).ppf((1+Betta)/2)

board1 = 2**((Tetta_mmp - 1)**(-1)) - (np.log(2) * 2**((Tetta_mmp - 1)**(-1)) * t2) / (n**0.5 * (Tetta_mmp - 1)) 
board2 = 2**((Tetta_mmp - 1)**(-1)) - (np.log(2) * 2**((Tetta_mmp - 1)**(-1)) * t1) / (n**0.5 * (Tetta_mmp - 1))
l = board2 - board1
print("Доверительный интервал: (", board1, ";", board2, ")")

print("Длина доверительного интервала", l)
print()

#-------------------------------------------------

print("2 : Асимптотический доверительный интервал для параметра Tetta (через ММП)")
print()
Tetta_mmp = n/np.sum(np.log(Xn)) + 1
t1 = sps.norm(loc = 0, scale = 1).ppf((1-Betta)/2)
t2 = sps.norm(loc = 0, scale = 1).ppf((1+Betta)/2)

board1_mmp = Tetta_mmp - ((Tetta_mmp - 1) * t2) / n**0.5
board2_mmp = Tetta_mmp - ((Tetta_mmp - 1) * t1) / n**0.5
l = board2_mmp - board1_mmp
print("Доверительный интервал: (", board1_mmp, ";", board2_mmp, ")")

print("Длина доверительного интервала", l)
print()

#---------------------------------------------------

print(" 3 :  Бутстраповский параметрический доверительный интервал")
print()

def get_bootstrap(x, n_sample):
    sample = np.random.choice(x, size = (x.size, n_sample), replace = True)
    return sample

N = 50000
x_boot = get_bootstrap(Xn, N)
x_boot_omm = np.sort(n / np.sum(np.log(x_boot), axis = 0) + 1) # вариац ряд
k1 = int((1 - Betta) * N / 2)
k2 = int((1 + Betta) * N / 2)

board1_boot_p = x_boot_omm[k1]
board2_boot_p = x_boot_omm[k2]
l = board2_boot_p - board1_boot_p
print("Доверительный интервал: (", board1_boot_p, ";", board2_boot_p, ")")
print("Длина доверительного интервала", l)
print()

#--------------------------------------------------

print("4 : Бутстраповский непараметрический доверительный интервал")
print()

def get_bootstrap(x, n_sample):
    sample = np.random.choice(x, size = (x.size, n_sample), replace = True)
    return sample

N = 1000
x_boot = get_bootstrap(Xn, N)
Tetta_mmp = n/np.sum(np.log(Xn)) + 1
teta_boot = n / np.sum(np.log(x_boot), axis = 0) + 1

delta = teta_boot - Tetta_mmp
delta_sort = np.sort(delta) # вариац ряд
k1 = int((1 - Betta) * N / 2)
k2 = int((1 + Betta) * N / 2)

board1_boot = Tetta_mmp - delta_sort[k2]
board2_boot = Tetta_mmp - delta_sort[k1]
l = board2_boot - board1_boot

print("Доверительный интервал: (", board1_boot, ";", board2_boot, ")")
print("Длина доверительного интервала", l)
print()

#-----------------------------------------------------------

print("5 : Сравнение доверительных интервалов")

plt.xlim(Tetta*0.5, Tetta*1.5)
plt.ylim(0, 3.5)

plt.text(Tetta - 1, 3.1, round(board2_mmp - board1_mmp, 4))
plt.plot([board1_mmp, board2_mmp], [3,3], color='b', label = "Асимптотический")

plt.text(Tetta - 1, 2.1, round(board2_boot_p - board1_boot_p, 4))
plt.plot([board1_boot_p, board2_boot_p], [2,2], color='orange', label = "Бутст. период.")

plt.text(Tetta - 1, 1.1, round(board2_boot - board1_boot, 4))
plt.plot([board1_boot, board2_boot], [1,1], color='purple', label = "Бутст. непериод.")

plt.plot([Tetta, Tetta], [0.25,3.25], color='black', linestyle='--')
plt.legend()
plt.show()

