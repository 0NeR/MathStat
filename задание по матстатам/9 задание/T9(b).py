import scipy.stats as sps
import scipy.optimize as opt
import scipy.integrate as integr
import numpy as np
import matplotlib.pyplot as plt

# Функции для подсчета логарифма функции правдоподобия 
def optLog(teta_cur, pairs_cur):
    L = 1
    for i in range(len(pairs_cur[0])):
        integral_value = integr.quad(get_norm_pdf, pairs_cur[0][i], pairs_cur[1][i], args = (teta_cur[0], teta_cur[1]))
        L *= integral_value[0]
        
    return -np.log(L)

def get_norm_pdf(x_cur, teta1_cur, teta2_cur):
    return sps.norm.pdf(x = x_cur, loc = teta1_cur, scale = teta2_cur)

# Эмпирическая функция распределения по выборке
def emp_func(sample_cur, n_cur):
    # Получаем точки - пары x и y координат 
    xy_pairs = np.vstack(( np.sort(sample_cur), np.arange(0, n_cur) / n_cur )).T
    uniq_elem = xy_pairs[0]
    for pair_cur in xy_pairs:
        if pair_cur[0] == uniq_elem[0]:
            pair_cur[1] = uniq_elem[1]
        else:
            uniq_elem = pair_cur
    return xy_pairs

# Функция распределения нормального закона
def norm_func(sample_cur, teta1_cur, teta2_cur):
    x1 = np.sort(sample_cur)
    y1 = sps.norm.cdf(x = x1, loc = teta1_cur, scale = teta2_cur)
    return np.vstack((x1, y1)).T

print("рассмотрим криетрий согласия Пирсона для сложной гипотезы")
# Константы нашей задачи
xn = np.array([0]*5 + [1]*8 + [2]*6 + [3]*12 + [4]*14 + [5]*18 + [6]*11 + [7]*6 + [8]*13 + [9]*7)
n = xn.size
sqrt_n = n ** 0.5

m = np.array([5, 8, 6, 12, 14, 18, 11, 6, 13, 7])
k = m.size
s = 2


pairs = np.array([[[-np.inf]*5 + [1]*8 + [2]*6 + [3]*12 + [4]*14
                   + [5]*18 + [6]*11 + [7]*6 + [8]*13 + [9]*7] , 
                  [[1]*5 + [2]*8 + [3]*6 + [4]*12 + [5]*14 +
                   [6]*18 + [7]*11 + [8]*6 + [9]*13 + [np.inf]*7]])

pairs = np.squeeze(pairs, axis=1)

print("(1) : n  = ", n)
for i in range(len(pairs[0])//10):
    print(pairs[0][i], pairs[1][i])

print()
print("(2) : ")
print(get_norm_pdf(-np.inf, 4, 3))
print()
print("(3) : ")
print(integr.quad(get_norm_pdf, -np.inf, 0, args = (4, 3))[0])
    
# Максимизируем логарифм
optima = opt.minimize(fun = optLog,  x0 = [5, 2], args = (pairs,)) # из графика выборки возьмем 5, 2*sigma (95%) = 4

tetta_1 = round(optima.x[0], 4)
tetta_2 = round(optima.x[1], 4)
print()
print("(4) : ", "tetta_1",tetta_1)
print("tetta_2",tetta_2)

# Посчитаем n*pi:
p_full = np.array([integr.quad(get_norm_pdf, pairs[0][i], pairs[1][i], args = (tetta_1, tetta_2))[0] for i in range(n)])
np_full = p_full * n

np_res = np.delete(np_full, np.where(np.diff(np_full) == 0)[0] + 1)
#убираем повторения соотвествующие одному mi и округляем
np_res = np.round(np_res, 3)

print()
print("(5) : ", np_res)

# Cчитаем ~delta
delta_ = np.sum((m - np_res)**2 / np_res)
print()
print("(6) : ")
print("delta_ = ", delta_)

# Посчитаем p-value
p_value = 1 - sps.chi2.cdf(delta_, k - 1 - s)
print()
print("p-value = ", p_value)
print("так как p-value > alpha(0.05) => нет оснований отвергать")
print("-------------------------------------------")
print("смотрим дальше. Rритерий Колмогорова для сложной гипотезы")



# Константы задачи
xn = np.array([0]*5 + [1]*8 + [2]*6 + [3]*12 + [4]*14 + [5]*18 + [6]*11 + [7]*6 + [8]*13 + [9]*7)
n = xn.size
sqrt_n = n ** 0.5

# Получаеncz из исходной выборки
tetta_1 = np.mean(xn)
tetta_2 = np.std(xn)
print()
print("(1) : ", n, tetta_1, tetta_2)

# ~Fn(x)
empf = emp_func(xn, n)
print()
print("(2) : ", empf.shape)
print()
print("(3) : ", empf[:10])

# F(x, vector my_teta)
normf = norm_func(xn, tetta_1, tetta_2) 
print()
print("(4) : ")
print(normf.shape)
print()
print("(5) : ")
print(normf[:10])
print()
# ~delta
delta_ = sqrt_n * max(np.max(np.abs(empf[:, 1] - normf[:, 1])), np.max(np.abs(empf[1:, 1] - normf[:-1, 1])), np.abs(1 - normf[-1, 1])) 
print("(6) : ")
print(np.abs(empf[:, 1] - normf[:, 1])[:10], np.abs(empf[1:, 1] - normf[:-1, 1])[:10], np.abs(1 - normf[-1, 1]))
print()
print("(7) : ")
print("delta_ = ", delta_)
print()
# Воспользуемся параметрическим bootstapом
N = 50000
x_boot = np.array([(np.round(sps.norm.rvs(size = n, loc = tetta_1, scale = tetta_2)))%10 for i in range (N)])
print("(8) : ")
print(x_boot.shape)
print()
teta1_boot = np.mean(x_boot, axis=1)
teta2_boot = np.std(x_boot, axis=1)
print("(9) : ")
print(teta1_boot.shape)
print()
print("(10) : ")
print(teta1_boot[:5])
print()
print("(11) : ")
print(teta2_boot[:5])
print()
empf_boot = np.apply_along_axis(func1d = emp_func, axis = 1, arr = x_boot, n_cur = n) 
print("(12) : ")
print(empf_boot.shape)
print()
print("(13) : ")
print(empf_boot[0][:10])


delta_boot = np.array([])
for i in range(N):
    empf_boot_cur = empf_boot[i]
    # F(x, vector teta*)
    normf_boot_cur = norm_func(x_boot[i], teta1_boot[i], teta2_boot[i]) 
    # *delta
    delta_boot_cur = sqrt_n * max(np.max(np.abs(empf_boot_cur[:, 1] - normf_boot_cur[:, 1])), 
                                  np.max(np.abs(empf_boot_cur[1:, 1] - normf_boot_cur[:-1, 1])), 
                                  np.abs(1 - normf_boot_cur[-1, 1])) 
    delta_boot = np.append(delta_boot, delta_boot_cur)


# вариационный ряд
delta_boot = np.sort(delta_boot) 
print()
print("(14) : ")
print(delta_boot.shape)
print()
L = len([delta for delta in delta_boot if delta >= delta_])
print("(15) : L =  ", L)
print()
p_value = L / N
print("(16) : ")


print("p-value = ", p_value)
print("p-value > alpha(0.05) => нет оснований отвергнуть гипотезу H0")
