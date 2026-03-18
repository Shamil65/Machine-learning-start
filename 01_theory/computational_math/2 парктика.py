import numpy as np

# Матрица коэффициентов
A = np.array([
    [-1.3, -1.7, 1.6],
    [-2.6, 3.3, 5.3],
    [-2.1, 2.9, 3.8]
])

# Вектор свободных членов
B = np.array([5.3, 6.6, -0.3])

# Главный определитель
OPR = np.linalg.det(A)

# Проверка, можно ли применить метод Крамера
if OPR == 0:
    print("Метод Крамера неприменим: система либо несовместна, либо имеет бесконечно много решений.")
else:
    # Определители для каждой переменной
    Ax1 = A.copy()
    Ax2 = A.copy()
    Ax3 = A.copy()
    
    Ax1[:,0] = B
    Ax2[:,1] = B
    Ax3[:,2] = B
    
    OPRx1 = np.linalg.det(Ax1)
    OPRx2 = np.linalg.det(Ax2)
    OPRx3 = np.linalg.det(Ax3)
    
    # Решения
    x1 = OPRx1 / OPR
    x2 = OPRx2 / OPR
    x3 = OPRx3 / OPR
    
    print("Главный определитель OPR =", OPR)
    print("Определители для переменных: OPRx1 =", OPRx1, ", OPRx2 =", OPRx2, ", OPRx3 =", OPRx3)
    print("\nРешение системы:")
    print("x1 =", x1)
    print("x2 =", x2)
    print("x3 =", x3)