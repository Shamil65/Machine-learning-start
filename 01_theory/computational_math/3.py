import numpy as np

def f(x):
    return 1 / (1 + np.cos(x) + np.sin(x))

def trapezoidal(a, b, n): # площадь = (ширина) × (средняя высота)
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    return h * (0.5*y[0] + np.sum(y[1:-1]) + 0.5*y[-1])

def simpson(a, b, n): # заменяем кривую не прямыми, а параболами
    if n % 2 != 0:
        raise ValueError("n должно быть чётным для метода Симпсона")
        
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    return (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

a = 0.1
b = 0.9

n = 100

trap_result = trapezoidal(a, b, n)
simp_result = simpson(a, b, n)

print("Метод трапеций:", trap_result)
print("Метод Симпсона:", simp_result)