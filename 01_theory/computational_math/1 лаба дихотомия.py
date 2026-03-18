def dichotomy_method(func, a, b, eps=1e-6, delta=1e-6, max_iter=50):

    for i in range(max_iter):
        x1 = (a + b)/2 - delta
        x2 = (a + b)/2 + delta
        
        f1 = func(x1)
        f2 = func(x2)
        

        if f1 < f2:
            b = x2
        else:
            a = x1
        
        if (b - a)/2 < eps:
            print(f"Сошлось за {i+1} итераций")
            return (a + b)/2
    
    print("Достигнуто максимальное число итераций")
    return (a + b)/2

import math

func_str = input("Введите функцию f(x), например, (x**4 - 18*x**2 + 6)**2: ")
a = float(input("Введите левую границу интервала a: "))
b = float(input("Введите правую границу интервала b: "))

func = eval(f"lambda x: {func_str}")

root = dichotomy_method(func, a, b)
print(f"Приближённый минимум: {root:.6f}")
print(f"f(root) ≈ {func(root):.6e}")

