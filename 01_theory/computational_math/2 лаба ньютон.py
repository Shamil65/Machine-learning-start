# Метод ньютона


import sympy as sp

def newton_method(func_str, x0, eps=1e-6, max_iter=100):

    x = sp.symbols('x')
    f_sym = sp.sympify(func_str)
    
    f_prime = sp.lambdify(x, sp.diff(f_sym, x), 'math')        # f'(x)
    f_double_prime = sp.lambdify(x, sp.diff(f_sym, x, 2), 'math')  # f''(x)
    
    x_n = x0
    for i in range(max_iter):
        f1 = f_prime(x_n)
        f2 = f_double_prime(x_n)
        
        if f2 == 0:
            print("Вторая производная равна нулю, метод не применим")
            return x_n
        
        x_next = x_n - f1 / f2
        
        if abs(x_next - x_n) < eps:
            print(f"Сошлось за {i+1} итераций")
            return x_next
        
        x_n = x_next
    
    print("Достигнуто максимальное число итераций")
    return x_n


func_str = input("Введите функцию f(x), например, x**4 - 18*x**2 + 6: ")
x0 = float(input("Введите начальное приближение x0: "))

root = newton_method(func_str, x0)
print(f"Приближённый минимум: {root:.6f}")
