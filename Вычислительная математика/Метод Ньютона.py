import math

def newton_sqrt_eq(x0, tol=1e-8, max_iter=100):
    x = x0
    for i in range(max_iter):
        f = math.sqrt(x - 1) - 1/x
        df = 1/(2*math.sqrt(x - 1)) + 1/(x**2)
        x_new = x - f/df
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise ValueError("Метод Ньютона не сошелся")

# Пример использования:
initial_guess = 1.5  # начальное приближение должно быть >1
root = newton_sqrt_eq(initial_guess)
print("Корень:", root)
