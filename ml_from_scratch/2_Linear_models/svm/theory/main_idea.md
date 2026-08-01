# SVM: Primal vs Dual

Два способа решить одну и ту же задачу — найти разделяющую гиперплоскость
с максимальным зазором (margin). Отличаются тем, по каким переменным идёт
оптимизация: по весам $w, b$ (primal) или по множителям Лагранжа $\alpha_i$ (dual).

---

## 1. Primal SVM (градиентный спуск)

### 1.1 Постановка задачи

Soft-margin SVM минимизирует норму весов (максимизирует зазор) плюс штраф
за нарушения margin (hinge loss):

$$
\min_{w,\, b} \; \|w\|^2 + C \sum_{i=1}^{n} \max\bigl(0,\; 1 - y_i(w \cdot x_i + b)\bigr)
$$

где:
- $w$ — вектор весов, $b$ — сдвиг (bias);
- $x_i \in \mathbb{R}^d$ — объект, $y_i \in \{-1, +1\}$ — метка класса;
- $C$ — коэффициент, насколько сильно штрафуем за нарушения margin
  (больше $C$ → меньше терпимости к ошибкам, меньше — шире margin);
- $\max(0,\; 1 - y_i(w\cdot x_i + b))$ — **hinge loss** одной точки.

### 1.2 Субградиент

Функция не дифференцируема в точке $y_i(w\cdot x_i+b)=1$ (излом hinge loss),
поэтому используется **субградиент**. Обозначим margin точки:

$$
m_i = y_i (w \cdot x_i + b)
$$

Тогда для каждой точки $(x_i, y_i)$:

$$
\frac{\partial L_i}{\partial w} =
\begin{cases}
2w, & m_i \geq 1 \\
2w - C\, y_i x_i, & m_i < 1
\end{cases}
\qquad
\frac{\partial L_i}{\partial b} =
\begin{cases}
0, & m_i \geq 1 \\
-C\, y_i, & m_i < 1
\end{cases}
$$

Смысл: если точка уже "правильно" за пределами margin ($m_i \ge 1$) — она
не штрафуется, градиент идёт только от регуляризации $\|w\|^2$. Если точка
внутри margin или неправильно классифицирована ($m_i < 1$) — добавляется
штрафующий член.

### 1.3 Обновление весов

$$
w \leftarrow w - \eta \, \nabla_w L_i
\qquad
b \leftarrow b - \eta \, \nabla_b L_i
$$

где $\eta$ — learning rate.

### 1.4 Псевдокод

```
вход: X (n×d), y (n, значения -1/+1), n_iter, learning_rate η, C, sgd_sample

w ← вектор единиц размерности d
b ← 1.0

для i = 1 .. n_iter:
    (X_batch, y_batch) ← выбрать батч из X, y
                          (весь датасет, если sgd_sample не задан;
                           случайные k строк, если задан)

    для каждой пары (xi, yi) из (X_batch, y_batch):
        margin ← yi * (xi · w + b)

        если margin >= 1:
            grad_w ← 2 * w
            grad_b ← 0
        иначе:
            grad_w ← 2 * w - C * yi * xi
            grad_b ← -C * yi

        w ← w - η * grad_w
        b ← b - η * grad_b

    (опционально) вычислить и залогировать loss на батче

вернуть w, b
```

### 1.5 Особенности

- Это **subgradient descent**, не обычный GD — из-за излома в hinge loss.
- Ядра (kernel trick) напрямую не подключить: чтобы использовать RBF/полиномиальное/
  сигмоидное ядро, нужно либо явно строить $\varphi(x)$ и подавать его вместо $x$,
  либо переходить к dual-форме.
- Хорошо масштабируется на большие $n$ (можно делать mini-batch / SGD).

---

## 2. Dual SVM (через двойственность, SMO)

### 2.1 От primal к dual через лагранжиан

Вводим множители Лагранжа $\alpha_i \ge 0$ для ограничений margin
($y_i(w\cdot x_i+b) \ge 1 - \xi_i$) и $\mu_i \ge 0$ для $\xi_i \ge 0$:

$$
\mathcal{L}(w,b,\xi,\alpha,\mu) = \|w\|^2 + C\sum_i \xi_i
- \sum_i \alpha_i\bigl[y_i(w\cdot x_i+b) - 1 + \xi_i\bigr] - \sum_i \mu_i \xi_i
$$

Из условий стационарности ($\partial \mathcal{L}/\partial w = 0$,
$\partial \mathcal{L}/\partial b = 0$, $\partial \mathcal{L}/\partial \xi_i = 0$)
получаем:

$$
w = \frac{1}{2}\sum_i \alpha_i y_i x_i,
\qquad
\sum_i \alpha_i y_i = 0,
\qquad
0 \le \alpha_i \le C
$$

Подставляя обратно, получаем **двойственную задачу**:

### 2.2 Формула двойственной задачи

$$
\max_{\alpha} \; \sum_{i=1}^n \alpha_i - \frac12 \sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j\, y_i y_j\, K(x_i, x_j)
$$

$$
\text{при ограничениях:} \quad 0 \le \alpha_i \le C \quad \text{и} \quad \sum_{i=1}^n \alpha_i y_i = 0
$$

где $K(x_i, x_j) = \varphi(x_i)\cdot\varphi(x_j)$ — **ядро** (linear, RBF,
polynomial, sigmoid — подставляется любое, отсюда и kernel trick).

После решения:

$$
w = \sum_i \alpha_i y_i x_i \quad \text{(для линейного случая; для ядер веса неявные)}
$$

$$
b = y_k - \sum_i \alpha_i y_i K(x_i, x_k) \quad \text{для любого опорного вектора } k \text{ с } 0 < \alpha_k < C
$$

Точки с $\alpha_i > 0$ — это **опорные векторы (support vectors)**, только
они влияют на итоговую границу.

### 2.3 Идея SMO (Sequential Minimal Optimization)

QP-задачу с $n$ переменными $\alpha_i$ и линейным ограничением
$\sum \alpha_i y_i = 0$ напрямую решать тяжело. SMO разбивает её на
множество маленьких подзадач: на каждом шаге берём **всего 2** множителя
$\alpha_i, \alpha_j$ (меньше нельзя — иначе нарушится ограничение
$\sum \alpha_i y_i = 0$), фиксируем остальные, и решаем задачу для этой
пары аналитически (без QP-солвера).

Ключевые формулы одного шага (для пары $i, j$):

$$
E_i = f(x_i) - y_i, \quad \text{где } f(x_i) = \sum_k \alpha_k y_k K(x_k, x_i) + b
$$

$$
\eta = 2K(x_i,x_j) - K(x_i,x_i) - K(x_j,x_j)
$$

$$
\alpha_j^{new} = \alpha_j^{old} - \frac{y_j(E_i - E_j)}{\eta}
$$

затем $\alpha_j^{new}$ **обрезается (clipping)** в границы $[L, H]$,
которые зависят от того, равны ли $y_i$ и $y_j$:

$$
\text{если } y_i \ne y_j:\quad L = \max(0,\, \alpha_j-\alpha_i),\ H = \min(C,\, C+\alpha_j-\alpha_i)
$$

$$
\text{если } y_i = y_j:\quad L = \max(0,\, \alpha_i+\alpha_j-C),\ H = \min(C,\, \alpha_i+\alpha_j)
$$

$$
\alpha_i^{new} = \alpha_i^{old} + y_i y_j(\alpha_j^{old} - \alpha_j^{new})
$$

### 2.4 Псевдокод

```
вход: X (n×d), y (n, значения -1/+1), C, ядро K, tol, max_passes

α ← вектор нулей размерности n
b ← 0
passes ← 0

пока passes < max_passes:
    num_changed ← 0

    для i = 1 .. n:
        E_i ← f(x_i) - y_i         # f(x_i) = Σ_k α_k y_k K(x_k, x_i) + b

        если (y_i*E_i < -tol и α_i < C) или (y_i*E_i > tol и α_i > 0):
            j ← выбрать случайный индекс, j ≠ i
            E_j ← f(x_j) - y_j

            α_i_old, α_j_old ← α_i, α_j

            вычислить границы L, H по y_i, y_j, α_i_old, α_j_old
            если L == H: перейти к следующему i

            η ← 2*K(x_i,x_j) - K(x_i,x_i) - K(x_j,x_j)
            если η >= 0: перейти к следующему i

            α_j ← α_j_old - y_j*(E_i - E_j) / η
            α_j ← clip(α_j, L, H)
            если |α_j - α_j_old| < 1e-5: перейти к следующему i

            α_i ← α_i_old + y_i*y_j*(α_j_old - α_j)

            b1 ← b - E_i - y_i*(α_i-α_i_old)*K(x_i,x_i) - y_j*(α_j-α_j_old)*K(x_i,x_j)
            b2 ← b - E_j - y_i*(α_i-α_i_old)*K(x_i,x_j) - y_j*(α_j-α_j_old)*K(x_j,x_j)

            если 0 < α_i < C:   b ← b1
            иначе если 0 < α_j < C: b ← b2
            иначе: b ← (b1+b2)/2

            num_changed ← num_changed + 1

    если num_changed == 0:
        passes ← passes + 1
    иначе:
        passes ← 0

вернуть α, b
```

### 2.5 Особенности

- В отличие от primal, здесь **можно подставлять ядра** $K(x_i,x_j)$
  вместо скалярного произведения — не нужно явно строить $\varphi(x)$.
- Итоговая модель хранит не $w$, а набор $(\alpha_i, x_i, y_i)$ для
  опорных векторов — предсказание идёт через сумму по ним:
  $\text{sign}\left(\sum_i \alpha_i y_i K(x_i, x) + b\right)$.
- Хуже масштабируется по числу объектов $n$ (нужна матрица $K$ размера
  $n \times n$), зато не требует, чтобы функция потерь была дифференцируема
  напрямую по $w$.

---

## 3. Сравнение

| | Primal (subgradient descent) | Dual (SMO) |
|---|---|---|
| Оптимизируемые переменные | $w \in \mathbb{R}^d$, $b$ | $\alpha \in \mathbb{R}^n$ |
| Метод | (стохастический) градиентный спуск | покоординатная оптимизация по парам $(\alpha_i,\alpha_j)$ |
| Ядра | нет (только через явный $\varphi(x)$) | да, напрямую через $K(x_i,x_j)$ |
| Итог модели | явные $w, b$ | опорные векторы + $\alpha_i$ + $b$ |
| Сложность на шаг | $O(d)$ на объект | $O(n)$ на пару (вычисление $E_i$) |
| Когда выгоднее | много признаков, мало объектов, линейное ядро | нужны ядра, объектов не миллионы |
