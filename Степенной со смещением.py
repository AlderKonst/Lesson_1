import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Установка шрифта Times New Roman для всех элементов
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.default'] = 'rm'  # 'rm' означает regular (прямой) шрифт

# Ваши экспериментальные данные
x = np.array(
    [1.85, 1.88, 1.88, 1.89, 2.25, 2.29, 2.29, 2.33, 2.43, 2.34, 2.52, 2.43, 1.77, 1.85, 1.82, 1.88, 2.24, 2.23, 2.23,
     2.22, 2.43, 2.37, 2.47, 2.38, 1.84, 1.86, 1.86, 1.87, 2.29, 2.22, 2.33, 2.29, 2.43, 2.4, 2.45, 2.54, 1.8, 1.9, 1.9,
     1.96, 2.41, 2.47, 2.3, 2.37, 2.49, 2.59, 2.41, 2.61])
y = np.array(
    [28.58, 31.23, 31.26, 32.02, 30.51, 32.75, 32.46, 34.32, 37.41626, 33.57125, 38.63887, 35.50919, 28.12, 30.8, 31.75,
     36.08, 34.13, 31.23, 32.17, 34.79, 37.56904, 32.59839, 36.50144, 39.66833, 31.58, 33.31, 35.06, 38.37, 28.79,
     33.59, 32.19, 36.88, 52.54, 41.54, 55.73, 54.37, 30.75, 32.45, 32.89, 35.23, 38.61, 43.38, 40.23, 42.62, 86.36,
     67.87, 61.54, 94.63])


# Определение степенной модели со смещением
def power_offset_model(x_val, a, b, c):
    return a * (x_val ** b) + c


# Функция для подбора параметров с ограничением a >= 0.0001
def fit_power_offset_model(x_data, y_data):
    # Начальные значения параметров (можно изменить при необходимости)
    initial_guess = [0.0001, 13.0, 28.0]

    # Ограничения: a >= 0.0001, b и c без ограничений
    bounds = ([0.0001, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

    try:
        # Подбор параметров методом наименьших квадратов
        popt, pcov = curve_fit(power_offset_model, x_data, y_data,
                               p0=initial_guess, bounds=bounds, maxfev=5000)

        # Извлекаем оптимальные параметры
        a_opt, b_opt, c_opt = popt

        # Обеспечиваем выполнение условия a >= 0.0001
        a_opt = max(a_opt, 0.0001)

        return a_opt, b_opt, c_opt

    except Exception as e:
        print(f"Ошибка при подборе параметров: {e}")
        # Возвращаем значения по умолчанию в случае ошибки
        return 0.0001, 13.59, 28.92


# Подбор оптимальных параметров
a, b, c = fit_power_offset_model(x, y)


# Определение функции с фиксированными параметрами для построения графика
def power_offset_model_fixed(x_val):
    return a * (x_val ** b) + c


# Расчет предсказанных значений и R²
y_pred = power_offset_model_fixed(x)
ss_res = np.sum((y - y_pred) ** 2)  # Сумма квадратов остатков
ss_tot = np.sum((y - np.mean(y)) ** 2)  # Общая сумма квадратов
r2 = 1 - (ss_res / ss_tot)

# Создание гладкой кривой для линии тренда
x_smooth = np.linspace(x.min() * 0.95, x.max() * 1.05, 500)
y_smooth = power_offset_model_fixed(x_smooth)

# Построение черно-белого графика
plt.figure(figsize=(10, 7))

# Экспериментальные точки (черные кружки)
plt.scatter(x, y, color='black', s=50, alpha=0.7, marker='o', edgecolors='black', linewidth=0.5)

# Линия тренда (черная сплошная линия)
plt.plot(x_smooth, y_smooth, 'k-', linewidth=2)

# Настройка осей
plt.xlabel('Содержание гумуса, %', fontsize=14)
plt.ylabel('Сбор обменной энергии', fontsize=14)

# Убираем сетку
plt.grid(False)

# Добавление уравнения и R² в середине графика
# Вычисляем координаты для размещения текста (примерно в центре области данных)
x_text = (x.min() + x.max()) / 2
y_text = (y.min() + y.max()) / 2

equation_text = f'$y = {a:.4f} \\cdot x^{{{b:.2f}}} + {c:.2f}$\n$R^2 = {r2:.2f}$'
plt.text(x_text, y_text, equation_text, fontsize=12,
         verticalalignment='center', horizontalalignment='center')

# Настройка границ графика
plt.xlim(x.min() * 0.95, x.max() * 1.05)
plt.ylim(y.min() * 0.95, y.max() * 1.05)

plt.tight_layout()
plt.show()

# Вывод информации в консоль
print("=" * 60)
print("СТЕПЕННАЯ МОДЕЛЬ С ОПТИМАЛЬНЫМИ ПАРАМЕТРАМИ")
print("=" * 60)
print(f"Параметры модели:")
print(f"  a = {a:.6f} (a ≥ 0.0001)")
print(f"  b = {b:.4f}")
print(f"  c = {c:.4f}")
print(f"Уравнение: y = {a:.6f} * x^{b:.4f} + {c:.4f}")
print(f"Коэффициент детерминации R² = {r2:.4f}")
print("=" * 60)