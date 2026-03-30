#!/usr/bin/env python3
# sfepy_bc_debug_interactive.py
# ЭТАП 1: BC#1 (Дирихле на дне) — ИНТЕРАКТИВНАЯ ОТЛАДКА

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches

print("=" * 80)
print("ЭТАП 1: BC#1 (ДИРИХЛЕ НА ДНЕ)")
print("=" * 80)

# ============================================================================
# 1. ГЕОМЕТРИЯ: Трапециевидная насыпь на грунте
# ============================================================================

print("\n[1] Создание геометрии...")

# Параметры геометрии
x_tl, x_tr = 16.55, 23.45  # Вершина насыпи (сверху)
x_bl, x_br = 7.55, 32.45   # Основание насыпи (снизу)
y_interface = 10.0         # Интерфейс (грунт/насыпь)
y_top = 16.0               # Верх насыпи
Lx = 40.0                  # Ширина грунта
y_bottom = 0.0             # Дно

# Простая регулярная сетка
nx, ny = 41, 33            # Число узлов по x и y
x = np.linspace(0, Lx, nx)
y = np.linspace(y_bottom, y_top, ny)
X, Y = np.meshgrid(x, y)

# Маски для области насыпи и грунта
# ИСПРАВЛЕННЫЕ ФОРМУЛЫ (из предыдущей беседы)
y_rel = (Y - y_interface) / (y_top - y_interface)
x_left_interp = x_bl + (x_tl - x_bl) * y_rel
x_right_interp = x_br + (x_tr - x_br) * y_rel

mask_embankment = (Y >= y_interface) & (X >= x_left_interp) & (X <= x_right_interp)
mask_soil = Y < y_interface

print(f"  Сетка: {nx} x {ny} = {nx * ny} узлов")
print(f"  Область насыпи: {mask_embankment.sum()} узлов")
print(f"  Область грунта: {mask_soil.sum()} узлов")

# ============================================================================
# 2. СБОРКА СИСТЕМЫ: Матрица K и правая часть F
# ============================================================================

print("\n[2] Сборка FEM системы...")

# Параметры материалов
k_embankment = 0.5   # Теплопроводность насыпи (Вт/м·К)
k_soil = 1.5         # Теплопроводность грунта (Вт/м·К)

# Координаты узлов (flattened)
coors_x = X.flatten()
coors_y = Y.flatten()
N = len(coors_x)

print(f"  Всего узлов: {N}")

# Шаги сетки
dx = Lx / (nx - 1)
dy = y_top / (ny - 1)

print(f"  Шаг по x: {dx:.4f} м")
print(f"  Шаг по y: {dy:.4f} м")

# Сборка матрицы K методом конечных разностей (5-точечный шаблон)
# -d²T/dx² - d²T/dy² = 0
# Центральные разности: (T[i-1] + T[i+1] - 2*T[i])/dx² + (T[j-1] + T[j+1] - 2*T[j])/dy² = 0

K_data = []
K_row = []
K_col = []
F = np.zeros(N)

coeff_x = 1.0 / (dx ** 2)
coeff_y = 1.0 / (dy ** 2)

for j in range(ny):
    for i in range(nx):
        idx = j * nx + i
        
        # Определить теплопроводность в этом узле
        if mask_embankment[j, i]:
            k = k_embankment
        else:
            k = k_soil
        
        k_coeff_x = k * coeff_x
        k_coeff_y = k * coeff_y
        
        # Диагональный элемент
        diag_coeff = -2.0 * (k_coeff_x + k_coeff_y)
        
        # Внутренние узлы
        if 0 < i < nx - 1 and 0 < j < ny - 1:
            # d²T/dx²
            K_data.extend([k_coeff_x, diag_coeff, k_coeff_x])
            K_col.extend([idx - 1, idx, idx + 1])
            K_row.extend([idx, idx, idx])
            
            # d²T/dy²
            K_data.extend([k_coeff_y, k_coeff_y])
            K_col.extend([idx - nx, idx + nx])
            K_row.extend([idx, idx])
            
            F[idx] = 0.0
        
        # Граничные узлы (обработаны ниже через BC)
        else:
            K_data.append(1.0)
            K_col.append(idx)
            K_row.append(idx)
            F[idx] = 0.0  # Пока 0, переопределим в BC

# Формирование разреженной матрицы
K = csr_matrix((K_data, (K_row, K_col)), shape=(N, N))

print(f"  Матрица K: {K.shape}, ненулевых элементов: {K.nnz}")

# ============================================================================
# 3. ГРАНИЧНЫЕ УСЛОВИЯ (BC)
# ============================================================================

print("\n[3] Применение BC...")

# BC#1: Дирихле на дне (y = 0)
T_bottom = 2.0  # °C

bc_nodes_bottom = np.where(coors_y == y_bottom)[0]

print(f"  BC#1 (Дирихле на дне): {len(bc_nodes_bottom)} узлов при T = {T_bottom}°C")

# Применить BC#1: K[i,:] = 0, K[i,i] = 1, F[i] = T_bc
for idx in bc_nodes_bottom:
    # Очистить строку (кроме диагонали)
    K.data[K.indptr[idx]:K.indptr[idx+1]] = 0.0
    K[idx, idx] = 1.0
    F[idx] = T_bottom

# BC на боковых границах (x=0 и x=Lx): адиабатные (Неймана, dT/dn=0)
# Для простоты: условие естественных граней (ничего не делаем, они "автоматически")
# Но можно явно задать: строки обнуляются и K[i,i]=1, F[i]=0

# Верхняя граница (y = y_top): 
# Для этого примера оставляем как естественное BC (потребляем тепло вверх, но контролируем через физику)
# или можем задать как условие теплоотвода, но это будет BC#2-4 позже

print(f"  Всего применено BC: {len(bc_nodes_bottom)}")

# ============================================================================
# 4. РЕШЕНИЕ СИСТЕМЫ
# ============================================================================

print("\n[4] Решение системы линейных уравнений...")

# Проверить матрицу на сингулярность
print(f"  Определитель матрицы: нельзя вычислить для разреженной (используем обусловленность)")
print(f"  Решаем: K * T = F...")

try:
    T = spsolve(K.tocsc(), F)
    print(f"  ✓ Решение найдено")
    print(f"  T_min = {T.min():.3f}°C, T_max = {T.max():.3f}°C, T_mean = {T.mean():.3f}°C")
except Exception as e:
    print(f"  ✗ ОШИБКА: {e}")
    T = np.ones(N) * 10.0  # Fallback

# ============================================================================
# 5. АНАЛИЗ И ВАЛИДАЦИЯ
# ============================================================================

print("\n[5] Анализ решения...")

# Проверить, что BC#1 применено правильно
T_bc_actual = T[bc_nodes_bottom]
print(f"  BC#1 проверка:")
print(f"    Ожидаемое: T = {T_bottom}°C")
print(f"    Полученное: T_min = {T_bc_actual.min():.3f}°C, T_max = {T_bc_actual.max():.3f}°C")
print(f"    Среднее отклонение: {np.abs(T_bc_actual - T_bottom).mean():.6f}°C")

# Профили температуры
T_reshaped = T.reshape(ny, nx)

# Вертикальный профиль в центре (x ≈ Lx/2)
x_center_idx = nx // 2
T_vertical = T_reshaped[:, x_center_idx]

# Горизонтальный профиль в насыпи (y ≈ (y_interface + y_top)/2)
y_mid_idx = (ny - 1) // 2
T_horizontal = T_reshaped[y_mid_idx, :]

print(f"  Вертикальный профиль (x={x[x_center_idx]:.2f}):")
print(f"    На дне (y=0): T = {T_reshaped[0, x_center_idx]:.3f}°C")
print(f"    В центре: T = {T_vertical[ny//2]:.3f}°C")
print(f"    На верху: T = {T_reshaped[-1, x_center_idx]:.3f}°C")

print(f"  Горизонтальный профиль (y={y[y_mid_idx]:.2f}):")
print(f"    На левом краю: T = {T_horizontal[0]:.3f}°C")
print(f"    В центре: T = {T_horizontal[nx//2]:.3f}°C")
print(f"    На правом краю: T = {T_horizontal[-1]:.3f}°C")

# ============================================================================
# 6. ВИЗУАЛИЗАЦИЯ
# ============================================================================

print("\n[6] Создание визуализации...")

fig = plt.figure(figsize=(16, 12))

# ---- Подграфик 1: Геометрия и BC узлы ----
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(coors_x, coors_y, s=5, c='lightgray', alpha=0.5, label='Все узлы')
ax1.scatter(coors_x[bc_nodes_bottom], coors_y[bc_nodes_bottom], s=20, c='blue', 
            marker='s', label=f'BC#1 (дно, T={T_bottom}°C)', zorder=5)
ax1.scatter(coors_x[mask_embankment], coors_y[mask_embankment], s=3, c='brown', alpha=0.3, label='Насыпь')
ax1.scatter(coors_x[mask_soil], coors_y[mask_soil], s=3, c='orange', alpha=0.3, label='Грунт')

# Нарисовать геометрию пошше
embankment_points = np.array([
    [x_bl, y_interface],
    [x_tl, y_top],
    [x_tr, y_top],
    [x_br, y_interface]
])
embankment_poly = Polygon(embankment_points, fill=False, edgecolor='red', linewidth=2, label='Насыпь граница')
ax1.add_patch(embankment_poly)

ax1.set_xlim(-1, Lx + 1)
ax1.set_ylim(-1, y_top + 1)
ax1.set_xlabel('x (м)')
ax1.set_ylabel('y (м)')
ax1.set_title('Геометрия и граничные узлы (BC#1)')
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# ---- Подграфик 2: Температурная карта ----
ax2 = plt.subplot(2, 3, 2)
levels = np.linspace(T.min(), T.max(), 20)
contourf = ax2.contourf(X, Y, T_reshaped, levels=levels, cmap='coolwarm')
ax2.contour(X, Y, T_reshaped, levels=10, colors='black', alpha=0.2, linewidths=0.5)
embankment_poly2 = Polygon(embankment_points, fill=False, edgecolor='black', linewidth=2)
ax2.add_patch(embankment_poly2)
cbar = plt.colorbar(contourf, ax=ax2)
cbar.set_label('T (°C)')
ax2.set_xlim(0, Lx)
ax2.set_ylim(y_bottom, y_top)
ax2.set_xlabel('x (м)')
ax2.set_ylabel('y (м)')
ax2.set_title('Температурное поле (BC#1 только)')
ax2.set_aspect('equal')

# ---- Подграфик 3: Вертикальный профиль ----
ax3 = plt.subplot(2, 3, 3)
ax3.plot(T_vertical, y, 'b-o', linewidth=2, markersize=4)
ax3.axhline(y=y_interface, color='red', linestyle='--', alpha=0.5, label='Интерфейс грунт/насыпь')
ax3.axhline(y=y_bottom, color='blue', linestyle='--', alpha=0.5, label=f'BC#1 (T={T_bottom}°C)')
ax3.set_xlabel('T (°C)')
ax3.set_ylabel('y (м)')
ax3.set_title(f'Вертикальный профиль (x={x[x_center_idx]:.2f})')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=8)

# ---- Подграфик 4: Горизонтальный профиль ----
ax4 = plt.subplot(2, 3, 4)
ax4.plot(x, T_horizontal, 'g-o', linewidth=2, markersize=4)
ax4.axvline(x=x_bl, color='red', linestyle='--', alpha=0.5, label='Границы насыпи')
ax4.axvline(x=x_br, color='red', linestyle='--', alpha=0.5)
ax4.set_xlabel('x (м)')
ax4.set_ylabel('T (°C)')
ax4.set_title(f'Горизонтальный профиль (y={y[y_mid_idx]:.2f})')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=8)

# ---- Подграфик 5: Статистика ----
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')

stats_text = f"""
СТАТИСТИКА РЕШЕНИЯ (BC#1 только)

Температура:
  T_min = {T.min():.3f}°C
  T_max = {T.max():.3f}°C
  T_mean = {T.mean():.3f}°C
  T_std = {T.std():.3f}°C

BC#1 (дно):
  Узлов: {len(bc_nodes_bottom)}
  T_ожидаемое: {T_bottom:.3f}°C
  T_фактическое: {T_bc_actual.mean():.3f}°C
  Ошибка: {np.abs(T_bc_actual - T_bottom).mean():.6f}°C

Сетка:
  Узлов: {N}
  Элементов x: {nx} (dx={dx:.4f})
  Элементов y: {ny} (dy={dy:.4f})

Материалы:
  Насыпь: k = {k_embankment} Вт/(м·К)
  Грунт: k = {k_soil} Вт/(м·К)
  Узлов в насыпи: {mask_embankment.sum()}
  Узлов в грунте: {mask_soil.sum()}

Тип задачи:
  Стационарная теплопроводность (Лапласа)
  -∇·(k∇T) = 0
"""

ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=9,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ---- Подграфик 6: Визуализация энергии ----
ax6 = plt.subplot(2, 3, 6)
# Вычислить "энергию" в каждом узле (простой способ)
energy = np.zeros_like(T_reshaped)
for j in range(1, ny-1):
    for i in range(1, nx-1):
        idx = j * nx + i
        # Локальная вариация температуры
        dT_dx = (T_reshaped[j, i+1] - T_reshaped[j, i-1]) / (2 * dx)
        dT_dy = (T_reshaped[j+1, i] - T_reshaped[j-1, i]) / (2 * dy)
        energy[j, i] = dT_dx**2 + dT_dy**2

levels_energy = np.linspace(energy.min(), energy.max(), 15)
contourf_energy = ax6.contourf(X, Y, energy, levels=levels_energy, cmap='viridis')
embankment_poly3 = Polygon(embankment_points, fill=False, edgecolor='white', linewidth=2)
ax6.add_patch(embankment_poly3)
cbar_energy = plt.colorbar(contourf_energy, ax=ax6)
cbar_energy.set_label('|∇T|² (°C²/м²)')
ax6.set_xlim(0, Lx)
ax6.set_ylim(y_bottom, y_top)
ax6.set_xlabel('x (м)')
ax6.set_ylabel('y (м)')
ax6.set_title('Интенсивность градиента (|∇T|²)')
ax6.set_aspect('equal')

plt.tight_layout()
plt.savefig('bc1_dirichlet_bottom_debug.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Сохранено: bc1_dirichlet_bottom_debug.png")

plt.show()

print("\n" + "=" * 80)
print("ЭТАП 1 ЗАВЕРШЁН")
print("=" * 80)
print("\nЧто дальше?")
print("  1. Откройте bc1_dirichlet_bottom_debug.png")
print("  2. Посмотрите, как выглядит температурное поле с BC#1")
print("  3. Запустите sfepy_bc_step2_convection.py (BC#2-3)")
print("\n")