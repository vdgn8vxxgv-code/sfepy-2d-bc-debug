python
#!/usr/bin/env python3
# sfepy_bc_step2_convection.py
# ЭТАП 2: BC#2-3 (КОНВЕКЦИЯ НА ОТКОСАХ) — ИНТЕРАКТИВНАЯ ОТЛАДКА

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches

print("=" * 80)
print("ЭТАП 2: BC#2-3 (КОНВЕКЦИЯ НА ОТКОСАХ)")
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

# ИСПРАВЛЕННЫЕ маски
y_rel = (Y - y_interface) / (y_top - y_interface)
x_left_interp = x_bl + (x_tl - x_bl) * y_rel
x_right_interp = x_br + (x_tr - x_br) * y_rel

mask_embankment = (Y >= y_interface) & (X >= x_left_interp) & (X <= x_right_interp)
mask_soil = Y < y_interface

print(f"  Сетка: {nx} x {ny} = {nx * ny} узлов")
print(f"  Область насыпи: {mask_embankment.sum()} узлов")
print(f"  Область грунта: {mask_soil.sum()} узлов")

# ============================================================================
# 2. ОПРЕДЕЛЕНИЕ ОТКОСОВ (границы между насыпью и воздухом)
# ============================================================================

print("\n[2] Определение откосов (граничные узлы BC#2-3)...")

# Левый откос: y_interface < y < y_top, x ≈ x_left_interp(y)
# Правый откос: y_interface < y < y_top, x ≈ x_right_interp(y)

left_slope_nodes = []
right_slope_nodes = []

tolerance = 0.5  # Допуск для обнаружения границы

for j in range(ny):
    for i in range(nx):
        idx = j * nx + i
        
        if y[j] >= y_interface and y[j] <= y_top:
            y_rel_j = (y[j] - y_interface) / (y_top - y_interface)
            x_left_target = x_bl + (x_tl - x_bl) * y_rel_j
            x_right_target = x_br + (x_tr - x_br) * y_rel_j
            
            # Левый откос
            if abs(x[i] - x_left_target) < tolerance and mask_embankment[j, i]:
                if idx not in left_slope_nodes:
                    left_slope_nodes.append(idx)
            
            # Правый откос
            if abs(x[i] - x_right_target) < tolerance and mask_embankment[j, i]:
                if idx not in right_slope_nodes:
                    right_slope_nodes.append(idx)

left_slope_nodes = np.array(left_slope_nodes)
right_slope_nodes = np.array(right_slope_nodes)

print(f"  Узлов на левом откосе: {len(left_slope_nodes)}")
print(f"  Узлов на правом откосе: {len(right_slope_nodes)}")

# ============================================================================
# 3. ФУНКЦИЯ ДЛЯ СБОРКИ И РЕШЕНИЯ СИСТЕМЫ
# ============================================================================

def solve_heat_conduction(bc_config):
    """
    Решить задачу теплопроводности с заданными BC.
    
    bc_config: dict с параметрами BC
        - 'bc1_enabled': bool (Дирихле на дне)
        - 'bc1_value': float (значение T на дне)
        - 'bc2_enabled': bool (конвекция на левом откосе)
        - 'bc3_enabled': bool (конвекция на правом откосе)
        - 'h': float (коэффициент теплоотдачи, Вт/м²К)
        - 'T_ext': float (внешняя температура, °С)
    """
    
    # Координаты узлов
    coors_x = X.flatten()
    coors_y = Y.flatten()
    N = len(coors_x)
    
    # Шаги сетки
    dx = Lx / (nx - 1)
    dy = y_top / (ny - 1)
    
    # Параметры материалов
    k_embankment = 0.5
    k_soil = 1.5
    
    # Сборка матрицы K
    K_data = []
    K_row = []
    K_col = []
    F = np.zeros(N)
    
    coeff_x = 1.0 / (dx ** 2)
    coeff_y = 1.0 / (dy ** 2)
    
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            
            if mask_embankment[j, i]:
                k = k_embankment
            else:
                k = k_soil
            
            k_coeff_x = k * coeff_x
            k_coeff_y = k * coeff_y
            
            diag_coeff = -2.0 * (k_coeff_x + k_coeff_y)
            
            # Внутренние узлы
            if 0 < i < nx - 1 and 0 < j < ny - 1:
                K_data.extend([k_coeff_x, diag_coeff, k_coeff_x])
                K_col.extend([idx - 1, idx, idx + 1])
                K_row.extend([idx, idx, idx])
                
                K_data.extend([k_coeff_y, k_coeff_y])
                K_col.extend([idx - nx, idx + nx])
                K_row.extend([idx, idx])
                
                F[idx] = 0.0
            else:
                K_data.append(1.0)
                K_col.append(idx)
                K_row.append(idx)
                F[idx] = 0.0
    
    K = csr_matrix((K_data, (K_row, K_col)), shape=(N, N))
    
    # ---- BC#1: Дирихле на дне ----
    if bc_config.get('bc1_enabled', True):
        T_bc = bc_config.get('bc1_value', 2.0)
        bc_nodes_bottom = np.where(coors_y == y_bottom)[0]
        
        for idx in bc_nodes_bottom:
            K[idx, idx] = 1.0
            F[idx] = T_bc
    
    # ---- BC#2: Конвекция на левом откосе (Робин) ----
    if bc_config.get('bc2_enabled', False):
        h = bc_config.get('h', 50.0)
        T_ext = bc_config.get('T_ext', 0.0)
        
        for idx in left_slope_nodes:
            # Robin BC: -k dT/dn = h(T - T_ext)
            # Дискретно: K[idx, idx] += h, F[idx] += h*T_ext
            K[idx, idx] += h
            F[idx] += h * T_ext
    
    # ---- BC#3: Конвекция на правом откосе (Робин) ----
    if bc_config.get('bc3_enabled', False):
        h = bc_config.get('h', 50.0)
        T_ext = bc_config.get('T_ext', 0.0)
        
        for idx in right_slope_nodes:
            K[idx, idx] += h
            F[idx] += h * T_ext
    
    # Решение
    try:
        T = spsolve(K.tocsc(), F)
    except Exception as e:
        print(f"  ✗ Ошибка при решении: {e}")
        T = np.ones(N) * 10.0
    
    return T

# ============================================================================
# 4. РЕШЕНИЕ БЕЗ И С BC#2-3
# ============================================================================

print("\n[3] Решение: BC#1 только (исходное)...")

bc_config_only_bc1 = {
    'bc1_enabled': True,
    'bc1_value': 2.0,
    'bc2_enabled': False,
    'bc3_enabled': False,
}

T_without_convection = solve_heat_conduction(bc_config_only_bc1)
print(f"  T_min = {T_without_convection.min():.3f}°C")
print(f"  T_max = {T_without_convection.max():.3f}°C")
print(f"  T_mean = {T_without_convection.mean():.3f}°C")

print("\n[4] Решение: BC#1 + BC#2-3 (с конвекцией на откосах)...")

bc_config_with_convection = {
    'bc1_enabled': True,
    'bc1_value': 2.0,
    'bc2_enabled': True,
    'bc3_enabled': True,
    'h': 50.0,        # Вт/(м²·К)
    'T_ext': 0.0,     # °С
}

T_with_convection = solve_heat_conduction(bc_config_with_convection)
print(f"  T_min = {T_with_convection.min():.3f}°C")
print(f"  T_max = {T_with_convection.max():.3f}°C")
print(f"  T_mean = {T_with_convection.mean():.3f}°C")

# ============================================================================
# 5. АНАЛИЗ РАЗНИЦЫ
# ============================================================================

print("\n[5] Анализ влияния BC#2-3...")

T_diff = T_without_convection - T_with_convection
print(f"  ΔT = T_без_конв - T_с_конв:")
print(f"    ΔT_min = {T_diff.min():.3f}°C (максимальное охлаждение)")
print(f"    ΔT_max = {T_diff.max():.3f}°C")
print(f"    ΔT_mean = {T_diff.mean():.3f}°C")

# Анализ в откосах
print(f"\n  Анализ в откосах:")
print(f"    Левый откос:")
T_left_without = T_without_convection[left_slope_nodes]
T_left_with = T_with_convection[left_slope_nodes]
print(f"      Без конвекции: T = {T_left_without.mean():.3f}°C")
print(f"      С конвекцией: T = {T_left_with.mean():.3f}°C")
print(f"      Охлаждение: {(T_left_without - T_left_with).mean():.3f}°C")

print(f"    Правый откос:")
T_right_without = T_without_convection[right_slope_nodes]
T_right_with = T_with_convection[right_slope_nodes]
print(f"      Без конвекции: T = {T_right_without.mean():.3f}°C")
print(f"      С конвекцией: T = {T_right_with.mean():.3f}°C")
print(f"      Охлаждение: {(T_right_without - T_right_with).mean():.3f}°C")

# ============================================================================
# 6. ВИЗУАЛИЗАЦИЯ
# ============================================================================

print("\n[6] Создание визуализации...")

fig = plt.figure(figsize=(18, 14))

T_without_reshaped = T_without_convection.reshape(ny, nx)
T_with_reshaped = T_with_convection.reshape(ny, nx)
T_diff_reshaped = T_diff.reshape(ny, nx)

embankment_points = np.array([
    [x_bl, y_interface],
    [x_tl, y_top],
    [x_tr, y_top],
    [x_br, y_interface]
])

# ---- Подграфик 1: Геометрия с откосами ----
ax1 = plt.subplot(3, 3, 1)
coors_x = X.flatten()
coors_y = Y.flatten()
ax1.scatter(coors_x, coors_y, s=5, c='lightgray', alpha=0.5, label='Все узлы')
ax1.scatter(coors_x[left_slope_nodes], coors_y[left_slope_nodes], s=20, c='blue', 
            marker='s', label='BC#2 (левый откос)', zorder=5)
ax1.scatter(coors_x[right_slope_nodes], coors_y[right_slope_nodes], s=20, c='cyan', 
            marker='s', label='BC#3 (правый откос)', zorder=5)
embankment_poly = Polygon(embankment_points, fill=False, edgecolor='red', linewidth=2, label='Границы насыпи')
ax1.add_patch(embankment_poly)
ax1.set_xlim(-1, Lx + 1)
ax1.set_ylim(-1, y_top + 1)
ax1.set_xlabel('x (м)')
ax1.set_ylabel('y (м)')
ax1.set_title('Геометрия и BC#2-3 узлы')
ax1.legend(fontsize=7, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# ---- Подграфик 2: Без конвекции ----
ax2 = plt.subplot(3, 3, 2)
levels = np.linspace(T_without_convection.min(), T_without_convection.max(), 20)
contourf2 = ax2.contourf(X, Y, T_without_reshaped, levels=levels, cmap='coolwarm')
ax2.contour(X, Y, T_without_reshaped, levels=10, colors='black', alpha=0.2, linewidths=0.5)
embankment_poly2 = Polygon(embankment_points, fill=False, edgecolor='black', linewidth=2)
ax2.add_patch(embankment_poly2)
cbar2 = plt.colorbar(contourf2, ax=ax2)
cbar2.set_label('T (°C)')
ax2.set_xlim(0, Lx)
ax2.set_ylim(y_bottom, y_top)
ax2.set_xlabel('x (м)')
ax2.set_ylabel('y (м)')
ax2.set_title('Без BC#2-3 (только BC#1)')
ax2.set_aspect('equal')

# ---- Подграфик 3: С конвекцией ----
ax3 = plt.subplot(3, 3, 3)
levels3 = np.linspace(T_with_convection.min(), T_with_convection.max(), 20)
contourf3 = ax3.contourf(X, Y, T_with_reshaped, levels=levels3, cmap='coolwarm')
ax3.contour(X, Y, T_with_reshaped, levels=10, colors='black', alpha=0.2, linewidths=0.5)
embankment_poly3 = Polygon(embankment_points, fill=False, edgecolor='black', linewidth=2)
ax3.add_patch(embankment_poly3)
cbar3 = plt.colorbar(contourf3, ax=ax3)
cbar3.set_label('T (°C)')
ax3.set_xlim(0, Lx)
ax3.set_ylim(y_bottom, y_top)
ax3.set_xlabel('x (м)')
ax3.set_ylabel('y (м)')
ax3.set_title('С BC#2-3 (конвекция откосы)')
ax3.set_aspect('equal')

# ---- Подграфик 4: Разница (ΔT) ----
ax4 = plt.subplot(3, 3, 4)
levels4 = np.linspace(T_diff.min(), T_diff.max(), 20)
contourf4 = ax4.contourf(X, Y, T_diff_reshaped, levels=levels4, cmap='RdBu_r')
embankment_poly4 = Polygon(embankment_points, fill=False, edgecolor='black', linewidth=2)
ax4.add_patch(embankment_poly4)
cbar4 = plt.colorbar(contourf4, ax=ax4)
cbar4.set_label('ΔT (°C)')
ax4.set_xlim(0, Lx)
ax4.set_ylim(y_bottom, y_top)
ax4.set_xlabel('x (м)')
ax4.set_ylabel('y (м)')
ax4.set_title('ΔT = T_без - T_с (охлаждение)')
ax4.set_aspect('equal')

# ---- Подграфик 5: Вертикальные профили ----
ax5 = plt.subplot(3, 3, 5)
x_center_idx = nx // 2
T_vertical_without = T_without_reshaped[:, x_center_idx]
T_vertical_with = T_with_reshaped[:, x_center_idx]
ax5.plot(T_vertical_without, y, 'b-o', linewidth=2, label='Без BC#2-3', markersize=4)
ax5.plot(T_vertical_with, y, 'r-s', linewidth=2, label='С BC#2-3', markersize=4)
ax5.axhline(y=y_interface, color='gray', linestyle='--', alpha=0.5, label='Интерфейс')
ax5.set_xlabel('T (°C)')
ax5.set_ylabel('y (м)')
ax5.set_title(f'Вертикальный профиль (x={x[x_center_idx]:.2f})')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=8)

# ---- Подграфик 6: Горизонтальные профили ----
ax6 = plt.subplot(3, 3, 6)
y_mid_idx = (ny - 1) // 2
T_horizontal_without = T_without_reshaped[y_mid_idx, :]
T_horizontal_with = T_with_reshaped[y_mid_idx, :]
ax6.plot(x, T_horizontal_without, 'b-o', linewidth=2, label='Без BC#2-3', markersize=4)
ax6.plot(x, T_horizontal_with, 'r-s', linewidth=2, label='С BC#2-3', markersize=4)
ax6.axvline(x=x_bl, color='gray', linestyle='--', alpha=0.5, label='Границы насыпи')
ax6.axvline(x=x_br, color='gray', linestyle='--', alpha=0.5)
ax6.set_xlabel('x (м)')
ax6.set_ylabel('T (°C)')
ax6.set_title(f'Горизонтальный профиль (y={y[y_mid_idx]:.2f})')
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=8)

# ---- Подграфик 7: Охлаждение в откосах (вертикальный) ----
ax7 = plt.subplot(3, 3, 7)
if len(left_slope_nodes) > 0 and len(right_slope_nodes) > 0:
    y_left = coors_y[left_slope_nodes]
    y_right = coors_y[right_slope_nodes]
    T_cooling_left = T_without_convection[left_slope_nodes] - T_with_convection[left_slope_nodes]
    T_cooling_right = T_without_convection[right_slope_nodes] - T_with_convection[right_slope_nodes]
    
    ax7.plot(T_cooling_left, y_left, 'b-o', linewidth=2, label='Левый откос', markersize=5)
    ax7.plot(T_cooling_right, y_right, 'r-s', linewidth=2, label='Правый откос', markersize=5)
    ax7.set_xlabel('Охлаждение ΔT (°C)')
    ax7.set_ylabel('y (м)')
    ax7.set_title('Охлаждение вдоль откосов')
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=8)
else:
    ax7.text(0.5, 0.5, 'Откосы не найдены', ha='center', va='center', transform=ax7.transAxes)

# ---- Подграфик 8: Гистограмма температур ----
ax8 = plt.subplot(3, 3, 8)
ax8.hist(T_without_convection, bins=30, alpha=0.5, label='Без BC#2-3', color='blue')
ax8.hist(T_with_convection, bins=30, alpha=0.5, label='С BC#2-3', color='red')
ax8.set_xlabel('T (°C)')
ax8.set_ylabel('Частота')
ax8.set_title('Распределение температуры')
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3, axis='y')

# ---- Подграфик 9: Статистика ----
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

stats_text = f"""
СТАТИСТИКА BC#2-3 (КОНВЕКЦИЯ)

Параметры BC#2-3:
  h = {bc_config_with_convection['h']} Вт/(м²·К)
  T_ext = {bc_config_with_convection['T_ext']}°C

Без BC#2-3 (только BC#1):
  T_min = {T_without_convection.min():.3f}°C
  T_max = {T_without_convection.max():.3f}°C
  T_mean = {T_without_convection.mean():.3f}°C

С BC#2-3:
  T_min = {T_with_convection.min():.3f}°C
  T_max = {T_with_convection.max():.3f}°C
  T_mean = {T_with_convection.mean():.3f}°C

Разница (ΔT = без - с):
  ΔT_mean = {T_diff.mean():.3f}°C
  ΔT_max = {T_diff.max():.3f}°C
  ΔT_min = {T_diff.min():.3f}°C

Откосы:
  Узлов слева: {len(left_slope_nodes)}
  Узлов справа: {len(right_slope_nodes)}
  Охлаждение слева: {(T_left_without - T_left_with).mean():.3f}°C
  Охлаждение справа: {(T_right_without - T_right_with).mean():.3f}°C

Вывод:
  Конвекция BC#2-3 охлаждает
  откосы на {T_diff.mean():.1f}°C в среднем
"""

ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=8,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('bc2_bc3_convection_debug.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Сохранено: bc2_bc3_convection_debug.png")

plt.show()

print("\n" + "=" * 80)
print("ЭТАП 2 ЗАВЕРШЁН")
print("=" * 80)
print("\nЧто дальше?")
print("  1. Откройте bc2_bc3_convection_debug.png")
print("  2. Сравните графики: как откосы охлаждаются конвекцией")
print("  3. Запустите sfepy_bc_step3_top_surface.py (BC#4)")
print("\n")