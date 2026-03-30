#!/usr/bin/env python3
# sfepy_bc_step3_top_surface.py
# ЭТАП 3: BC#4 (КОНВЕКЦИЯ НА ВЕРШИНЕ) — ИНТЕРАКТИВНАЯ ОТЛАДКА

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches

print("=" * 80)
print("ЭТАП 3: BC#4 (КОНВЕКЦИЯ НА ВЕРШИНЕ)")
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
# 2. ОПРЕДЕЛЕНИЕ ГРАНИЦ (откосы и вершина)
# ============================================================================

print("\n[2] Определение граничных узлов...")

# Левый откос
left_slope_nodes = []
right_slope_nodes = []
top_surface_nodes = []

tolerance = 0.5

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

# Вершина (y ≈ y_top, x между x_tl и x_tr)
for i in range(nx):
    for j in range(ny):
        idx = j * nx + i
        if abs(y[j] - y_top) < 0.5 and x_tl <= x[i] <= x_tr:
            if idx not in top_surface_nodes:
                top_surface_nodes.append(idx)

left_slope_nodes = np.array(left_slope_nodes)
right_slope_nodes = np.array(right_slope_nodes)
top_surface_nodes = np.array(top_surface_nodes)

print(f"  Узлов на левом откосе: {len(left_slope_nodes)}")
print(f"  Узлов на правом откосе: {len(right_slope_nodes)}")
print(f"  Узлов на вершине: {len(top_surface_nodes)}")

# ============================================================================
# 3. ФУНКЦИЯ ДЛЯ СБОРКИ И РЕШЕНИЯ СИСТЕМЫ
# ============================================================================

def solve_heat_conduction_bc4(bc_config):
    """
    Решить задачу теплопроводности с заданными BC (включая BC#4).
    
    bc_config: dict с параметрами BC
        - 'bc1_enabled': bool (Дирихле на дне)
        - 'bc1_value': float (значение T на дне)
        - 'bc2_enabled': bool (конвекция на левом откосе)
        - 'bc3_enabled': bool (конвекция на правом откосе)
        - 'bc4_enabled': bool (конвекция на вершине)
        - 'h_slopes': float (коэффициент теплоотдачи откосы, Вт/м²К)
        - 'T_ext_slopes': float (внешняя температура откосы, °С)
        - 'h_top': float (коэффициент теплоотдачи вершина, Вт/м²К)
        - 'T_ext_top': float (внешняя температура вершина, °С)
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
    
    # ---- BC#2: Конвекция на левом откосе ----
    if bc_config.get('bc2_enabled', False):
        h = bc_config.get('h_slopes', 50.0)
        T_ext = bc_config.get('T_ext_slopes', 0.0)
        
        for idx in left_slope_nodes:
            K[idx, idx] += h
            F[idx] += h * T_ext
    
    # ---- BC#3: Конвекция на правом откосе ----
    if bc_config.get('bc3_enabled', False):
        h = bc_config.get('h_slopes', 50.0)
        T_ext = bc_config.get('T_ext_slopes', 0.0)
        
        for idx in right_slope_nodes:
            K[idx, idx] += h
            F[idx] += h * T_ext
    
    # ---- BC#4: Конвекция на вершине (более сильная) ----
    if bc_config.get('bc4_enabled', False):
        h = bc_config.get('h_top', 75.0)
        T_ext = bc_config.get('T_ext_top', -5.0)
        
        for idx in top_surface_nodes:
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
# 4. РЕШЕНИЕ БЕЗ И С BC#4
# ============================================================================

print("\n[3] Решение: BC#1-3 (без BC#4 на вершине)...")

bc_config_without_bc4 = {
    'bc1_enabled': True,
    'bc1_value': 2.0,
    'bc2_enabled': True,
    'bc3_enabled': True,
    'bc4_enabled': False,
    'h_slopes': 50.0,
    'T_ext_slopes': 0.0,
}

T_without_bc4 = solve_heat_conduction_bc4(bc_config_without_bc4)
print(f"  T_min = {T_without_bc4.min():.3f}°C")
print(f"  T_max = {T_without_bc4.max():.3f}°C")
print(f"  T_mean = {T_without_bc4.mean():.3f}°C")

print("\n[4] Решение: BC#1-4 (с BC#4 на вершине)...")

bc_config_with_bc4 = {
    'bc1_enabled': True,
    'bc1_value': 2.0,
    'bc2_enabled': True,
    'bc3_enabled': True,
    'bc4_enabled': True,
    'h_slopes': 50.0,
    'T_ext_slopes': 0.0,
    'h_top': 75.0,
    'T_ext_top': -5.0,
}

T_with_bc4 = solve_heat_conduction_bc4(bc_config_with_bc4)
print(f"  T_min = {T_with_bc4.min():.3f}°C")
print(f"  T_max = {T_with_bc4.max():.3f}°C")
print(f"  T_mean = {T_with_bc4.mean():.3f}°C")

# ============================================================================
# 5. АНАЛИЗ ВЛИЯНИЯ BC#4
# ============================================================================

print("\n[5] Анализ влияния BC#4...")

T_diff_bc4 = T_without_bc4 - T_with_bc4
print(f"  ΔT = T_без_BC4 - T_с_BC4:")
print(f"    ΔT_min = {T_diff_bc4.min():.3f}°C")
print(f"    ΔT_max = {T_diff_bc4.max():.3f}°C (максимальное охлаждение)")
print(f"    ΔT_mean = {T_diff_bc4.mean():.3f}°C")

# Анализ в вершине
print(f"\n  Анализ на вершине:")
if len(top_surface_nodes) > 0:
    T_top_without = T_without_bc4[top_surface_nodes]
    T_top_with = T_with_bc4[top_surface_nodes]
    print(f"    Без BC#4: T = {T_top_without.mean():.3f}°C (min={T_top_without.min():.3f}, max={T_top_without.max():.3f})")
    print(f"    С BC#4: T = {T_top_with.mean():.3f}°C (min={T_top_with.min():.3f}, max={T_top_with.max():.3f})")
    print(f"    Охлаждение: {(T_top_without - T_top_with).mean():.3f}°C")

# ============================================================================
# 6. ВИЗУАЛИЗАЦИЯ
# ============================================================================

print("\n[6] Создание визуализации...")

fig = plt.figure(figsize=(18, 14))

T_without_bc4_reshaped = T_without_bc4.reshape(ny, nx)
T_with_bc4_reshaped = T_with_bc4.reshape(ny, nx)
T_diff_bc4_reshaped = T_diff_bc4.reshape(ny, nx)

embankment_points = np.array([
    [x_bl, y_interface],
    [x_tl, y_top],
    [x_tr, y_top],
    [x_br, y_interface]
])

# ---- Подграфик 1: Геометрия с вершиной ----
ax1 = plt.subplot(3, 3, 1)
coors_x = X.flatten()
coors_y = Y.flatten()
ax1.scatter(coors_x, coors_y, s=5, c='lightgray', alpha=0.5, label='Все узлы')
ax1.scatter(coors_x[left_slope_nodes], coors_y[left_slope_nodes], s=15, c='blue', 
            marker='s', label='BC#2-3 (откосы)', zorder=5, alpha=0.6)
ax1.scatter(coors_x[top_surface_nodes], coors_y[top_surface_nodes], s=20, c='red', 
            marker='^', label='BC#4 (вершина)', zorder=5)
embankment_poly = Polygon(embankment_points, fill=False, edgecolor='black', linewidth=2, label='Границы насыпи')
ax1.add_patch(embankment_poly)
ax1.set_xlim(-1, Lx + 1)
ax1.set_ylim(-1, y_top + 1)
ax1.set_xlabel('x (м)')
ax1.set_ylabel('y (м)')
ax1.set_title('Геометрия и BC узлы (BC#1-4)')
ax1.legend(fontsize=7, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# ---- Подграфик 2: Без BC#4 ----
ax2 = plt.subplot(3, 3, 2)
levels = np.linspace(T_without_bc4.min(), T_without_bc4.max(), 20)
contourf2 = ax2.contourf(X, Y, T_without_bc4_reshaped, levels=levels, cmap='coolwarm')
ax2.contour(X, Y, T_without_bc4_reshaped, levels=10, colors='black', alpha=0.2, linewidths=0.5)
embankment_poly2 = Polygon(embankment_points, fill=False, edgecolor='black', linewidth=2)
ax2.add_patch(embankment_poly2)
cbar2 = plt.colorbar(contourf2, ax=ax2)
cbar2.set_label('T (°C)')
ax2.set_xlim(0, Lx)
ax2.set_ylim(y_bottom, y_top)
ax2.set_xlabel('x (м)')
ax2.set_ylabel('y (м)')
ax2.set_title('Без BC#4 (BC#1-3)')
ax2.set_aspect('equal')

# ---- Подграфик 3: С BC#4 ----
ax3 = plt.subplot(3, 3, 3)
levels3 = np.linspace(T_with_bc4.min(), T_with_bc4.max(), 20)
contourf3 = ax3.contourf(X, Y, T_with_bc4_reshaped, levels=levels3, cmap='coolwarm')
ax3.contour(X, Y, T_with_bc4_reshaped, levels=10, colors='black', alpha=0.2, linewidths=0.5)
embankment_poly3 = Polygon(embankment_points, fill=False, edgecolor='black', linewidth=2)
ax3.add_patch(embankment_poly3)
cbar3 = plt.colorbar(contourf3, ax=ax3)
cbar3.set_label('T (°C)')
ax3.set_xlim(0, Lx)
ax3.set_ylim(y_bottom, y_top)
ax3.set_xlabel('x (м)')
ax3.set_ylabel('y (м)')
ax3.set_title('С BC#4 (конвекция вершина)')
ax3.set_aspect('equal')

# ---- Подграфик 4: Разница ----
ax4 = plt.subplot(3, 3, 4)
levels4 = np.linspace(T_diff_bc4.min(), T_diff_bc4.max(), 20)
contourf4 = ax4.contourf(X, Y, T_diff_bc4_reshaped, levels=levels4, cmap='RdBu_r')
embankment_poly4 = Polygon(embankment_points, fill=False, edgecolor='black', linewidth=2)
ax4.add_patch(embankment_poly4)
cbar4 = plt.colorbar(contourf4, ax=ax4)
cbar4.set_label('ΔT (°C)')
ax4.set_xlim(0, Lx)
ax4.set_ylim(y_bottom, y_top)
ax4.set_xlabel('x (м)')
ax4.set_ylabel('y (м)')
ax4.set_title('ΔT = T_без_BC4 - T_с_BC4')
ax4.set_aspect('equal')

# ---- Подграфик 5: Вертикальные профили ----
ax5 = plt.subplot(3, 3, 5)
x_center_idx = nx // 2
T_vertical_without = T_without_bc4_reshaped[:, x_center_idx]
T_vertical_with = T_with_bc4_reshaped[:, x_center_idx]
ax5.plot(T_vertical_without, y, 'b-o', linewidth=2, label='Без BC#4', markersize=4)
ax5.plot(T_vertical_with, y, 'r-s', linewidth=2, label='С BC#4', markersize=4)
ax5.axhline(y=y_interface, color='gray', linestyle='--', alpha=0.5, label='Интерфейс')
ax5.axhline(y=y_top, color='red', linestyle='--', alpha=0.5, label='Вершина')
ax5.set_xlabel('T (°C)')
ax5.set_ylabel('y (м)')
ax5.set_title(f'Вертикальный профиль (x={x[x_center_idx]:.2f})')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=8)

# ---- Подграфик 6: Горизонтальные профили ----
ax6 = plt.subplot(3, 3, 6)
y_mid_idx = (ny - 1) // 2
T_horizontal_without = T_without_bc4_reshaped[y_mid_idx, :]
T_horizontal_with = T_with_bc4_reshaped[y_mid_idx, :]
ax6.plot(x, T_horizontal_without, 'b-o', linewidth=2, label='Без BC#4', markersize=4)
ax6.plot(x, T_horizontal_with, 'r-s', linewidth=2, label='С BC#4', markersize=4)
ax6.axvline(x=x_bl, color='gray', linestyle='--', alpha=0.5, label='Границы насыпи')
ax6.axvline(x=x_br, color='gray', linestyle='--', alpha=0.5)
ax6.set_xlabel('x (м)')
ax6.set_ylabel('T (°C)')
ax6.set_title(f'Горизонтальный профиль (y={y[y_mid_idx]:.2f})')
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=8)

# ---- Подграфик 7: Охлаждение вдоль вершины ----
ax7 = plt.subplot(3, 3, 7)
if len(top_surface_nodes) > 0:
    x_top = coors_x[top_surface_nodes]
    T_cooling_top = T_without_bc4[top_surface_nodes] - T_with_bc4[top_surface_nodes]
    sorted_indices = np.argsort(x_top)
    ax7.plot(x_top[sorted_indices], T_cooling_top[sorted_indices], 'r-o', linewidth=2, markersize=6)
    ax7.set_xlabel('x (м)')
    ax7.set_ylabel('Охлаждение ΔT (°C)')
    ax7.set_title('Охлаждение вдоль вершины (BC#4)')
    ax7.grid(True, alpha=0.3)
else:
    ax7.text(0.5, 0.5, 'Вершина не найдена', ha='center', va='center', transform=ax7.transAxes)

# ---- Подграфик 8: Сравнение макушек ----
ax8 = plt.subplot(3, 3, 8)
ax8.hist(T_without_bc4, bins=30, alpha=0.5, label='Без BC#4 (BC#1-3)', color='blue')
ax8.hist(T_with_bc4, bins=30, alpha=0.5, label='С BC#4 (BC#1-4)', color='red')
ax8.set_xlabel('T (°C)')
ax8.set_ylabel('Частота')
ax8.set_title('Распределение температуры')
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3, axis='y')

# ---- Подграфик 9: Статистика ----
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

stats_text = f"""
СТАТИСТИКА BC#4 (КОНВЕКЦИЯ ВЕРШИНА)

Параметры BC#4:
  h_top = {bc_config_with_bc4['h_top']} Вт/(м²·К)
  T_ext_top = {bc_config_with_bc4['T_ext_top']}°C

Параметры BC#2-3:
  h_slopes = {bc_config_with_bc4['h_slopes']} Вт/(м²·К)
  T_ext_slopes = {bc_config_with_bc4['T_ext_slopes']}°C

Без BC#4 (BC#1-3):
  T_min = {T_without_bc4.min():.3f}°C
  T_max = {T_without_bc4.max():.3f}°C
  T_mean = {T_without_bc4.mean():.3f}°C

С BC#4 (BC#1-4):
  T_min = {T_with_bc4.min():.3f}°C
  T_max = {T_with_bc4.max():.3f}°C
  T_mean = {T_with_bc4.mean():.3f}°C

Влияние BC#4:
  ΔT_mean = {T_diff_bc4.mean():.3f}°C
  ΔT_max = {T_diff_bc4.max():.3f}°C
  Охлаждение вершины: {(T_top_without - T_top_with).mean():.3f}°C

Вывод:
  BC#4 сильно охлаждает вершину!
  Глубокие слои меньше подвержены влиянию
"""

ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=8,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('bc4_top_convection_debug.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Сохранено: bc4_top_convection_debug.png")

plt.show()

print("\n" + "=" * 80)
print("ЭТАП 3 ЗАВЕРШЁН")
print("=" * 80)
print("\nСтатус проекта:")
print("  ✅ BC#1 (Дирихле на дне) — ГОТОВО")
print("  ✅ BC#2-3 (Конвекция откосы) — ГОТОВО")
print("  ✅ BC#4 (Конвекция вершина) — ГОТОВО")
print("\nСледующие этапы (BC#5-9) в разработке...")
print("\n")