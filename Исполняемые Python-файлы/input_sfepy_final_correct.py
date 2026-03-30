#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sfepy_final_correct.py

Базовая модель теплопроводности в 2D области (грунт + насыпь).
Стационарная задача с граничными условиями.

Используется как:
1. Минимальный рабочий пример (MWE)
2. Точка отладки при добавлении новых BC
3. Проверка корректности геометрии и базовой сборки

Запуск:
    python sfepy_final_correct.py

Выход:
    Консоль: статистика, информация о шагах, анализ решения
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def create_geometry(nx=41, ny=33, embankment_height=3.0, embankment_base=6.0, 
                    soil_depth=5.0):
    """
    Создаёт геометрию: грунт (внизу) + трапециевидная насыпь (сверху).
    
    Parameters
    ----------
    nx : int
        Число узлов по X
    ny : int
        Число узлов по Y
    embankment_height : float
        Высота насыпи (м)
    embankment_base : float
        Ширина основания насыпи (м)
    soil_depth : float
        Глубина грунтового слоя (м)
    
    Returns
    -------
    dict
        Словарь с геометрией и масками
    """
    
    print("\n[1] Создание геометрии...")
    
    # Общие размеры области
    Lx = embankment_base + 2.0  # Ширина (с запасом по сторонам)
    Ly = soil_depth + embankment_height
    
    # Создание сетки
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    
    # Координаты интерфейса грунт/насыпь
    y_interface = soil_depth
    
    # Координаты вершины насыпи
    y_top = Ly
    
    # Координаты основания насыпи (левого и правого края)
    x_bl = (Lx - embankment_base) / 2.0
    x_br = x_bl + embankment_base
    
    # Координаты вершины насыпи (верхнего треугольника)
    x_tl = x_bl + embankment_base / 4.0  # Левый угол вершины
    x_tr = x_br - embankment_base / 4.0  # Правый угол вершины
    
    # Маска: узлы в насыпи
    mask_embankment = np.zeros((ny, nx), dtype=bool)
    
    for j in range(ny):
        for i in range(nx):
            y_node = Y[j, i]
            x_node = X[j, i]
            
            if y_node >= y_interface:
                # Проверяем находимся ли внутри трапеции
                # Левый откос: линия от (x_bl, y_interface) к (x_tl, y_top)
                # Правый откос: линия от (x_br, y_interface) к (x_tr, y_top)
                
                if y_node <= y_top:
                    # Левый откос
                    x_left_slope = x_bl + (x_tl - x_bl) * (y_node - y_interface) / (y_top - y_interface)
                    # Правый откос
                    x_right_slope = x_br + (x_tr - x_br) * (y_node - y_interface) / (y_top - y_interface)
                    
                    if x_left_slope <= x_node <= x_right_slope:
                        mask_embankment[j, i] = True
    
    mask_soil = ~mask_embankment
    
    print(f"  Область: {Lx:.1f} x {Ly:.1f} м")
    print(f"  Сетка: {nx} x {ny} узлов")
    print(f"  y_interface = {y_interface:.1f} м")
    print(f"  y_top = {y_top:.1f} м")
    print(f"  Насыпь в области: {mask_embankment.sum()} узлов")
    print(f"  Грунт в области: {mask_soil.sum()} узлов")
    
    geometry = {
        'x': x,
        'y': y,
        'X': X,
        'Y': Y,
        'Lx': Lx,
        'Ly': Ly,
        'nx': nx,
        'ny': ny,
        'y_interface': y_interface,
        'y_top': y_top,
        'x_bl': x_bl,
        'x_br': x_br,
        'x_tl': x_tl,
        'x_tr': x_tr,
        'mask_embankment': mask_embankment,
        'mask_soil': mask_soil,
    }
    
    return geometry


def build_fem_system(geometry, k_embankment=0.5, k_soil=1.5):
    """
    Собирает матрицу жёсткости K и правую часть F 
    для стационарной задачи теплопроводности методом конечных разностей.
    
    Уравнение: -div(k * grad(T)) = 0
    
    5-точечный шаблон КРД:
        T_top
           |
      T_left - T_center - T_right
           |
        T_bottom
    
    Коэффициент: k_avg = (k_center + k_neighbor) / 2
    
    Parameters
    ----------
    geometry : dict
        Выход create_geometry()
    k_embankment : float
        Теплопроводность насыпи (Вт/(м·К))
    k_soil : float
        Теплопроводность грунта (Вт/(м·К))
    
    Returns
    -------
    tuple
        (K_sparse, F, coors, k_field)
    """
    
    print("\n[2] Сборка матрицы K и вектора F...")
    
    x = geometry['x']
    y = geometry['y']
    nx = geometry['nx']
    ny = geometry['ny']
    mask_embankment = geometry['mask_embankment']
    mask_soil = geometry['mask_soil']
    
    # Шаги сетки
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    print(f"  Шаги сетки: dx = {dx:.4f} м, dy = {dy:.4f} м")
    
    # Поле теплопроводности на узлах
    k_field = np.zeros((ny, nx))
    k_field[mask_embankment] = k_embankment
    k_field[mask_soil] = k_soil
    
    # Матрица жёсткости в формате LIL (удобна для построения)
    n_nodes = nx * ny
    K = lil_matrix((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)
    
    # Индексирование: узел (i, j) → индекс i + j * nx
    def idx(i, j):
        return i + j * nx
    
    # Сборка: для каждого внутреннего узла
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            
            c = idx(i, j)  # Центральный узел
            l = idx(i - 1, j)  # Левый
            r = idx(i + 1, j)  # Правый
            t = idx(i, j + 1)  # Верхний
            b = idx(i, j - 1)  # Нижний
            
            k_c = k_field[j, i]
            
            # Коэффициенты с усреднением между соседними узлами
            k_l = (k_c + k_field[j, i - 1]) / 2.0
            k_r = (k_c + k_field[j, i + 1]) / 2.0
            k_t = (k_c + k_field[j + 1, i]) / 2.0
            k_b = (k_c + k_field[j - 1, i]) / 2.0
            
            # Коэффициент при T_center (центр 5-точечного шаблона)
            coeff_c = -2.0 * (k_l + k_r) / (dx * dx) - 2.0 * (k_t + k_b) / (dy * dy)
            
            # Коэффициенты при соседних узлах
            coeff_l = k_l / (dx * dx)
            coeff_r = k_r / (dx * dx)
            coeff_t = k_t / (dy * dy)
            coeff_b = k_b / (dy * dy)
            
            K[c, c] += coeff_c
            K[c, l] += coeff_l
            K[c, r] += coeff_r
            K[c, t] += coeff_t
            K[c, b] += coeff_b
            
            F[c] = 0.0  # Без источников
    
    # Граничные условия: Дирихле на дне (y = 0)
    y_bottom = y[0]
    bc_nodes_bottom = np.where(np.abs(y - y_bottom) < 1e-10)[0]
    
    print(f"  BC узлов на дне: {len(bc_nodes_bottom)}")
    
    # Преобразование в CSR (более эффективный формат для решения)
    K = K.tocsr()
    
    print(f"  Размер системы: {n_nodes} x {n_nodes}")
    print(f"  NNZ (ненулевых элементов): {K.nnz}")
    
    coors = np.column_stack((geometry['X'].ravel(), geometry['Y'].ravel()))
    
    return K, F, coors, k_field, bc_nodes_bottom


def apply_boundary_conditions(K, F, bc_nodes_bottom, T_bottom=2.0):
    """
    Применяет граничные условия Дирихле на дне.
    
    Parameters
    ----------
    K : sparse matrix
        Матрица жёсткости (в формате LIL)
    F : array
        Правая часть
    bc_nodes_bottom : array
        Индексы узлов на дне
    T_bottom : float
        Значение температуры на дне (°C)
    """
    
    print("\n[3] Применение граничных условий...")
    
    # Дирихле BC: T = T_bottom на дне
    # Преобразуем K в LIL для модификации
    K_lil = K.tolil()
    
    for idx in bc_nodes_bottom:
        # Обнулить строку
        K_lil.data[idx] = []
        K_lil.rows[idx] = []
        
        # Диагональный элемент = 1
        K_lil[idx, idx] = 1.0
        
        # Правая часть
        F[idx] = T_bottom
    
    K = K_lil.tocsr()
    
    print(f"  BC#1 (Дирихле на дне): T = {T_bottom}°C")
    print(f"  Применено к {len(bc_nodes_bottom)} узлам")
    
    return K, F


def solve_system(K, F):
    """
    Решает систему линейных уравнений K*T = F.
    
    Parameters
    ----------
    K : sparse matrix (CSR)
        Матрица жёсткости
    F : array
        Правая часть
    
    Returns
    -------
    array
        Вектор температур T
    """
    
    print("\n[4] Решение системы...")
    
    try:
        T = spsolve(K.tocsc(), F)
        print(f"  Решение получено успешно")
        print(f"  Тип: {type(T)}, размер: {T.shape}")
        return T
    except Exception as e:
        print(f"  ОШИБКА при решении: {e}")
        return None


def analyze_solution(T, geometry, coors):
    """
    Анализирует полученное решение.
    
    Parameters
    ----------
    T : array
        Вектор температур
    geometry : dict
        Геометрия области
    coors : array
        Координаты узлов
    """
    
    print("\n[5] Анализ решения...")
    
    nx = geometry['nx']
    ny = geometry['ny']
    y_bottom = geometry['y'][0]
    y_interface = geometry['y_interface']
    y_top = geometry['y_top']
    
    # Статистика
    T_min = np.min(T)
    T_max = np.max(T)
    T_mean = np.mean(T)
    
    print(f"  T_min  = {T_min:.3f}°C")
    print(f"  T_max  = {T_max:.3f}°C")
    print(f"  T_mean = {T_mean:.3f}°C")
    
    # Температура на дне
    bc_nodes_bottom = np.where(np.abs(coors[:, 1] - y_bottom) < 1e-10)[0]
    T_bottom_actual = T[bc_nodes_bottom]
    print(f"\n  На дне (y={y_bottom:.1f}):")
    print(f"    T_min = {T_bottom_actual.min():.3f}°C")
    print(f"    T_max = {T_bottom_actual.max():.3f}°C")
    print(f"    Дисперсия = {T_bottom_actual.std():.6f}°C")
    
    # Температура в интерфейсе
    nodes_interface = np.where(np.abs(coors[:, 1] - y_interface) < 1e-10)[0]
    if len(nodes_interface) > 0:
        T_interface = T[nodes_interface]
        print(f"\n  На интерфейсе (y={y_interface:.1f}):")
        print(f"    T_min = {T_interface.min():.3f}°C")
        print(f"    T_max = {T_interface.max():.3f}°C")
    
    # Температура на вершине
    nodes_top = np.where(np.abs(coors[:, 1] - y_top) < 1e-10)[0]
    if len(nodes_top) > 0:
        T_top = T[nodes_top]
        print(f"\n  На вершине (y={y_top:.1f}):")
        print(f"    T_min = {T_top.min():.3f}°C")
        print(f"    T_max = {T_top.max():.3f}°C")
        print(f"    T_mean = {T_top.mean():.3f}°C")


def visualize_solution(T, geometry, k_field):
    """
    Визуализирует решение (простая 2-панельная диаграмма).
    
    Parameters
    ----------
    T : array
        Вектор температур
    geometry : dict
        Геометрия
    k_field : array
        Поле теплопроводности
    """
    
    print("\n[6] Визуализация...")
    
    nx = geometry['nx']
    ny = geometry['ny']
    X = geometry['X']
    Y = geometry['Y']
    
    # Преобразуем T в 2D массив
    T_2d = T.reshape((ny, nx))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Левая панель: температура
    ax = axes[0]
    levels = np.linspace(T.min(), T.max(), 20)
    cf = ax.contourf(X, Y, T_2d, levels=levels, cmap='RdYlBu_r')
    ax.contour(X, Y, T_2d, levels=levels, colors='k', linewidths=0.3, alpha=0.3)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('T (°C)', fontsize=10)
    ax.set_xlabel('X (м)', fontsize=10)
    ax.set_ylabel('Y (м)', fontsize=10)
    ax.set_title('Температурное поле', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    
    # Правая панель: материалы
    ax = axes[1]
    im = ax.contourf(X, Y, k_field, levels=[0, 0.5, 1.5, 2.0], cmap='Paired')
    ax.set_xlabel('X (м)', fontsize=10)
    ax.set_ylabel('Y (м)', fontsize=10)
    ax.set_title('Материалы (k)', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('sfepy_final_correct_result.png', dpi=100, bbox_inches='tight')
    print("  Сохранено: sfepy_final_correct_result.png")
    
    plt.show()


def main():
    """Главная функция."""
    
    print("="*80)
    print("SFEpy 2D BC Debug — БАЗОВАЯ МОДЕЛЬ (sfepy_final_correct.py)")
    print("="*80)
    
    # [1] Геометрия
    geometry = create_geometry(nx=41, ny=33)
    
    # [2] Сборка системы
    K, F, coors, k_field, bc_nodes_bottom = build_fem_system(
        geometry, k_embankment=0.5, k_soil=1.5
    )
    
    # [3] BC
    K, F = apply_boundary_conditions(K, F, bc_nodes_bottom, T_bottom=2.0)
    
    # [4] Решение
    T = solve_system(K, F)
    
    if T is not None:
        # [5] Анализ
        analyze_solution(T, geometry, coors)
        
        # [6] Визуализация
        visualize_solution(T, geometry, k_field)
        
        print("\n" + "="*80)
        print("✓ Базовая модель успешно решена!")
        print("="*80)
    else:
        print("\n❌ Ошибка при решении системы")


if __name__ == '__main__':
    main()