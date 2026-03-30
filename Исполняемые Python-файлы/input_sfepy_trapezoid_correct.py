python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sfepy_trapezoid_correct.py

Визуализация геометрии 2D области (грунт + насыпь).
НЕ решает задачу теплопроводности, только показывает геометрию и маски.

Используется для:
1. Проверки правильности параметров геометрии
2. Отладки масок (mask_embankment, mask_soil)
3. Проверки интерфейса грунт/насыпь
4. Визуального контроля перед запуском полной модели

Запуск:
    python sfepy_trapezoid_correct.py

Выход:
    Окно matplotlib с визуализацией
    Возможно PNG файл
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_geometry_with_visualization(nx=41, ny=33, embankment_height=3.0, 
                                      embankment_base=6.0, soil_depth=5.0):
    """
    Создаёт и визуализирует геометрию: грунт (внизу) + трапециевидная насыпь (сверху).
    
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
    
    print("\n[1] Создание геометрии с визуализацией...")
    
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
    
    print(f"  Параметры:")
    print(f"    Область: {Lx:.2f} x {Ly:.2f} м")
    print(f"    Сетка: {nx} x {ny} узлов")
    print(f"    Глубина грунта: {soil_depth:.2f} м")
    print(f"    Высота насыпи: {embankment_height:.2f} м")
    print(f"    Ширина основания насыпи: {embankment_base:.2f} м")
    
    print(f"\n  Координаты:")
    print(f"    y_bottom = {y[0]:.2f} м")
    print(f"    y_interface (грунт/насыпь) = {y_interface:.2f} м")
    print(f"    y_top (вершина) = {y_top:.2f} м")
    print(f"\n    x_bl (основание насыпи, левое) = {x_bl:.2f} м")
    print(f"    x_br (основание насыпи, правое) = {x_br:.2f} м")
    print(f"    x_tl (вершина, левое) = {x_tl:.2f} м")
    print(f"    x_tr (вершина, правое) = {x_tr:.2f} м")
    
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
                    # Параметр t: 0 на y_interface, 1 на y_top
                    t = (y_node - y_interface) / (y_top - y_interface)
                    
                    # Левый откос: линейная интерполяция
                    x_left_slope = x_bl + (x_tl - x_bl) * t
                    
                    # Правый откос: линейная интерполяция
                    x_right_slope = x_br + (x_tr - x_br) * t
                    
                    if x_left_slope <= x_node <= x_right_slope:
                        mask_embankment[j, i] = True
    
    mask_soil = ~mask_embankment
    
    print(f"\n  Маски:")
    print(f"    Узлов в насыпи: {mask_embankment.sum()}")
    print(f"    Узлов в грунте: {mask_soil.sum()}")
    print(f"    Всего узлов: {nx * ny}")
    
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


def plot_geometry(geometry):
    """
    Создаёт 2x2 визуализацию геометрии.
    
    Parameters
    ----------
    geometry : dict
        Выход create_geometry_with_visualization()
    """
    
    print("\n[2] Создание визуализации...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    X = geometry['X']
    Y = geometry['Y']
    mask_embankment = geometry['mask_embankment']
    mask_soil = geometry['mask_soil']
    nx = geometry['nx']
    ny = geometry['ny']
    
    y_interface = geometry['y_interface']
    y_top = geometry['y_top']
    x_bl = geometry['x_bl']
    x_br = geometry['x_br']
    x_tl = geometry['x_tl']
    x_tr = geometry['x_tr']
    
    # ========================================================================
    # Панель 1: Маски (сетка узлов, цветовая карта)
    # ========================================================================
    ax = axes[0, 0]
    
    # Цветовая карта: 0 = грунт (синий), 1 = насыпь (жёлтый)
    mask_display = mask_embankment.astype(float)
    im = ax.contourf(X, Y, mask_display, levels=[0, 0.5, 1], colors=['lightblue', 'yellow'])
    
    # Граница между грунтом и насыпью
    ax.axhline(y=y_interface, color='red', linestyle='--', linewidth=2, label='Интерфейс грунт/насыпь')
    
    # Сетка узлов (редко)
    if nx <= 50 and ny <= 50:
        ax.scatter(X, Y, s=5, alpha=0.3, color='gray', label='Узлы сетки')
    
    ax.set_xlabel('X (м)', fontsize=11)
    ax.set_ylabel('Y (м)', fontsize=11)
    ax.set_title('Маски: Грунт (голубой) / Насыпь (жёлтый)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)
    
    # ========================================================================
    # Панель 2: Геометрия с контурами трапеции
    # ========================================================================
    ax = axes[0, 1]
    
    # Фон: цветовая карта масок
    im = ax.contourf(X, Y, mask_display, levels=[0, 0.5, 1], colors=['lightblue', 'yellow'])
    
    # Трапеция (контур насыпи)
    trapezoid = patches.Polygon(
        [(x_bl, y_interface), (x_br, y_interface), (x_tr, y_top), (x_tl, y_top)],
        fill=False,
        edgecolor='red',
        linewidth=2.5,
        linestyle='-',
        label='Контур насыпи'
    )
    ax.add_patch(trapezoid)
    
    # Отметить ключевые точки
    ax.plot([x_bl, x_br], [y_interface, y_interface], 'ro', markersize=8, label='Основание')
    ax.plot([x_tl, x_tr], [y_top, y_top], 'go', markersize=8, label='Вершина')
    
    # Аннотации
    ax.text(x_bl, y_interface - 0.2, f'({x_bl:.1f}, {y_interface:.1f})', fontsize=9, ha='center')
    ax.text(x_br, y_interface - 0.2, f'({x_br:.1f}, {y_interface:.1f})', fontsize=9, ha='center')
    ax.text(x_tl - 0.3, y_top + 0.2, f'({x_tl:.1f}, {y_top:.1f})', fontsize=9, ha='right')
    ax.text(x_tr + 0.3, y_top + 0.2, f'({x_tr:.1f}, {y_top:.1f})', fontsize=9, ha='left')
    
    ax.axhline(y=y_interface, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('X (м)', fontsize=11)
    ax.set_ylabel('Y (м)', fontsize=11)
    ax.set_title('Геометрия: Контур трапеции', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)
    
    # ========================================================================
    # Панель 3: Гистограмма высот узлов в насыпи
    # ========================================================================
    ax = axes[1, 0]
    
    # Высоты узлов в насыпи
    y_embankment = Y[mask_embankment]
    
    ax.hist(y_embankment, bins=20, color='orange', alpha=0.7, edgecolor='black')
    ax.axvline(x=y_interface, color='red', linestyle='--', linewidth=2, label=f'y_interface = {y_interface:.2f}')
    ax.axvline(x=y_top, color='green', linestyle='--', linewidth=2, label=f'y_top = {y_top:.2f}')
    ax.set_xlabel('Y (м)', fontsize=11)
    ax.set_ylabel('Количество узлов', fontsize=11)
    ax.set_title('Распределение Y координат в насыпи', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')
    
    # ========================================================================
    # Панель 4: Статистика и информация
    # ========================================================================
    ax = axes[1, 1]
    ax.axis('off')
    
    # Текстовый блок со статистикой
    info_text = f"""
ГЕОМЕТРИЯ И МАСКИ

Параметры области:
  Ширина (Lx):           {geometry['Lx']:.2f} м
  Высота (Ly):           {geometry['Ly']:.2f} м
  Сетка:                 {geometry['nx']} × {geometry['ny']} узлов

Интерфейсы:
  y_interface (грунт):   {y_interface:.2f} м
  y_top (вершина):       {y_top:.2f} м

Основание насыпи:
  x_bl (левое):          {x_bl:.2f} м
  x_br (правое):         {x_br:.2f} м
  Ширина:                {x_br - x_bl:.2f} м

Вершина насыпи:
  x_tl (левое):          {x_tl:.2f} м
  x_tr (правое):         {x_tr:.2f} м
  Ширина вершины:        {x_tr - x_tl:.2f} м

Маски узлов:
  В насыпи:             {mask_embankment.sum()} узлов
  В грунте:             {mask_soil.sum()} узлов
  Всего:                {nx * ny} узлов
  
Проверки:
  ✓ Трапеция правильная?  Визуально проверить панель 2
  ✓ Маски корректны?       Синий (грунт) / Жёлтый (насыпь)
  ✓ Интерфейс на месте?    Красная линия на панели 1
    """
    
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========================================================================
    
    plt.tight_layout()
    plt.savefig('sfepy_trapezoid_geometry.png', dpi=100, bbox_inches='tight')
    print("  Сохранено: sfepy_trapezoid_geometry.png")
    
    print("\n[3] Визуализация завершена!")
    plt.show()


def validate_geometry(geometry):
    """
    Проверяет корректность геометрии.
    
    Parameters
    ----------
    geometry : dict
        Выход create_geometry_with_visualization()
    """
    
    print("\n[4] Валидация геометрии...")
    
    x_bl = geometry['x_bl']
    x_br = geometry['x_br']
    x_tl = geometry['x_tl']
    x_tr = geometry['x_tr']
    y_interface = geometry['y_interface']
    y_top = geometry['y_top']
    Lx = geometry['Lx']
    Ly = geometry['Ly']
    
    checks = []
    
    # Проверка 1: основание шире вершины
    base_width = x_br - x_bl
    top_width = x_tr - x_tl
    check1 = base_width > top_width
    checks.append(check1)
    print(f"  ✓ Основание шире вершины: {check1}")
    print(f"    Ширина основания: {base_width:.2f} м")
    print(f"    Ширина вершины: {top_width:.2f} м")
    
    # Проверка 2: вершина находится между левой и правой границей
    check2 = x_bl < x_tl and x_tr < x_br
    checks.append(check2)
    print(f"  ✓ Вершина внутри основания: {check2}")
    
    # Проверка 3: интерфейс находится внутри области
    check3 = 0 < y_interface < Ly
    checks.append(check3)
    print(f"  ✓ Интерфейс внутри области: {check3}")
    
    # Проверка 4: есть узлы в насыпи
    check4 = geometry['mask_embankment'].sum() > 0
    checks.append(check4)
    print(f"  ✓ Есть узлы в насыпи: {check4}")
    
    # Проверка 5: есть узлы в грунте
    check5 = geometry['mask_soil'].sum() > 0
    checks.append(check5)
    print(f"  ✓ Есть узлы в грунте: {check5}")
    
    # Общий результат
    all_ok = all(checks)
    if all_ok:
        print("\n  ✅ Все проверки пройдены! Геометрия корректна.")
    else:
        print("\n  ❌ Некоторые проверки не пройдены. Проверьте параметры.")
    
    return all_ok


def main():
    """Главная функция."""
    
    print("="*80)
    print("SFEpy 2D BC Debug — ВИЗУАЛИЗАЦИЯ ГЕОМЕТРИИ (sfepy_trapezoid_correct.py)")
    print("="*80)
    
    # Параметры (можно менять)
    nx = 41
    ny = 33
    embankment_height = 3.0
    embankment_base = 6.0
    soil_depth = 5.0
    
    # [1] Создание геометрии
    geometry = create_geometry_with_visualization(
        nx=nx,
        ny=ny,
        embankment_height=embankment_height,
        embankment_base=embankment_base,
        soil_depth=soil_depth
    )
    
    # [2] Визуализация
    plot_geometry(geometry)
    
    # [3] Валидация
    all_ok = validate_geometry(geometry)
    
    print("\n" + "="*80)
    if all_ok:
        print("✓ Геометрия успешно визуализирована и валидирована!")
        print("  Теперь можно запускать полную модель:")
        print("  python sfepy_bc_debug_interactive.py")
    else:
        print("❌ Есть проблемы с геометрией. Проверьте параметры выше.")
    print("="*80)


if __name__ == '__main__':
    main()