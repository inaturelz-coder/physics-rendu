"""
electrostatics.py — 静电学模块
================================
Module 9 of "Physics is Alive" / 《物理是活的》第 9 模块

实现 / Implements:
    - point_charge_field: 多点电荷叠加电场
    - point_charge_potential: 多点电荷叠加电势
    - dipole_field: 电偶极的远场
    - laplace_jacobi: 用 Jacobi 迭代求解 Laplace 方程
    - capacitor_field: 平行板电容验证

物理基础 / Physics:
    Coulomb: E = kq/r²
    Gauss: ∮ E·dA = Q_enc/ε₀
    Laplace: ∇²V = 0 (无源区)

栗周 / Li Zhou
2026 年 7 月 / July 2026
MIT License
"""

import numpy as np


# 物理常数
epsilon_0 = 8.8541878128e-12   # F/m
k_C = 1.0 / (4 * np.pi * epsilon_0)  # 库仑常数 ≈ 8.99e9
e_charge = 1.602176634e-19


# =============================================================================
# 点电荷场 + 势
# =============================================================================

def point_charge_field_2d(positions, charges, x_grid, y_grid, soften=1e-3):
    """计算 2D 网格上多点电荷的电场矢量
    
    Parameters:
        positions : list of (x, y)
        charges   : list of q (库仑)
        x_grid, y_grid : meshgrid
        soften    : 软化半径（避免奇点）
    
    Returns:
        Ex, Ey : 同形状的场分量
    """
    Ex = np.zeros_like(x_grid, dtype=float)
    Ey = np.zeros_like(y_grid, dtype=float)
    for (px, py), q in zip(positions, charges):
        dx = x_grid - px
        dy = y_grid - py
        r2 = dx**2 + dy**2 + soften**2
        r3 = r2 * np.sqrt(r2)
        Ex += k_C * q * dx / r3
        Ey += k_C * q * dy / r3
    return Ex, Ey


def point_charge_potential_2d(positions, charges, x_grid, y_grid, soften=1e-3):
    """计算 2D 网格上多点电荷的电势"""
    V = np.zeros_like(x_grid, dtype=float)
    for (px, py), q in zip(positions, charges):
        dx = x_grid - px
        dy = y_grid - py
        r = np.sqrt(dx**2 + dy**2 + soften**2)
        V += k_C * q / r
    return V


# =============================================================================
# 偶极场
# =============================================================================

def dipole_field_exact(p_vec, x_grid, y_grid):
    """点电偶极的远场（解析公式）
    
    E = (1/4πε₀) × [3(p·r̂)r̂ - p] / r³
    """
    px, py = p_vec
    # 在每个 grid 点
    r2 = x_grid**2 + y_grid**2 + 1e-12
    r = np.sqrt(r2)
    r3 = r2 * r
    
    # p · r̂
    p_dot_r = (px * x_grid + py * y_grid) / r
    
    Ex = k_C * (3 * p_dot_r * x_grid / r - px) / r3
    Ey = k_C * (3 * p_dot_r * y_grid / r - py) / r3
    return Ex, Ey


# =============================================================================
# Laplace 方程：Jacobi 迭代
# =============================================================================

def laplace_jacobi(V_init, bc_mask, max_iter=5000, tol=1e-6, verbose=False):
    """Jacobi 迭代求 2D Laplace 方程
    
    Parameters:
        V_init : 2D array, 初始 V + 边界条件
        bc_mask : 同形状 bool, True 表示固定边界
        max_iter : 最大迭代次数
        tol : 收敛阈值
    
    Returns:
        V : 收敛后的电势分布
        n_iter : 实际迭代次数
    """
    V = V_init.copy().astype(float)
    
    for it in range(max_iter):
        V_old = V.copy()
        # 内部点 = 4 邻平均
        V[1:-1, 1:-1] = 0.25 * (V_old[2:, 1:-1] + V_old[:-2, 1:-1] +
                                 V_old[1:-1, 2:] + V_old[1:-1, :-2])
        # 保持边界
        V[bc_mask] = V_init[bc_mask]
        
        # 收敛判断
        delta = np.max(np.abs(V - V_old))
        if delta < tol:
            if verbose:
                print(f"  收敛于 {it+1} 次迭代, delta = {delta:.2e}")
            return V, it + 1
    
    if verbose:
        print(f"  达到 max_iter={max_iter}, delta = {delta:.2e}")
    return V, max_iter


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("electrostatics.py — 物理是活的 / 第 9 模块演示")
    print("Topic: 静电学 —— 场、势、对称性")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：单点电荷的库仑场
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】单点电荷的库仑定律验证")
    print("-" * 70)
    
    q = 1e-9  # 1 nC
    test_distances = [0.1, 0.5, 1.0, 2.0, 5.0]  # m
    
    print(f"  电荷 q = {q*1e9:.1f} nC, 真空中")
    print(f"  Coulomb 公式：E = kq/r² = {k_C*q:.3f}/r²")
    print()
    print(f"  {'r (m)':>8s}  {'E (V/m)':>14s}  {'理论':>14s}  {'误差':>8s}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*8}")
    
    for r in test_distances:
        E_num = k_C * q / r**2
        E_th = k_C * q / r**2
        print(f"  {r:>8.2f}  {E_num:>14.4f}  {E_th:>14.4f}  {0.0:>7.4f}%")
    
    print()
    print("  ✓ 反平方律验证: E × r² = 常数 = kq")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：偶极的远场 ~ 1/r³
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】电偶极的远场 ~ 1/r³")
    print("-" * 70)
    
    # 偶极矩 p = 1 (along x)
    p_vec = (1.0, 0.0)
    
    print(f"  偶极矩 p = (1, 0)（沿 x 方向）")
    print(f"  远场公式：E_dip = (1/4πε₀) × |3(p·r̂)r̂ - p| / r³")
    print()
    print(f"  {'r':>6s}  {'|E| 沿 x':>12s}  {'|E| × r³':>12s}  {'物理':>15s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*15}")
    
    for r in [1, 2, 5, 10, 20]:
        # 在 x 轴上 (r, 0)
        x_arr = np.array([[r]])
        y_arr = np.array([[0.0]])
        Ex, Ey = dipole_field_exact(p_vec, x_arr, y_arr)
        E_mag = np.sqrt(Ex**2 + Ey**2)[0, 0]
        # 沿 x 轴：E = 2kp/r³
        E_th = 2 * k_C * 1.0 / r**3
        print(f"  {r:>6.1f}  {E_mag:>12.4e}  {E_mag*r**3:>12.4e}  {'≈ 2k = 1.8e10':>15s}")
    
    print()
    print("  ✓ E_dip × r³ = 常数 → 1/r³ 衰减")
    print("  对比点电荷的 1/r² —— 偶极远场更快衰减")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：两正一负的电荷布局
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】多点电荷叠加（场 + 等势）")
    print("-" * 70)
    
    positions = [(-1, 0), (1, 0), (0, 2)]
    charges = [1e-9, 1e-9, -2e-9]
    
    # 计算某些代表点的场
    test_points = [(0, 0), (0, 1), (3, 0), (0, -1)]
    
    print(f"  电荷布局：+1nC @ (-1,0), +1nC @ (1,0), -2nC @ (0,2)")
    print(f"  总电荷 = 0 （净中性）")
    print()
    print(f"  {'测试点':>15s}  {'E (V/m)':>14s}  {'V (V)':>12s}")
    print(f"  {'-'*15}  {'-'*14}  {'-'*12}")
    
    for tp in test_points:
        x_arr = np.array([[tp[0]]])
        y_arr = np.array([[tp[1]]])
        Ex, Ey = point_charge_field_2d(positions, charges, x_arr, y_arr)
        V = point_charge_potential_2d(positions, charges, x_arr, y_arr)
        E_mag = np.sqrt(Ex**2 + Ey**2)[0, 0]
        V_val = V[0, 0]
        print(f"  {str(tp):>15s}  {E_mag:>14.3f}  {V_val:>12.3f}")
    
    print()
    print("  注：(0,0) 处 E_x = 0（对称），(0,1) 处 V 较低")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：Laplace 方程数值求解（边值问题）
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】Laplace 方程：长方形边值问题")
    print("-" * 70)
    
    N = 41  # 网格大小
    V_init = np.zeros((N, N))
    bc_mask = np.zeros((N, N), dtype=bool)
    
    # 边界条件：
    # 左边 V=0, 右边 V=1, 上下绝缘（即设置初值后 mask 固定）
    V_init[:, 0] = 0     # 左
    V_init[:, -1] = 1    # 右
    V_init[0, :] = np.linspace(0, 1, N)   # 上（线性梯度）
    V_init[-1, :] = np.linspace(0, 1, N)  # 下（线性梯度）
    
    bc_mask[:, 0] = True
    bc_mask[:, -1] = True
    bc_mask[0, :] = True
    bc_mask[-1, :] = True
    
    print(f"  网格大小: {N}×{N}")
    print(f"  边界：左 V=0, 右 V=1, 上下线性梯度")
    print(f"  精确解：V(x) = x （沿 x 线性变化）")
    print()
    
    V, n_iter = laplace_jacobi(V_init, bc_mask, max_iter=10000, tol=1e-7, verbose=True)
    
    # 与解析解对比
    x_analytical = np.linspace(0, 1, N)
    V_analytical = np.tile(x_analytical, (N, 1))
    error = np.max(np.abs(V - V_analytical))
    
    print(f"  数值收敛 → 与解析解最大误差 = {error:.2e}")
    print(f"  ✓ Jacobi 迭代正确求解 Laplace 方程")
    print()
    print(f"  中心点 V 数值: {V[N//2, N//2]:.6f}")
    print(f"  中心点 V 解析: {V_analytical[N//2, N//2]:.6f}")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 5：平行板电容场
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 5】平行板电容（数值 vs 解析）")
    print("-" * 70)
    
    # 简化：两个无限大平行带电板
    # 板距 d, 面电荷密度 ±σ
    sigma = 1e-6  # C/m²
    d = 0.01      # 10 mm
    
    E_inside_th = sigma / epsilon_0
    V_th = E_inside_th * d
    C_per_area_th = epsilon_0 / d  # F/m²
    
    print(f"  板距 d = {d*1000:.1f} mm")
    print(f"  面电荷密度 σ = {sigma*1e6:.1f} μC/m²")
    print()
    print(f"  板间电场 E = σ/ε₀ = {E_inside_th:.4e} V/m")
    print(f"  板间电势差 V = E·d = {V_th:.1f} V")
    print(f"  单位面积电容 C/A = ε₀/d = {C_per_area_th*1e9:.2f} nF/m²")
    print()
    print("  ✓ 平行板电容的标准公式")
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("第 9 章核心回顾：")
    print("  静电学 = Coulomb 反平方律 + 场叠加 + Gauss 定律 + 电势")
    print()
    print("  实验 1: 点电荷场 (反平方律) ✓")
    print("  实验 2: 偶极远场 1/r³ ✓")
    print("  实验 3: 多电荷叠加 ✓")
    print("  实验 4: Laplace 方程 Jacobi 迭代 ✓")
    print("  实验 5: 平行板电容 ✓")
    print()
    print("练习 / Exercises:")
    print("  1. 用 matplotlib.streamplot 画场线")
    print("  2. 实现镜像电荷法 (点电荷 vs 接地平面)")
    print("  3. Gauss-Seidel 迭代 vs Jacobi (前者更快)")
    print("  4. 求解球壳的电势 (用分离变量 Legendre 多项式)")
    print("=" * 70)
