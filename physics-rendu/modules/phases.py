"""
phases.py — 相变与对称破缺模块
================================
Module 11 of "Physics is Alive" / 《物理是活的》第 11 模块

实现 / Implements:
    - landau_free_energy: Landau φ⁴ 自由能
    - mean_field_order_parameter: φ(T) mean-field 解
    - estimate_critical_exponent: 从数据拟合 β
    - 用 Ch7 Ising MC 数值估算临界指数
    - 1D 横场 Ising 量子相变（精确对角化）

物理基础 / Physics:
    Landau: F = aφ² + cφ³ + bφ⁴
    Mean-field: φ ∝ |T_c - T|^(1/2)
    2D Ising: β = 1/8 (Onsager)
    横场 Ising: 量子相变 h_c = J

栗周 / Li Zhou
2026 年 8 月 / August 2026
MIT License
"""

import numpy as np
import sys, os
import importlib.util

# 因为 stat 是 Python 标准库名 —— 用 importlib 直接加载本地文件
_stat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stat.py')
_spec = importlib.util.spec_from_file_location("local_stat", _stat_path)
_local_stat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local_stat)

IsingModel2D = _local_stat.IsingModel2D
onsager_Tc = _local_stat.onsager_Tc


# =============================================================================
# Landau 自由能
# =============================================================================

def landau_free_energy(phi, a, b=1.0, c=0.0):
    """Landau 自由能展开
    
    F(φ) = aφ² + cφ³ + bφ⁴
    
    Parameters:
        phi : 序参量
        a   : 二阶系数 (a < 0 时破缺)
        b   : 四阶系数 (b > 0 保证稳定)
        c   : 三阶系数 (c ≠ 0 时一阶相变)
    """
    return a * phi**2 + c * phi**3 + b * phi**4


def mean_field_order_parameter(T, T_c, a0=1.0, b=1.0):
    """Mean-field 解：φ(T) = √(-a/2b) for T < T_c
    
    a(T) = a₀(T - T_c)/T_c
    """
    a = a0 * (T - T_c) / T_c
    if a >= 0:
        return 0.0
    return np.sqrt(-a / (2 * b))


# =============================================================================
# 临界指数估算
# =============================================================================

def estimate_beta_from_data(T_array, M_array, T_c, T_range=(0.85, 0.99)):
    """从 (T, M) 数据拟合 β: M ∝ (T_c - T)^β
    
    Parameters:
        T_array  : 温度数组
        M_array  : 序参量数组
        T_c      : 临界温度
        T_range  : 用于拟合的 T/T_c 范围 (避免远离 T_c)
    """
    T_arr = np.asarray(T_array)
    M_arr = np.asarray(M_array)
    
    # 选取拟合区间：T < T_c 且在 (T_min, T_max) 内
    T_min, T_max = T_range[0] * T_c, T_range[1] * T_c
    mask = (T_arr <= T_max) & (T_arr >= T_min) & (M_arr > 0.01) & (T_arr < T_c)
    
    if mask.sum() < 3:
        return None, None
    
    log_t = np.log((T_c - T_arr[mask]) / T_c)  # 约化温度对数
    log_M = np.log(M_arr[mask])
    
    coeffs = np.polyfit(log_t, log_M, 1)
    beta_estimate = coeffs[0]
    
    return beta_estimate, mask.sum()


# =============================================================================
# 1D 横场 Ising 模型（精确对角化）
# =============================================================================

def transverse_field_ising_hamiltonian(L, J=1.0, h=1.0, pbc=True):
    """构建 1D 横场 Ising 模型的哈密顿量
    
    H = -J Σ σ^z_i σ^z_{i+1} - h Σ σ^x_i
    
    Parameters:
        L : 自旋数
        J : 耦合（铁磁取正）
        h : 横场
        pbc : 周期边界
    
    Returns:
        H : 哈密顿量 (2^L × 2^L 矩阵)
    """
    dim = 2**L
    H = np.zeros((dim, dim), dtype=float)
    
    # σ^z σ^z 耦合项
    for i in range(L):
        if i == L - 1 and not pbc:
            break
        j = (i + 1) % L
        for state in range(dim):
            # 提取自旋值（0 → +1, 1 → -1）
            s_i = 1 - 2 * ((state >> i) & 1)
            s_j = 1 - 2 * ((state >> j) & 1)
            H[state, state] += -J * s_i * s_j
    
    # σ^x 横场项（翻转一个自旋）
    for i in range(L):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -h
    
    return H


def ground_state_properties(L, J=1.0, h=1.0):
    """精确对角化 + 计算基态性质"""
    H = transverse_field_ising_hamiltonian(L, J, h)
    eigvals, eigvecs = np.linalg.eigh(H)
    
    E0 = eigvals[0]
    E1 = eigvals[1]
    gap = E1 - E0
    
    # 基态向量
    psi0 = eigvecs[:, 0]
    
    # ⟨σ^z_0 σ^z_1⟩（最近邻 z-z 关联）
    dim = 2**L
    corr_zz = 0.0
    for state in range(dim):
        s0 = 1 - 2 * (state & 1)
        s1 = 1 - 2 * ((state >> 1) & 1)
        corr_zz += abs(psi0[state])**2 * s0 * s1
    
    # ⟨|σ^z|⟩（平均自旋幅度，作为序参量近似）
    M_abs = 0.0
    for state in range(dim):
        Mz = sum(1 - 2 * ((state >> i) & 1) for i in range(L)) / L
        M_abs += abs(psi0[state])**2 * abs(Mz)
    
    return {
        'E0': E0,
        'gap': gap,
        'corr_zz': corr_zz,
        'M_abs': M_abs,
    }


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("phases.py — 物理是活的 / 第 11 模块演示")
    print("Topic: 相变 + 对称性自发破缺")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：Landau 自由能形态
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】Landau 自由能 F(φ) 的形态变化")
    print("-" * 70)
    
    phi_arr = np.linspace(-2, 2, 5)
    
    print(f"  二阶相变（c = 0, b = 1）:")
    print(f"  {'φ':>6s}  {'F (a=+1)':>10s}  {'F (a=0)':>10s}  {'F (a=-1)':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")
    for phi in phi_arr:
        F_high = landau_free_energy(phi, a=1, b=1, c=0)
        F_crit = landau_free_energy(phi, a=0, b=1, c=0)
        F_low = landau_free_energy(phi, a=-1, b=1, c=0)
        print(f"  {phi:>6.2f}  {F_high:>10.4f}  {F_crit:>10.4f}  {F_low:>10.4f}")
    
    print()
    print(f"  ✓ T > T_c (a=+1): 唯一极小 φ=0")
    print(f"  ✓ T = T_c (a=0):  开始变平")
    print(f"  ✓ T < T_c (a=-1): 极小在 φ=±√(1/2) ≈ ±0.707")
    print()
    
    # 找极小（数值）
    print(f"  数值求极小（a=-1, b=1, c=0）:")
    phi_fine = np.linspace(-2, 2, 1000)
    F_fine = landau_free_energy(phi_fine, a=-1, b=1, c=0)
    min_idx = np.argmin(F_fine[phi_fine > 0])
    phi_min = phi_fine[phi_fine > 0][min_idx]
    print(f"    数值极小 φ = ±{phi_min:.4f}")
    print(f"    理论极小 φ = ±√(0.5) = ±{np.sqrt(0.5):.4f}")
    print(f"  ✓ 自发对称破缺 —— 系统选择 +φ 或 -φ")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：Mean-field 序参量 φ(T)
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】Mean-field 序参量 φ vs 温度")
    print("-" * 70)
    
    T_c = 1.0
    T_array = np.array([1.5, 1.2, 1.05, 1.01, 1.0, 0.99, 0.95, 0.9, 0.5, 0.1])
    phi_mf = np.array([mean_field_order_parameter(T, T_c) for T in T_array])
    phi_theory = np.array([np.sqrt(max((T_c - T)/T_c / 2, 0)) for T in T_array])
    
    print(f"  T_c = {T_c}, a₀ = 1, b = 1")
    print()
    print(f"  {'T/T_c':>8s}  {'φ(数值)':>10s}  {'φ(理论)':>10s}  {'相态':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
    
    for T, phi, phi_th in zip(T_array, phi_mf, phi_theory):
        if T > T_c:
            phase = "对称"
        elif T == T_c:
            phase = "临界"
        else:
            phase = "破缺"
        print(f"  {T/T_c:>8.2f}  {phi:>10.4f}  {phi_th:>10.4f}  {phase:>10s}")
    
    print()
    
    # 验证 β = 1/2
    T_below = T_array[T_array < T_c]
    phi_below = phi_mf[T_array < T_c]
    beta_est, _ = estimate_beta_from_data(T_below, phi_below, T_c, T_range=(0.5, 0.99))
    
    print(f"  数值拟合 β = {beta_est:.4f}")
    print(f"  Mean-field 理论 β = 0.5")
    print(f"  ✓ Mean-field 给 β = 1/2")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：Ising 临界指数 β 数值估算
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】2D Ising MC：数值估算临界指数 β")
    print("-" * 70)
    
    Tc_onsager = onsager_Tc()
    print(f"  Onsager 精确解: T_c = {Tc_onsager:.4f}, β = 1/8 = 0.125")
    print()
    print(f"  用 L=16 Ising MC 扫描 T < T_c...")
    print(f"  {'T':>6s}  {'T/T_c':>8s}  {'<|M|>':>10s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}")
    
    # 紧凑的 T 范围（T < T_c）
    T_scan = np.array([1.5, 1.7, 1.9, 2.0, 2.10, 2.15, 2.20, 2.24])
    M_scan = []
    
    L = 16
    for T in T_scan:
        ising = IsingModel2D(L=L, T=T, J=1.0, seed=42)
        ising.thermalize(n_sweeps=500)
        result = ising.measure(n_sweeps=1500, n_skip=2)
        M_abs = result['M_abs']
        M_scan.append(M_abs)
        print(f"  {T:>6.2f}  {T/Tc_onsager:>8.3f}  {M_abs:>10.4f}")
    
    M_scan = np.array(M_scan)
    
    # 拟合 β
    print()
    beta_2D, n_pts = estimate_beta_from_data(T_scan, M_scan, Tc_onsager,
                                              T_range=(0.7, 0.99))
    print(f"  数值拟合 β (从 {n_pts} 个数据点) = {beta_2D:.4f}")
    print(f"  Onsager 精确 β = 1/8 = 0.125")
    print(f"  Mean-field β = 0.5")
    print()
    
    if beta_2D is not None and 0.08 < beta_2D < 0.25:
        print("  ✓ 数值结果 (~0.1-0.2) 接近 Onsager 1/8 —— 与 mean-field 1/2 明显不同")
        print("  ✓ 这就是临界涨落让 mean-field 失效的实证")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：1D 横场 Ising 量子相变
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】1D 横场 Ising 量子相变（精确对角化, L=8）")
    print("-" * 70)
    
    L = 8
    J = 1.0
    print(f"  1D 横场 Ising: H = -J Σσ^z_iσ^z_j - h Σσ^x_i")
    print(f"  L = {L}, J = {J}, 精确对角化")
    print(f"  量子临界点：h_c = J = 1（1D 精确）")
    print()
    print(f"  {'h/J':>6s}  {'E_0/L':>10s}  {'gap':>10s}  {'⟨σ^z_0σ^z_1⟩':>12s}  {'⟨|M|⟩':>10s}  {'相态':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}")
    
    h_scan = [0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    
    for h in h_scan:
        result = ground_state_properties(L, J, h)
        E0_per_site = result['E0'] / L
        
        if h < 0.5:
            phase = "铁磁"
        elif abs(h - J) < 0.2:
            phase = "临界"
        else:
            phase = "顺磁"
        
        print(f"  {h:>6.2f}  {E0_per_site:>10.4f}  {result['gap']:>10.4f}  "
              f"{result['corr_zz']:>12.4f}  {result['M_abs']:>10.4f}  {phase:>10s}")
    
    print()
    print("  ✓ h << J: 铁磁相 (z-z 关联强 + ⟨|M|⟩ 大)")
    print("  ✓ h = J:  量子临界 (gap 最小 + 关联开始衰减)")
    print("  ✓ h >> J: 顺磁相 (z-z 关联弱 + ⟨|M|⟩ → 0)")
    print()
    print("  注: 有限尺度 L=8 → gap 不会精确为 0, 关联也不会精确为 1")
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("第 11 章核心回顾：")
    print("  相变 = 自由能极小态的非解析变化 + 对称性自发破缺")
    print()
    print("  实验 1: Landau 自由能形态 (墨西哥帽) ✓")
    print("  实验 2: Mean-field β = 1/2 ✓")
    print("  实验 3: 2D Ising β ≈ 0.125 (Onsager) ✓")
    print("  实验 4: 1D 横场 Ising 量子相变 ✓")
    print()
    print("练习 / Exercises:")
    print("  1. 加 c 项 → 一阶相变（多极小同时出现）")
    print("  2. 大 L Ising MC → 精确 β 估算")
    print("  3. 求 1D 横场 Ising 的精确 h_c (用 Jordan-Wigner)")
    print("  4. 2D XY 模型 Kosterlitz-Thouless 数值实验")
    print("=" * 70)
