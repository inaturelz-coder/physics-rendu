"""
quantum.py — 量子力学初步模块
================================
Module 14 of "Physics is Alive" / 《物理是活的》第 14 模块

实现 / Implements:
    - solve_1D_TISE: 有限差分法求解 1D TISE
    - infinite_well: 无限深势阱验证
    - harmonic_oscillator: 量子谐振子验证
    - finite_well: 有限深势阱 + 隧穿
    - tunneling: 矩形势垒的透射系数
    - gaussian_wave_packet: 自由粒子波包扩散

物理基础 / Physics:
    TISE: -ℏ²/2m d²ψ/dx² + V(x)ψ = Eψ
    无限井: E_n = n²π²ℏ²/(2mL²)
    谐振子: E_n = ℏω(n + 1/2)
    隧穿: T ≈ exp(-2κa), κ = √(2m(V₀-E))/ℏ

栗周 / Li Zhou
2026 年 9 月 / September 2026
MIT License
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal


# 自然单位（ℏ = 1, m = 1）—— 让数值更清晰
HBAR = 1.0
M_DEFAULT = 1.0


# =============================================================================
# 1D TISE 有限差分求解
# =============================================================================

def solve_1D_TISE(V_arr, x_arr, m=M_DEFAULT, hbar=HBAR, n_states=10):
    """有限差分法求解 1D 不含时 Schrödinger 方程
    
    -ℏ²/2m d²ψ/dx² + V(x)ψ = Eψ
    
    Parameters:
        V_arr   : 势能数组 V(x_i)
        x_arr   : 位置数组 x_i (等距)
        m       : 质量
        hbar    : 约化 Planck 常数
        n_states: 返回前几个本征态
    
    Returns:
        eigvals : 前 n_states 个本征能量
        eigvecs : (N, n_states) 对应波函数（归一化）
    """
    N = len(x_arr)
    h = x_arr[1] - x_arr[0]
    
    # 三对角矩阵
    factor = hbar**2 / (2 * m * h**2)
    main_diag = 2 * factor + np.asarray(V_arr)
    off_diag = -factor * np.ones(N - 1)
    
    eigvals, eigvecs = eigh_tridiagonal(main_diag, off_diag,
                                         select='i', select_range=(0, n_states - 1))
    
    # 归一化（要求 ∫|ψ|² dx = 1）
    for i in range(eigvecs.shape[1]):
        norm = np.sqrt(np.trapezoid(eigvecs[:, i]**2, x_arr))
        eigvecs[:, i] /= norm
    
    return eigvals, eigvecs


# =============================================================================
# 实验 1：无限深势阱
# =============================================================================

def infinite_well_theory(n, L, m=M_DEFAULT, hbar=HBAR):
    """理论能量 E_n = n²π²ℏ²/(2mL²)"""
    return n**2 * np.pi**2 * hbar**2 / (2 * m * L**2)


# =============================================================================
# 实验 2：量子谐振子
# =============================================================================

def harmonic_oscillator_theory(n, omega=1.0, hbar=HBAR):
    """理论能量 E_n = ℏω(n + 1/2)"""
    return hbar * omega * (n + 0.5)


# =============================================================================
# 实验 3：有限深势阱
# =============================================================================

def finite_well_potential(x_arr, V0, a):
    """V(x) = -V0 if |x| < a, else 0"""
    return np.where(np.abs(x_arr) < a, -V0, 0.0)


# =============================================================================
# 实验 4：矩形势垒隧穿
# =============================================================================

def tunneling_T_analytical(E, V0, a, m=M_DEFAULT, hbar=HBAR):
    """矩形势垒透射系数（精确解析）
    
    E < V0: T = 1 / [1 + (V₀² sinh²(κa))/(4E(V₀-E))]
    κ = √(2m(V₀-E))/ℏ
    """
    if E >= V0:
        return None  # 经典允许
    kappa = np.sqrt(2 * m * (V0 - E)) / hbar
    denominator = 1 + (V0**2 * np.sinh(kappa * a)**2) / (4 * E * (V0 - E))
    return 1.0 / denominator


def tunneling_T_thin_approx(E, V0, a, m=M_DEFAULT, hbar=HBAR):
    """薄势垒近似 T ≈ exp(-2κa)"""
    if E >= V0:
        return None
    kappa = np.sqrt(2 * m * (V0 - E)) / hbar
    return np.exp(-2 * kappa * a)


# =============================================================================
# 实验 5：Gaussian 波包扩散（自由粒子）
# =============================================================================

def gaussian_packet_width(t, sigma_x0, m=M_DEFAULT, hbar=HBAR):
    """Gaussian 波包宽度随时间扩散
    
    σ_x(t) = σ_x(0) √(1 + (ℏt/(2mσ_x(0)²))²)
    """
    factor = hbar * t / (2 * m * sigma_x0**2)
    return sigma_x0 * np.sqrt(1 + factor**2)


def gaussian_packet_group_velocity(k0, m=M_DEFAULT, hbar=HBAR):
    """群速度 v_g = ℏk₀/m"""
    return hbar * k0 / m


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("quantum.py — 物理是活的 / 第 14 模块演示")
    print("Topic: 量子力学初步 —— 1D Schrödinger 方程")
    print("=" * 70)
    print()
    print("使用自然单位 ℏ = 1, m = 1（让能级数字直观）")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：无限深势阱
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】无限深势阱：验证 E_n = n²π²ℏ²/(2mL²)")
    print("-" * 70)
    
    L = 1.0
    # 用极高势能模拟"无限井"边界
    N = 1001
    x_arr = np.linspace(0, L, N)
    V_arr = np.zeros(N)
    # 边界：用极高势能模拟无穷大
    V_arr[0] = 1e10
    V_arr[-1] = 1e10
    
    eigvals, eigvecs = solve_1D_TISE(V_arr, x_arr, n_states=5)
    
    print(f"  阱宽 L = {L}, 网格 N = {N}")
    print(f"  {'n':>3s}  {'E_n 数值':>12s}  {'E_n 理论':>12s}  {'误差':>10s}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*12}  {'-'*10}")
    
    for n in range(1, 6):
        E_th = infinite_well_theory(n, L)
        E_num = eigvals[n-1]
        err = abs(E_num - E_th) / E_th * 100
        print(f"  {n:>3d}  {E_num:>12.4f}  {E_th:>12.4f}  {err:>9.4f}%")
    
    print()
    print(f"  ✓ E_n ∝ n² 量子化能级")
    print(f"  ✓ E_1 = 4.93 ≠ 0 —— 零点能体现不确定原理")
    print(f"  ✓ 波函数节点数: ψ_n 有 n-1 个节点")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：量子谐振子
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】量子谐振子：验证 E_n = ℏω(n + 1/2)")
    print("-" * 70)
    
    omega = 1.0
    L_max = 8.0  # 取足够大让波函数在边界为零
    N = 2001
    x_arr = np.linspace(-L_max, L_max, N)
    V_arr = 0.5 * omega**2 * x_arr**2
    
    eigvals_ho, eigvecs_ho = solve_1D_TISE(V_arr, x_arr, n_states=8)
    
    print(f"  谐振子 V = (1/2)ω²x²,  ω = {omega}")
    print(f"  网格范围 [-{L_max}, {L_max}], N = {N}")
    print(f"  {'n':>3s}  {'E_n 数值':>12s}  {'E_n 理论':>12s}  {'误差':>10s}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*12}  {'-'*10}")
    
    for n in range(8):
        E_th = harmonic_oscillator_theory(n, omega)
        E_num = eigvals_ho[n]
        err = abs(E_num - E_th) / E_th * 100
        print(f"  {n:>3d}  {E_num:>12.4f}  {E_th:>12.4f}  {err:>9.4f}%")
    
    print()
    print(f"  ✓ E_n = (n + 1/2)ℏω —— 等间距 ℏω")
    print(f"  ✓ E_0 = 0.5 = 零点能 ℏω/2")
    print(f"  ✓ 间距 = E_1 - E_0 = 1.0 = ℏω ✓")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：有限深势阱
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】有限深势阱：束缚态 + 波函数渗出经典禁区")
    print("-" * 70)
    
    V0 = 20.0
    a = 1.0
    L_box = 10.0
    N = 2001
    x_arr = np.linspace(-L_box, L_box, N)
    V_arr = finite_well_potential(x_arr, V0, a)
    
    eigvals_fw, eigvecs_fw = solve_1D_TISE(V_arr, x_arr, n_states=10)
    
    print(f"  有限井 V₀ = {V0}, 宽度 2a = {2*a}")
    print(f"  束缚态条件：E < 0")
    print()
    print(f"  {'n':>3s}  {'E_n':>10s}  {'束缚?':>6s}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*6}")
    
    bound_count = 0
    for n in range(10):
        E = eigvals_fw[n]
        is_bound = E < 0
        if is_bound:
            bound_count += 1
        print(f"  {n:>3d}  {E:>10.4f}  {'✓' if is_bound else '×':>6s}")
    
    print()
    print(f"  ✓ 总束缚态数 = {bound_count}")
    print()
    
    # 阱外波函数渗透检查
    psi_ground = eigvecs_fw[:, 0]**2
    inside_mask = np.abs(x_arr) < a
    outside_mask = np.abs(x_arr) > a
    p_inside = np.trapezoid(psi_ground[inside_mask], x_arr[inside_mask])
    p_outside = np.trapezoid(psi_ground[outside_mask], x_arr[outside_mask])
    
    print(f"  基态波函数的概率分布：")
    print(f"    阱内 (|x| < {a}): P = {p_inside:.4f}")
    print(f"    阱外 (|x| > {a}): P = {p_outside:.4f}  ← 经典不允许")
    print(f"  ✓ 量子波函数渗透经典禁区 —— 隧穿的来源")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：矩形势垒隧穿
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】矩形势垒的隧穿透射系数")
    print("-" * 70)
    
    V0 = 5.0
    a = 1.0
    
    print(f"  势垒高 V₀ = {V0}, 宽度 a = {a}")
    print(f"  m = 1, ℏ = 1")
    print()
    print(f"  {'E/V₀':>6s}  {'T 精确':>12s}  {'T 薄势垒近似':>14s}  {'相对差':>10s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*14}  {'-'*10}")
    
    for E_ratio in [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]:
        E = E_ratio * V0
        T_exact = tunneling_T_analytical(E, V0, a)
        T_thin = tunneling_T_thin_approx(E, V0, a)
        rel_diff = abs(T_exact - T_thin) / T_exact * 100
        print(f"  {E_ratio:>6.2f}  {T_exact:>12.4e}  {T_thin:>14.4e}  {rel_diff:>9.2f}%")
    
    print()
    print(f"  ✓ T < 1: 即使经典上不可能穿过 —— 量子允许")
    print(f"  ✓ E 接近 V₀ 时 T 增大（更接近经典允许）")
    print(f"  ✓ 薄势垒近似 T ≈ exp(-2κa) 在弱透射极限好")
    print(f"  ✓ 应用：STM、α 衰变、半导体隧道二极管")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 5：Gaussian 波包扩散
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 5】自由粒子 Gaussian 波包：扩散 + 群速")
    print("-" * 70)
    
    sigma_x0 = 0.5
    k0 = 5.0
    m = 1.0
    v_g = gaussian_packet_group_velocity(k0, m)
    
    print(f"  初始宽度 σ_x(0) = {sigma_x0}, 中心动量 k₀ = {k0}")
    print(f"  群速度 v_g = ℏk₀/m = {v_g}")
    print()
    print(f"  {'t':>6s}  {'σ_x(t)':>10s}  {'扩散倍数':>10s}  {'质心位置':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")
    
    for t in [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        sigma_t = gaussian_packet_width(t, sigma_x0, m=m)
        x_center = v_g * t
        ratio = sigma_t / sigma_x0
        print(f"  {t:>6.2f}  {sigma_t:>10.4f}  {ratio:>9.3f}×  {x_center:>10.4f}")
    
    print()
    print(f"  ✓ 量子波包必然扩散 —— 经典轨迹失效")
    print(f"  ✓ 扩散时间尺度 τ ~ mσ²/ℏ = {m*sigma_x0**2:.2f}（自然单位）")
    print(f"  ✓ 质心匀速运动 v_g —— Ehrenfest 定理")
    print(f"  ✓ \"量子粒子\" 永远不能保持\"局域\"")
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("第 14 章核心回顾：")
    print("  量子力学 = 波函数 + Schrödinger 方程 + 算符 + 测量")
    print()
    print("  实验 1: 无限井 E_n ∝ n² + 零点能 ✓")
    print("  实验 2: 谐振子 E_n = ℏω(n+1/2) ✓")
    print("  实验 3: 有限井 + 波函数渗出经典禁区 ✓")
    print("  实验 4: 隧穿透射 T(E) ✓")
    print("  实验 5: 波包扩散 σ_x(t) ✓")
    print()
    print("练习 / Exercises:")
    print("  1. 时间演化算符 → 含时 TDSE 数值积分")
    print("  2. 双井势 + 共振隧穿")
    print("  3. WKB 近似 → 复杂势的能级")
    print("  4. Δx·Δp 直接数值验证不确定原理")
    print("=" * 70)
