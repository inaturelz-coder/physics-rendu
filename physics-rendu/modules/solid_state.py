"""
solid_state.py — 凝聚态入门模块
==================================
Module 18 of "打通物理脉络" / 第 18 模块

实现 / Implements:
    - tight_binding_1d: 1D 紧束缚能带
    - tight_binding_2d_square: 2D 方晶格能带
    - graphene_bands: 石墨烯能带 + Dirac 锥
    - dos_1d_tight_binding: 1D 态密度（van Hove 奇异）
    - fermi_surface_2d: 2D 费米面

物理基础 / Physics:
    Bloch: ψ_k(r) = e^{ik·r} u_k(r)
    1D 紧束缚: E(k) = E_0 - 2t cos(ka)
    石墨烯: E_± = ±t|f(k)|, Dirac 锥 at K
    DOS: g(E) = Σ δ(E - E_n(k))

栗周 / Li Zhou
2026 年 10 月 / October 2026
MIT License
"""

import numpy as np


# =============================================================================
# 1. 1D 紧束缚能带
# =============================================================================

def tight_binding_1d(k_arr, E0=0.0, t=1.0, a=1.0):
    """1D 紧束缚色散关系
    
    E(k) = E_0 - 2t cos(ka)
    
    Parameters:
        k_arr : 波数数组
        E0    : 原子轨道能量
        t     : 跳跃积分（hopping）
        a     : 晶格常数
    """
    return E0 - 2 * t * np.cos(k_arr * a)


def bandwidth_1d(t):
    """1D 紧束缚带宽 = 4t"""
    return 4 * t


# =============================================================================
# 2. 2D 方晶格紧束缚
# =============================================================================

def tight_binding_2d_square(kx, ky, t=1.0, a=1.0):
    """2D 方晶格紧束缚
    
    E(k) = -2t [cos(k_x a) + cos(k_y a)]
    
    BZ: [-π/a, π/a] × [-π/a, π/a]
    高对称点:
        Γ = (0, 0)
        X = (π/a, 0)
        M = (π/a, π/a)
    """
    return -2 * t * (np.cos(kx * a) + np.cos(ky * a))


def evaluate_2d_at_high_symmetry(t=1.0, a=1.0):
    """计算 2D 方晶格高对称点的能量"""
    return {
        'Γ (0, 0)': tight_binding_2d_square(0, 0, t, a),
        'X (π, 0)': tight_binding_2d_square(np.pi/a, 0, t, a),
        'M (π, π)': tight_binding_2d_square(np.pi/a, np.pi/a, t, a),
    }


# =============================================================================
# 3. 石墨烯能带
# =============================================================================

def graphene_dispersion(kx, ky, t=2.7, a=1.42e-10):
    """石墨烯 2 子格紧束缚色散
    
    E_±(k) = ±t |f(k)|
    f(k) = e^{i k_y a / √3} + 2 cos(k_x a / 2) e^{-i k_y a / (2√3)}
    
    a = C-C 键长 ≈ 1.42 Å
    晶格常数 a₀ = a√3
    Dirac 点位于 K = (4π/3a₀, 0) 等
    
    Parameters:
        kx, ky : 波数
        t      : 跳跃积分 (eV)
        a      : C-C 键长
    
    Returns:
        E_plus, E_minus : 上下能带
    """
    # 用 |f(k)|² 形式（数值更稳）：
    # |f|² = 1 + 4 cos²(k_x a √3 /2) + 4 cos(k_x a √3 /2) cos(3 k_y a /2)
    # 或者用 Wallace 1947 标准公式：
    a0 = a * np.sqrt(3)  # 晶格常数
    
    f_squared = (1 + 4 * np.cos(kx * a0 / 2)**2 +
                 4 * np.cos(kx * a0 / 2) * np.cos(ky * a0 * np.sqrt(3) / 2))
    f_squared = np.maximum(f_squared, 0)  # 数值保护
    abs_f = np.sqrt(f_squared)
    
    return t * abs_f, -t * abs_f


def graphene_dirac_velocity(t=2.7, a=1.42e-10):
    """石墨烯 Dirac 锥的费米速度
    
    v_F = (3/2) t a / ℏ ≈ 10⁶ m/s
    """
    hbar = 1.054571817e-34
    e = 1.602176634e-19
    return (3/2) * t * e * a / hbar  # t in eV → 转 Joule


# =============================================================================
# 4. 态密度 (DOS)
# =============================================================================

def dos_1d_tight_binding(E, t=1.0):
    """1D 紧束缚态密度
    
    g(E) = 1 / (π √(4t² - E²))   for |E| < 2t
         = 0                      otherwise
    
    在 E = ±2t 处发散 → van Hove 奇异
    """
    E_arr = np.asarray(E, dtype=float)
    mask = np.abs(E_arr) < 2 * t
    g = np.zeros_like(E_arr)
    g[mask] = 1 / (np.pi * np.sqrt(4 * t**2 - E_arr[mask]**2))
    return g


def dos_2d_square_numerical(E_arr, n_k=200, t=1.0):
    """2D 方晶格态密度（数值）"""
    kx = np.linspace(-np.pi, np.pi, n_k)
    ky = np.linspace(-np.pi, np.pi, n_k)
    KX, KY = np.meshgrid(kx, ky)
    E_grid = tight_binding_2d_square(KX, KY, t=t)
    E_flat = E_grid.flatten()
    
    # 直方图近似 DOS
    bins = np.linspace(E_arr.min(), E_arr.max(), len(E_arr) + 1)
    counts, _ = np.histogram(E_flat, bins=bins)
    g = counts / (n_k**2)  # 归一化到 BZ 大小
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return bin_centers, g


# =============================================================================
# 5. 2D 费米面（半填充时）
# =============================================================================

def fermi_surface_2d_square_half_filling(n_k=100, t=1.0):
    """2D 方晶格半填充时的费米面
    
    E_F = 0 → 满足 cos(kx) + cos(ky) = 0 的 k
    
    解：kx + ky = ±π (mod 2π) → 在 BZ 内是两条交叉的直线
    """
    kx = np.linspace(-np.pi, np.pi, n_k)
    ky = np.linspace(-np.pi, np.pi, n_k)
    KX, KY = np.meshgrid(kx, ky)
    E = tight_binding_2d_square(KX, KY, t=t)
    
    # 找接近 E = 0 的点
    on_fermi = np.abs(E) < 0.05 * t
    return KX[on_fermi], KY[on_fermi]


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("solid_state.py — 打通物理脉络 / 第 18 模块演示")
    print("Topic: 凝聚态入门 —— 能带、Bloch、紧束缚")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：1D 紧束缚能带
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】1D 紧束缚能带：E(k) = -2t cos(ka)")
    print("-" * 70)
    
    t = 1.0
    a = 1.0
    
    print(f"  跳跃积分 t = {t}, 晶格常数 a = {a}")
    print(f"  带宽 W = 4t = {bandwidth_1d(t)}")
    print()
    print(f"  {'ka (单位 π)':>12s}  {'k 物理':>12s}  {'E(k) 数值':>12s}  {'物理':>20s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*20}")
    
    for ka_pi in [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]:
        k = ka_pi * np.pi
        E = tight_binding_1d(np.array([k]), t=t)[0]
        if abs(ka_pi) < 0.01:
            phys = "Γ (带底)"
        elif abs(abs(ka_pi) - 1.0) < 0.01:
            phys = "BZ 边界 (带顶)"
        elif abs(abs(ka_pi) - 0.5) < 0.01:
            phys = "BZ 中点"
        else:
            phys = ""
        print(f"  {ka_pi:>+12.2f}  {k:>+12.4f}  {E:>+12.4f}  {phys:>20s}")
    
    print()
    print(f"  ✓ k = 0: E = -2t = {-2*t} (带底)")
    print(f"  ✓ k = ±π/a: E = +2t = {+2*t} (带顶)")
    print(f"  ✓ 余弦能带 + 带宽 = 4t")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：2D 方晶格能带 + 高对称点
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】2D 方晶格紧束缚 + 高对称点")
    print("-" * 70)
    
    high_sym = evaluate_2d_at_high_symmetry(t=t, a=a)
    
    print(f"  能带：E(k) = -2t [cos(k_x a) + cos(k_y a)]")
    print()
    print(f"  {'高对称点':>15s}  {'位置 (1/a)':>15s}  {'E (能量)':>12s}")
    print(f"  {'-'*15}  {'-'*15}  {'-'*12}")
    for name, E in high_sym.items():
        print(f"  {name:>15s}  {'-':>15s}  {E:>+12.4f}")
    
    print()
    print(f"  ✓ Γ 点：E = -4t (带底，正方对称中心)")
    print(f"  ✓ X 点：E = 0 (中间)")
    print(f"  ✓ M 点：E = +4t (带顶，BZ 角)")
    print(f"  ✓ 2D 方晶格带宽 = 8t = {8*t}")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：石墨烯 Dirac 锥
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】石墨烯能带 + Dirac 锥")
    print("-" * 70)
    
    t_graphene = 2.7  # eV
    a_CC = 1.42e-10   # m
    
    v_F = graphene_dirac_velocity(t_graphene, a_CC)
    
    print(f"  石墨烯 (C-C 距离 = 1.42 Å)")
    print(f"  跳跃积分 t = {t_graphene} eV")
    print(f"  Dirac 锥的费米速度 v_F = {v_F:.3e} m/s ≈ c/{2.998e8/v_F:.0f}")
    print()
    
    # 计算几个特征点
    a0 = a_CC * np.sqrt(3)  # 晶格常数
    
    # Γ 点
    E_plus_G, E_minus_G = graphene_dispersion(0, 0, t_graphene, a_CC)
    # K 点 = (4π/(3a₀), 0)
    K_kx = 4 * np.pi / (3 * a0)
    E_plus_K, E_minus_K = graphene_dispersion(K_kx, 0, t_graphene, a_CC)
    # M 点 = (π/a₀, π/(√3 a₀))
    M_kx = np.pi / a0
    M_ky = np.pi / (np.sqrt(3) * a0)
    E_plus_M, E_minus_M = graphene_dispersion(M_kx, M_ky, t_graphene, a_CC)
    
    print(f"  关键高对称点能量：")
    print(f"  {'点':>10s}  {'E_+ (eV)':>12s}  {'E_- (eV)':>12s}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}")
    print(f"  {'Γ':>10s}  {E_plus_G:>+12.4f}  {E_minus_G:>+12.4f}")
    print(f"  {'K (Dirac)':>10s}  {E_plus_K:>+12.4f}  {E_minus_K:>+12.4f}")
    print(f"  {'M':>10s}  {E_plus_M:>+12.4f}  {E_minus_M:>+12.4f}")
    
    print()
    print(f"  ✓ Γ 点：E_± = ±3t = ±{3*t_graphene} eV (带宽最大)")
    print(f"  ✓ K 点：E_± = 0 → 上下能带在此相接！")
    print(f"  ✓ K 点附近 → 线性色散 E ≈ ±v_F·|q| → \"无质量 Dirac 费米子\"")
    print(f"  ✓ 这就是石墨烯 \"二维 Dirac 物质\" 的源头")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：1D 态密度（van Hove 奇异）
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】1D 紧束缚态密度（van Hove 奇异）")
    print("-" * 70)
    
    E_test = np.array([-2.5, -2.0, -1.99, -1.0, 0.0, 1.0, 1.99, 2.0, 2.5])
    g_test = dos_1d_tight_binding(E_test, t=1.0)
    
    print(f"  公式: g(E) = 1/(π √(4t² - E²))  for |E| < 2t")
    print()
    print(f"  {'E':>8s}  {'g(E)':>12s}  {'物理':>30s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*30}")
    
    for E, g in zip(E_test, g_test):
        if g < 1e-10:
            phys = "带外（无态）"
        elif np.abs(np.abs(E) - 2.0) < 0.02:
            phys = "← van Hove 奇异（发散）"
        elif np.abs(E) < 0.01:
            phys = "带中心（极小态密度）"
        else:
            phys = ""
        print(f"  {E:>+8.2f}  {g:>12.4f}  {phys:>30s}")
    
    print()
    print(f"  ✓ |E| > 2t: g(E) = 0 (没有态)")
    print(f"  ✓ |E| → 2t: g(E) → ∞ (van Hove 奇异)")
    print(f"  ✓ 这种奇异是 1D 体系特有 (2D 是对数发散，3D 是平方根)")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 5：2D 费米面（半填充）
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 5】2D 方晶格半填充 → 费米面")
    print("-" * 70)
    
    print(f"  半填充：每个 k 上 1 个电子 / 总共 2 个态（自旋）→ 占据 1/2")
    print(f"  E_F = 0 → 费米面方程：cos(kx) + cos(ky) = 0")
    print(f"  → kx + ky = ±π (mod 2π) → 两条交叉直线")
    print()
    
    kx_FS, ky_FS = fermi_surface_2d_square_half_filling()
    
    print(f"  费米面点数（接近 E=0）: {len(kx_FS)}")
    print(f"  典型几个点：")
    if len(kx_FS) > 0:
        n_show = min(8, len(kx_FS))
        idx_show = np.linspace(0, len(kx_FS) - 1, n_show).astype(int)
        for i in idx_show:
            print(f"    (kx, ky) = ({kx_FS[i]:+.3f}, {ky_FS[i]:+.3f})  |kx+ky| = {abs(kx_FS[i] + ky_FS[i]):.3f}")
    
    print()
    print(f"  ✓ 半填充的方格子有\"完美 nesting\"（费米面是直线）")
    print(f"  ✓ 这种 nesting 容易引起密度波相变（高 Tc 母体常见）")
    print(f"  ✓ 现实中：铜氧化物超导的母体 = 半填充正方反铁磁绝缘体")
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("第 18 章核心回顾：")
    print("  凝聚态入门 = 晶格 + 倒空间 + Bloch + 能带")
    print()
    print("  实验 1: 1D 紧束缚 E = -2t cos(ka) ✓")
    print("  实验 2: 2D 方晶格高对称点 ✓")
    print("  实验 3: 石墨烯 Dirac 锥（K 点能隙=0）✓")
    print("  实验 4: 1D van Hove 奇异 ✓")
    print("  实验 5: 2D 半填充费米面 nesting ✓")
    print()
    print("练习 / Exercises:")
    print("  1. 加次近邻跳跃 t' → 石墨烯打开能隙 (h-BN)")
    print("  2. 周期势 V cos(2πx/a) → 微扰算近自由电子能带")
    print("  3. 三角格子紧束缚 + 自旋阻挫")
    print("  4. 1D 二聚化 → SSH 模型（拓扑入门）")
    print("=" * 70)
