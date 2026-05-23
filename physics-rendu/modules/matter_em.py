"""
matter_em.py — 物质中的电磁模块
================================
Module 13 of "Physics is Alive" / 《物理是活的》第 13 模块

实现 / Implements:
    - drude_dielectric: Drude 模型介电函数
    - lorentz_dielectric: Lorentz 振子介电函数
    - fresnel_coefficients: 反射 + 透射系数
    - london_penetration: 超导穿透深度
    - kramers_kronig: KK 变换数值验证

物理基础 / Physics:
    Drude: ε(ω) = 1 - ω_p²/(ω² + iω/τ)
    Lorentz: ε(ω) = 1 + ω_p²/(ω_0² - ω² - iωγ)
    Fresnel: r = (n₁ - n₂)/(n₁ + n₂)
    London: B(z) = B₀ exp(-z/λ_L)
    KK: ε₁(ω) - 1 = (2/π) P∫ ω' ε₂(ω')/(ω'² - ω²) dω'

栗周 / Li Zhou
2026 年 8 月 / August 2026
MIT License
"""

import numpy as np


# 物理常数
hbar = 1.054571817e-34
e_charge = 1.602176634e-19
m_e = 9.1093837e-31
epsilon_0 = 8.8541878128e-12
c = 2.99792458e8
mu_0 = 4 * np.pi * 1e-7


# =============================================================================
# Drude 介电函数
# =============================================================================

def drude_epsilon(omega, omega_p, tau):
    """Drude 模型介电函数
    
    ε(ω) = 1 - ω_p² / (ω² + iω/τ)
    
    Parameters:
        omega   : 频率 (rad/s)
        omega_p : 等离子体频率 (rad/s)
        tau     : 弛豫时间 (s)
    
    Returns:
        eps : 复介电常数
    """
    return 1.0 - omega_p**2 / (omega**2 + 1j * omega / tau)


def plasma_frequency_from_density(n_e, m_eff=None):
    """从电子密度计算等离子体频率
    
    ω_p = √(n_e e² / (m_eff ε₀))
    """
    if m_eff is None:
        m_eff = m_e
    return np.sqrt(n_e * e_charge**2 / (m_eff * epsilon_0))


# =============================================================================
# Lorentz 模型
# =============================================================================

def lorentz_epsilon(omega, omega_0, gamma, omega_p, eps_inf=1.0):
    """Lorentz 振子介电函数
    
    ε(ω) = ε_∞ + ω_p² / (ω_0² - ω² - iωγ)
    
    Parameters:
        omega   : 频率
        omega_0 : 共振频率
        gamma   : 阻尼率
        omega_p : 等离子体频率（振子强度）
        eps_inf : 高频极限介电
    """
    return eps_inf + omega_p**2 / (omega_0**2 - omega**2 - 1j * omega * gamma)


# =============================================================================
# 复折射率 + 反射率
# =============================================================================

def complex_refractive_index(epsilon):
    """从介电函数算复折射率 ñ = √ε
    
    Returns:
        n + iκ
    """
    return np.sqrt(epsilon)


def reflectance_normal(epsilon, n_outside=1.0):
    """垂直入射反射率
    
    R = |(n_outside - ñ) / (n_outside + ñ)|²
    """
    n_tilde = complex_refractive_index(epsilon)
    r = (n_outside - n_tilde) / (n_outside + n_tilde)
    return np.abs(r)**2


def fresnel_coefficients(n1, n2, theta_i):
    """完整 Fresnel 反射 + 透射系数
    
    Parameters:
        n1, n2 : 两介质折射率（可复数）
        theta_i : 入射角（弧度）
    
    Returns:
        r_s, r_p, t_s, t_p
    """
    sin_t = (n1 / n2) * np.sin(theta_i)
    cos_t = np.sqrt(1 - sin_t**2 + 0j)
    cos_i = np.cos(theta_i)
    
    # s 偏振（垂直入射面）
    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    t_s = 2 * n1 * cos_i / (n1 * cos_i + n2 * cos_t)
    
    # p 偏振（平行入射面）
    r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    t_p = 2 * n1 * cos_i / (n2 * cos_i + n1 * cos_t)
    
    return r_s, r_p, t_s, t_p


def brewster_angle(n1, n2):
    """布鲁斯特角（无 p 偏振反射）"""
    return np.arctan(n2 / n1)


# =============================================================================
# London 穿透
# =============================================================================

def london_penetration_depth(n_s, m_eff=None):
    """London 穿透深度
    
    λ_L = √(m / (μ₀ n_s e²))
    """
    if m_eff is None:
        m_eff = m_e
    return np.sqrt(m_eff / (mu_0 * n_s * e_charge**2))


def meissner_field(z, B0, lambda_L):
    """超导内磁场分布（半无限超导体）
    
    B(z) = B₀ exp(-z/λ_L)
    """
    return B0 * np.exp(-z / lambda_L)


# =============================================================================
# Kramers-Kronig 关系
# =============================================================================

def kramers_kronig_eps1_from_eps2(omega_arr, eps2_arr):
    """用 KK 关系从 ε₂(ω) 计算 ε₁(ω) - 1
    
    ε₁(ω) - 1 = (2/π) P ∫₀^∞ ω' ε₂(ω') / (ω'² - ω²) dω'
    
    用主值积分（避开奇点）
    """
    omega_arr = np.asarray(omega_arr)
    eps2_arr = np.asarray(eps2_arr)
    n = len(omega_arr)
    eps1_minus_one = np.zeros(n)
    
    for i in range(n):
        omega = omega_arr[i]
        # 避开 ω' = ω 的奇点
        denominator = omega_arr**2 - omega**2
        # 主值：将 denominator 中接近 0 的项排除
        mask = np.abs(denominator) > 1e-10 * omega**2
        if mask.sum() < 2:
            eps1_minus_one[i] = 0
            continue
        integrand = omega_arr[mask] * eps2_arr[mask] / denominator[mask]
        # 梯形积分
        eps1_minus_one[i] = (2.0 / np.pi) * np.trapezoid(integrand, omega_arr[mask])
    
    return eps1_minus_one


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("matter_em.py — 物理是活的 / 第 13 模块演示")
    print("Topic: 物质中的电磁 —— 凝聚态视角")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：Drude 介电函数 + 金属反射率
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】Drude 介电函数 + 金属等离激元")
    print("-" * 70)
    
    # 银 (Ag): ω_p ≈ 9 eV, τ ≈ 1e-14 s
    metals = [
        ("Al", 15.8, 8e-15),
        ("Ag", 9.0, 1e-14),
        ("Au", 9.0, 1e-14),
        ("Cu", 7.4, 6.9e-15),
    ]
    
    print(f"  金属反射率（垂直入射）：")
    print(f"  {'金属':>4s}  {'ℏω_p (eV)':>10s}  {'λ_p (nm)':>10s}  "
          f"{'R(可见 2.5eV)':>14s}  {'R(UV 8eV)':>12s}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*14}  {'-'*12}")
    
    for name, hwp_eV, tau in metals:
        omega_p = hwp_eV * e_charge / hbar
        # 可见光 2.5 eV
        omega_vis = 2.5 * e_charge / hbar
        eps_vis = drude_epsilon(omega_vis, omega_p, tau)
        R_vis = reflectance_normal(eps_vis)
        
        # UV 8 eV
        omega_uv = 8.0 * e_charge / hbar
        eps_uv = drude_epsilon(omega_uv, omega_p, tau)
        R_uv = reflectance_normal(eps_uv)
        
        lambda_p_nm = 2 * np.pi * c / omega_p * 1e9
        print(f"  {name:>4s}  {hwp_eV:>10.1f}  {lambda_p_nm:>10.1f}  "
              f"{R_vis:>14.4f}  {R_uv:>12.4f}")
    
    print()
    print(f"  ✓ ω < ω_p: 金属高反射（镜面）")
    print(f"  ✓ ω > ω_p: 金属变透明")
    print(f"  ✓ Al 在 UV 也反射（ω_p 高）→ 用于紫外光学")
    print()
    
    # 等离子体频率与电子密度
    print(f"  从电子密度推 ω_p:")
    n_metals = [
        ("Cu", 8.49e28),
        ("Al", 1.81e29),
        ("Au", 5.90e28),
    ]
    for name, ne in n_metals:
        wp = plasma_frequency_from_density(ne)
        hwp = hbar * wp / e_charge
        print(f"    {name}: n_e = {ne:.2e} /m³ → ℏω_p = {hwp:.2f} eV")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：Lorentz 振子 + 共振吸收
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】Lorentz 振子模型 —— 共振吸收 + 色散")
    print("-" * 70)
    
    # 模拟一个共振峰：ω_0 = 4 eV (UV)
    omega_0 = 4.0 * e_charge / hbar
    gamma = 0.1 * e_charge / hbar
    omega_p_Lor = 5.0 * e_charge / hbar
    
    print(f"  共振频率 ℏω_0 = 4.0 eV (UV)")
    print(f"  阻尼 ℏγ = 0.1 eV")
    print(f"  振子强度 ℏω_p = 5.0 eV")
    print()
    print(f"  {'ℏω (eV)':>8s}  {'ε_1':>10s}  {'ε_2':>10s}  {'n':>8s}  {'κ':>8s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    
    for hwn in [1.0, 2.0, 3.0, 3.5, 3.9, 4.0, 4.1, 4.5, 6.0, 10.0]:
        omega = hwn * e_charge / hbar
        eps = lorentz_epsilon(omega, omega_0, gamma, omega_p_Lor)
        n_tilde = complex_refractive_index(eps)
        print(f"  {hwn:>8.2f}  {eps.real:>10.4f}  {eps.imag:>10.4f}  "
              f"{n_tilde.real:>8.4f}  {n_tilde.imag:>8.4f}")
    
    print()
    print(f"  ✓ ω → 0: ε₁ 趋于静态介电常数（高）")
    print(f"  ✓ ω = ω_0: ε₂ 峰值 → 共振吸收最强")
    print(f"  ✓ ω >> ω_0: ε → 1 (高频透明)")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：Fresnel 反射 + 布鲁斯特角
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】Fresnel 系数 + 布鲁斯特角")
    print("-" * 70)
    
    n1, n2 = 1.0, 1.5  # 空气 → 玻璃
    theta_B = brewster_angle(n1, n2)
    
    print(f"  入射介质 n₁ = {n1} (空气)")
    print(f"  目标介质 n₂ = {n2} (玻璃)")
    print(f"  布鲁斯特角 θ_B = arctan(n₂/n₁) = {np.degrees(theta_B):.2f}°")
    print()
    print(f"  {'θ_i (°)':>8s}  {'|r_s|²':>10s}  {'|r_p|²':>10s}  {'物理':>20s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*20}")
    
    for theta_deg in [0, 30, 50, np.degrees(theta_B), 70, 89]:
        theta = np.deg2rad(theta_deg)
        r_s, r_p, _, _ = fresnel_coefficients(n1, n2, theta)
        R_s = np.abs(r_s)**2
        R_p = np.abs(r_p)**2
        
        if abs(theta_deg - np.degrees(theta_B)) < 0.5:
            phys = "← 布鲁斯特角"
        elif theta_deg < 30:
            phys = "近垂直"
        elif theta_deg > 80:
            phys = "近全反射"
        else:
            phys = "中等角度"
        
        print(f"  {theta_deg:>8.2f}  {R_s:>10.4f}  {R_p:>10.4f}  {phys:>20s}")
    
    print()
    print(f"  ✓ 在 θ_B 处 R_p = 0 → p 偏振完全透射")
    print(f"  ✓ 这就是偏光太阳镜原理（滤掉水平偏振反射光）")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：London 穿透 + Meissner 效应
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】超导体的 London 穿透深度 + Meissner 效应")
    print("-" * 70)
    
    superconductors = [
        ("Pb (铅)", 1.5e29),
        ("Al (铝)", 1.81e29),
        ("Nb (铌)", 5.56e28),
        ("YBCO", 5e27),
    ]
    
    print(f"  London 穿透深度 λ_L = √(m / (μ₀ n_s e²))")
    print()
    print(f"  {'超导体':>10s}  {'n_s (/m³)':>14s}  {'λ_L (nm)':>10s}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*10}")
    
    for name, n_s in superconductors:
        lam = london_penetration_depth(n_s)
        print(f"  {name:>10s}  {n_s:>14.2e}  {lam*1e9:>10.2f}")
    
    print()
    
    # Meissner: 1D 衰减
    print(f"  Meissner 效应（外磁场 B₀ = 1 T）：")
    lambda_L_Nb = london_penetration_depth(5.56e28)
    print(f"  Nb 中 λ_L = {lambda_L_Nb*1e9:.1f} nm")
    print()
    print(f"  {'深度 z (nm)':>12s}  {'B(z) (T)':>12s}  {'B(z)/B₀':>10s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*10}")
    
    for z_nm in [0, 10, 50, 100, 200, 500, 1000]:
        z = z_nm * 1e-9
        B = meissner_field(z, 1.0, lambda_L_Nb)
        ratio = B / 1.0
        print(f"  {z_nm:>12.0f}  {B:>12.4e}  {ratio:>10.4f}")
    
    print()
    print(f"  ✓ 磁场在超导内指数衰减 → Meissner 完美抗磁")
    print(f"  ✓ U(1) 规范对称破缺的体现 (Ch11 相变语言)")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 5：Kramers-Kronig 关系验证
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 5】Kramers-Kronig 关系验证")
    print("-" * 70)
    
    # 用 Lorentz 模型生成 ε₂，用 KK 推 ε₁，对比解析值
    omega_grid = np.linspace(0.1, 20, 500) * e_charge / hbar
    omega_0_KK = 5.0 * e_charge / hbar
    gamma_KK = 0.3 * e_charge / hbar
    omega_p_KK = 3.0 * e_charge / hbar
    
    eps_arr = lorentz_epsilon(omega_grid, omega_0_KK, gamma_KK, omega_p_KK)
    eps1_analytic = eps_arr.real - 1  # ε₁ - 1
    eps2_arr = eps_arr.imag
    
    # KK 数值推导
    eps1_KK = kramers_kronig_eps1_from_eps2(omega_grid, eps2_arr)
    
    print(f"  用 Lorentz 模型 ε(ω) 生成 ε₂ → 用 KK 反推 ε₁ - 1 → 与解析比较")
    print()
    print(f"  {'ℏω (eV)':>8s}  {'ε₁-1 解析':>12s}  {'ε₁-1 KK 数值':>14s}  {'差':>10s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*14}  {'-'*10}")
    
    sample_idx = [10, 50, 100, 150, 200, 250, 300, 400]
    for idx in sample_idx:
        hwn = hbar * omega_grid[idx] / e_charge
        diff = eps1_analytic[idx] - eps1_KK[idx]
        print(f"  {hwn:>8.2f}  {eps1_analytic[idx]:>12.4f}  "
              f"{eps1_KK[idx]:>14.4f}  {diff:>+10.4f}")
    
    print()
    print(f"  ✓ KK 数值结果接近解析值（有限离散积分误差）")
    print(f"  ✓ KK 关系是因果性 + 解析性的数学结果")
    print(f"  ✓ 实验中：测反射率 → KK → ε₁, ε₂ → 完整光学响应")
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("第 13 章核心回顾：")
    print("  物质中的电磁 = ε(ω, k) + μ(ω) 浓缩一切")
    print("  Drude + Lorentz 模型 → 经典电磁的微观图像")
    print("  ARPES + MOKE + Raman = 凝聚态实验的眼睛")
    print()
    print("  实验 1: Drude → 金属反射 + 等离激元 ✓")
    print("  实验 2: Lorentz → 共振吸收 + 色散 ✓")
    print("  实验 3: Fresnel + 布鲁斯特角 ✓")
    print("  实验 4: London + Meissner 效应 ✓")
    print("  实验 5: Kramers-Kronig 关系 ✓")
    print()
    print("练习 / Exercises:")
    print("  1. 拟合实测金属反射率得到 ω_p, τ")
    print("  2. 用多个 Lorentz 振子拟合 SiO₂ 光谱")
    print("  3. 模拟超导反射率谱看 2Δ 边")
    print("  4. KK 关系应用于真实材料光谱")
    print("=" * 70)
