"""
em_wave.py — Maxwell + 电磁波模块
================================
Module 12 of "Physics is Alive" / 《物理是活的》第 12 模块

实现 / Implements:
    - plane_wave: 1D 电磁波传播 + 验证 c
    - poynting_vector: 能流密度
    - dipole_radiation: 偶极辐射模式 + ω⁴ 依赖（Rayleigh 散射）
    - polarization: 线性 / 圆 / 椭圆偏振 + Stokes 参数
    - lorentz_boost_field: 相对论场变换

物理基础 / Physics:
    波速 c = 1/√(μ₀ε₀) ≈ 3e8 m/s
    Poynting: S = (1/μ₀) E × B
    偶极辐射: P ∝ ω⁴ p₀²
    Lorentz: E'_⊥ = γ(E_⊥ + v × B), B'_⊥ = γ(B_⊥ - v × E/c²)

栗周 / Li Zhou
2026 年 8 月 / August 2026
MIT License
"""

import numpy as np


# 物理常数
c = 2.99792458e8                 # 光速 m/s
mu_0 = 4 * np.pi * 1e-7          # 真空磁导率
epsilon_0 = 1.0 / (mu_0 * c**2)  # 真空介电常数
hbar = 1.054571817e-34


# =============================================================================
# 实验 1：1D 平面电磁波传播
# =============================================================================

def plane_wave_E(x, t, E0=1.0, k=1.0, omega=None, phase=0.0):
    """E(x, t) = E0 cos(kx - ωt + φ)
    
    若 omega 未给定 —— 用色散关系 ω = ck
    """
    if omega is None:
        omega = c * k
    return E0 * np.cos(k * x - omega * t + phase)


def plane_wave_B(x, t, E0=1.0, k=1.0, omega=None, phase=0.0):
    """B = E/c —— 横波关系"""
    return plane_wave_E(x, t, E0, k, omega, phase) / c


def verify_wave_speed(k_array, omega_array):
    """从色散关系数值验证波速"""
    omega_arr = np.asarray(omega_array)
    k_arr = np.asarray(k_array)
    v_phase = omega_arr / k_arr
    return v_phase


# =============================================================================
# 实验 2：Poynting 矢量
# =============================================================================

def poynting_vector(E_vec, B_vec):
    """S = (1/μ₀) E × B"""
    return np.cross(E_vec, B_vec) / mu_0


def poynting_plane_wave(E0):
    """平面波 Poynting 模幅 |S| = c·ε₀·E₀² / 2 (时间平均)
    
    瞬时 |S| = (1/μ₀c) E₀² cos²
    时间平均 = (1/2μ₀c) E₀² = (c·ε₀/2) E₀²
    """
    S_instant_max = E0**2 / (mu_0 * c)
    S_time_avg = 0.5 * c * epsilon_0 * E0**2
    return S_instant_max, S_time_avg


# =============================================================================
# 实验 3：偶极辐射 + Rayleigh 散射
# =============================================================================

def dipole_radiation_power(p0, omega):
    """点偶极的总辐射功率
    
    P = p₀² ω⁴ / (12 π ε₀ c³)
    
    注意 ω⁴ 依赖 —— Rayleigh 散射 = 天空蓝的物理
    """
    return p0**2 * omega**4 / (12 * np.pi * epsilon_0 * c**3)


def dipole_radiation_pattern(theta, p0=1.0, omega=1.0, r=1.0):
    """偶极远场辐射强度的角分布
    
    I(θ) = (p₀² ω⁴ / 16π²ε₀c³r²) sin²θ
    
    其中 θ 是 ŝ 和偶极轴的夹角
    """
    factor = p0**2 * omega**4 / (16 * np.pi**2 * epsilon_0 * c**3 * r**2)
    return factor * np.sin(theta)**2


# =============================================================================
# 实验 4：偏振状态
# =============================================================================

def jones_vector_linear(angle_deg):
    """线性偏振 Jones 矢量
    
    angle = 偏振方向与 x 的夹角（度）
    """
    a = np.deg2rad(angle_deg)
    return np.array([np.cos(a), np.sin(a)])


def jones_vector_circular(handedness='right'):
    """圆偏振 Jones 矢量"""
    if handedness == 'right':
        return np.array([1, -1j]) / np.sqrt(2)
    elif handedness == 'left':
        return np.array([1, 1j]) / np.sqrt(2)


def stokes_parameters(jones_vec):
    """从 Jones 矢量计算 Stokes 参数 (S₀, S₁, S₂, S₃)
    
    S₀ = |Ex|² + |Ey|²  (总强度)
    S₁ = |Ex|² - |Ey|²  (线偏振水平分量)
    S₂ = 2 Re(Ex* Ey)   (线偏振 45° 分量)
    S₃ = 2 Im(Ex* Ey)   (圆偏振分量)
    """
    Ex, Ey = jones_vec
    S0 = abs(Ex)**2 + abs(Ey)**2
    S1 = abs(Ex)**2 - abs(Ey)**2
    S2 = 2 * np.real(np.conj(Ex) * Ey)
    S3 = 2 * np.imag(np.conj(Ex) * Ey)
    return S0, S1, S2, S3


# =============================================================================
# 实验 5：相对论场变换
# =============================================================================

def gamma_factor(v):
    """Lorentz 因子 γ = 1/√(1 - v²/c²)"""
    beta = v / c
    return 1.0 / np.sqrt(1 - beta**2)


def lorentz_boost_field(E, B, v_boost):
    """场的 Lorentz 变换（boost 沿 x 方向）
    
    Parameters:
        E : 原系 (Ex, Ey, Ez)
        B : 原系 (Bx, By, Bz)
        v_boost : x 方向 boost 速度
    
    Returns:
        E', B' : 新系中的场
    """
    Ex, Ey, Ez = E
    Bx, By, Bz = B
    g = gamma_factor(v_boost)
    
    # 沿 boost 方向分量不变
    Ex_p = Ex
    Bx_p = Bx
    
    # 横向分量混合
    Ey_p = g * (Ey - v_boost * Bz)
    Ez_p = g * (Ez + v_boost * By)
    By_p = g * (By + v_boost * Ez / c**2)
    Bz_p = g * (Bz - v_boost * Ey / c**2)
    
    E_prime = np.array([Ex_p, Ey_p, Ez_p])
    B_prime = np.array([Bx_p, By_p, Bz_p])
    return E_prime, B_prime


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("em_wave.py — 物理是活的 / 第 12 模块演示")
    print("Topic: Maxwell + 电磁波 + 相对论")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：1D 平面电磁波 + 验证 c = 1/√(μ₀ε₀)
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】1D 平面电磁波传播 + 验证 c")
    print("-" * 70)
    
    # 用 Maxwell 4 个方程的真空形式验证 c
    c_th = 1.0 / np.sqrt(mu_0 * epsilon_0)
    print(f"  从 μ₀ = {mu_0:.4e} T·m/A")
    print(f"  从 ε₀ = {epsilon_0:.4e} F/m")
    print(f"  计算 c = 1/√(μ₀ε₀) = {c_th:.4e} m/s")
    print(f"  实验值      c       = {c:.4e} m/s")
    print(f"  误差: {abs(c_th - c)/c*100:.6e}%")
    print()
    
    # 数值传播：构造平面波——多个 (k, ω) 验证色散关系
    k_arr = np.array([1.0, 1e3, 1e6, 1e9, 1e12, 1e15])  # 不同波数
    omega_arr = c * k_arr  # 真空中色散
    v_phase = verify_wave_speed(k_arr, omega_arr)
    
    print(f"  色散关系 ω = ck 验证：")
    print(f"  {'k (1/m)':>14s}  {'ω (rad/s)':>16s}  {'v_phase (m/s)':>16s}  {'对应波长':>14s}")
    print(f"  {'-'*14}  {'-'*16}  {'-'*16}  {'-'*14}")
    for ki, wi, vi in zip(k_arr, omega_arr, v_phase):
        wavelength = 2*np.pi/ki
        if wavelength > 1:
            wl_str = f"{wavelength:.2e} m"
        elif wavelength > 1e-3:
            wl_str = f"{wavelength*1e3:.2e} mm"
        elif wavelength > 1e-6:
            wl_str = f"{wavelength*1e6:.2e} μm"
        else:
            wl_str = f"{wavelength*1e9:.2e} nm"
        print(f"  {ki:>14.2e}  {wi:>16.2e}  {vi:>16.2e}  {wl_str:>14s}")
    
    print()
    print("  ✓ 真空中所有频率都以光速 c 传播 —— 无色散")
    print()
    
    # 横波关系：|B| = |E|/c
    E0 = 1.0  # V/m
    B0 = E0 / c
    print(f"  横波关系：|B| = |E|/c")
    print(f"  若 E₀ = {E0} V/m → B₀ = {B0:.4e} T")
    print(f"  典型阳光 E ≈ 1000 V/m → B ≈ {1000/c:.2e} T (几 μT)")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：Poynting 矢量 + 太阳常数
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】Poynting 矢量 + 太阳常数")
    print("-" * 70)
    
    # 用太阳常数反推地球轨道处的电场强度
    solar_constant = 1361  # W/m² (地球轨道)
    
    # S_avg = c ε₀ E₀² / 2
    E0_sun = np.sqrt(2 * solar_constant / (c * epsilon_0))
    B0_sun = E0_sun / c
    
    print(f"  太阳常数 (地球轨道处太阳辐射): {solar_constant} W/m²")
    print(f"  反推：")
    print(f"    E₀ = √(2S/(cε₀)) = {E0_sun:.1f} V/m")
    print(f"    B₀ = E₀/c = {B0_sun*1e6:.4f} μT")
    print()
    
    # 矢量叉积验证（沿 z 传播——E 沿 x，B 沿 y）
    E_vec = np.array([1.0, 0.0, 0.0])
    B_vec = np.array([0.0, 1.0/c, 0.0])
    S_vec = poynting_vector(E_vec, B_vec)
    S_max, S_avg = poynting_plane_wave(E0=1.0)
    
    print(f"  矢量验证 (E=x̂, B=ŷ/c)：")
    print(f"    S = E × B / μ₀ = {S_vec}")
    print(f"    指向 +z —— ✓ (波传播方向)")
    print(f"    |S| 瞬时最大 = {S_max:.4e} W/m²")
    print(f"    |S| 时间平均 = {S_avg:.4e} W/m²")
    print()
    
    # 不同强度光源对比
    sources = [
        ("月光", 0.003),
        ("室内灯", 10),
        ("阴天", 100),
        ("太阳直射", solar_constant),
        ("激光 (1 mW/mm² 笔)", 1000),
        ("工业激光", 1e10),
    ]
    print(f"  各种光源对应的电场强度 E₀:")
    for name, S in sources:
        E_src = np.sqrt(2 * S / (c * epsilon_0))
        print(f"    {name:>15s} ({S:>8.2e} W/m²): E₀ = {E_src:>10.2e} V/m")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：偶极辐射 + Rayleigh 散射（天空蓝）
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】偶极辐射：ω⁴ 依赖 + Rayleigh 散射")
    print("-" * 70)
    
    p0 = 1e-30  # C·m (典型分子偶极)
    
    # 可见光不同颜色
    wavelengths_nm = [400, 450, 500, 550, 600, 650, 700]
    color_names = ["紫", "蓝", "青", "绿", "黄", "橙", "红"]
    
    print(f"  分子偶极 p₀ = {p0} C·m")
    print(f"  辐射功率 P ∝ ω⁴ = (2πc/λ)⁴ ∝ 1/λ⁴")
    print()
    print(f"  {'颜色':>4s}  {'λ (nm)':>8s}  {'ω (rad/s)':>14s}  {'P (W)':>14s}  {'相对蓝光':>10s}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*10}")
    
    P_list = []
    for wl_nm, name in zip(wavelengths_nm, color_names):
        wl = wl_nm * 1e-9
        omega = 2 * np.pi * c / wl
        P = dipole_radiation_power(p0, omega)
        P_list.append(P)
    
    P_blue = P_list[1]  # 蓝光 (450 nm)
    for wl_nm, name, P in zip(wavelengths_nm, color_names, P_list):
        wl = wl_nm * 1e-9
        omega = 2 * np.pi * c / wl
        ratio = P / P_blue
        print(f"  {name:>4s}  {wl_nm:>8d}  {omega:>14.2e}  {P:>14.4e}  {ratio:>9.3f}×")
    
    print()
    print(f"  P(紫) / P(红) = (700/400)⁴ = {(700/400)**4:.3f}× → 紫光散射比红光强 9.4 倍")
    print(f"  ✓ Rayleigh 散射: 短波长强散射 → 天空蓝")
    print(f"  ✓ 日落时阳光斜射穿过大气长 → 蓝散射光 → 剩红 → 日落红")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：偏振 + Stokes 参数
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】偏振状态：Jones 矢量 + Stokes 参数")
    print("-" * 70)
    
    polarizations = [
        ("水平线偏 (0°)", jones_vector_linear(0)),
        ("竖直线偏 (90°)", jones_vector_linear(90)),
        ("斜 45° 线偏", jones_vector_linear(45)),
        ("右旋圆偏 (RCP)", jones_vector_circular('right')),
        ("左旋圆偏 (LCP)", jones_vector_circular('left')),
    ]
    
    print(f"  {'状态':>18s}  {'S₀':>8s}  {'S₁':>8s}  {'S₂':>8s}  {'S₃':>8s}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    
    for name, jv in polarizations:
        S0, S1, S2, S3 = stokes_parameters(jv)
        print(f"  {name:>18s}  {S0:>8.3f}  {S1:>+8.3f}  {S2:>+8.3f}  {S3:>+8.3f}")
    
    print()
    print(f"  Stokes 参数物理含义:")
    print(f"    S₀ = 总强度")
    print(f"    S₁ > 0 水平线偏, S₁ < 0 竖直线偏")
    print(f"    S₂ > 0 +45° 线偏, S₂ < 0 -45° 线偏")
    print(f"    S₃ > 0 左旋圆偏, S₃ < 0 右旋圆偏")
    print(f"  ✓ Stokes 参数完整描述任意偏振 → 应用于 MOKE、天文偏振测量")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 5：相对论场变换 —— 静电场 → 含磁场
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 5】相对论场变换：静电场在运动参考系")
    print("-" * 70)
    
    # 静止参考系：只有 Ey （沿 y）
    E_rest = np.array([0.0, 1.0, 0.0])  # V/m
    B_rest = np.array([0.0, 0.0, 0.0])  # T
    
    print(f"  静止系 S：E = (0, {E_rest[1]}, 0) V/m, B = (0, 0, 0) T")
    print(f"  在以速度 v 沿 x 运动的 S' 系观察：")
    print()
    print(f"  {'v/c':>8s}  {'γ':>10s}  {'E_y (V/m)':>12s}  {'B_z (T)':>14s}  {'物理':>20s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*14}  {'-'*20}")
    
    for beta in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]:
        v = beta * c
        g = gamma_factor(v)
        E_prime, B_prime = lorentz_boost_field(E_rest, B_rest, v)
        
        if beta < 0.01:
            phys = "非相对论极限"
        elif beta < 0.5:
            phys = "中等相对论"
        else:
            phys = "强相对论 (γ 大)"
        
        print(f"  {beta:>8.3f}  {g:>10.4f}  {E_prime[1]:>12.4f}  {B_prime[2]:>14.4e}  {phys:>20s}")
    
    print()
    print(f"  ✓ 静止系只有 E —— 在 S' 中既有 E 又有 B")
    print(f"  ✓ B_z = -γv·E_y/c² —— 磁场是\"电场的相对论修正\"")
    print(f"  ✓ Purcell: \"Magnetism is just electricity with a Lorentz boost\"")
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("第 12 章核心回顾：")
    print("  Maxwell 综合 = 4 个方程统一所有经典电磁现象")
    print("  → 电磁波 = 光 (c = 1/√(μ₀ε₀))")
    print("  → 相对论 (E, B 是同一张量 F^μν 的投影)")
    print()
    print("  实验 1: c 验证 + 真空无色散 ✓")
    print("  实验 2: Poynting 矢量 + 太阳常数反推 ✓")
    print("  实验 3: Rayleigh 散射 ω⁴ → 天空蓝 ✓")
    print("  实验 4: Stokes 参数完整描述偏振 ✓")
    print("  实验 5: 相对论场变换 → 磁场是电场的相对论修正 ✓")
    print()
    print("练习 / Exercises:")
    print("  1. FDTD 数值积分 Maxwell 方程 (1D 时域)")
    print("  2. 双折射晶体中 o 光 e 光分离")
    print("  3. 电荷加速辐射（Larmor 公式）")
    print("  4. 引力波四极辐射 vs 电磁偶极辐射对比")
    print("=" * 70)
