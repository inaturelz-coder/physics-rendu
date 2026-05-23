"""
quantum_stat.py — 量子统计模块
================================
Module 8 of "Physics is Alive" / 《物理是活的》第 8 模块

实现 / Implements:
    - 三种分布：Bose-Einstein, Fermi-Dirac, Maxwell-Boltzmann
    - 自由费米气：费米能、Sommerfeld 比热、相对论极限
    - BEC：临界温度估算（铷、钠典型实验）
    - Brillouin 函数：自旋系统磁化
    - Debye 模型：固体比热

物理基础 / Physics:
    BE: <n> = 1/(exp((ε-μ)/kT) - 1)
    FD: <n> = 1/(exp((ε-μ)/kT) + 1)
    费米能: ε_F = ℏ²(3π²n)^(2/3)/(2m)
    BEC: T_c = 2πℏ²/(mk_B) (n/ζ(3/2))^(2/3)
    Debye: C_V = 9Nk_B (T/Θ_D)³ ∫₀^(Θ_D/T) x⁴eˣ/(eˣ-1)² dx

栗周 / Li Zhou
2026 年 7 月 / July 2026
MIT License
"""

import numpy as np
from scipy.integrate import quad


# 基本物理常数
hbar = 1.054571817e-34   # 约化普朗克常数 J·s
k_B = 1.380649e-23       # Boltzmann 常数 J/K
m_e = 9.1093837e-31      # 电子质量 kg
amu = 1.66053907e-27     # 原子质量单位 kg
e_charge = 1.602176634e-19  # 元电荷 C
eV = e_charge            # 1 eV = 1.602e-19 J
mu_B = 9.2740100783e-24  # Bohr 磁子 J/T
ZETA_32 = 2.6123753486   # Riemann ζ(3/2)


# =============================================================================
# 三种统计分布
# =============================================================================

def bose_einstein(epsilon, mu, T):
    """Bose-Einstein 分布
    
    要求 epsilon > mu，否则发散
    """
    x = (epsilon - mu) / (k_B * T)
    return 1.0 / (np.exp(x) - 1.0)


def fermi_dirac(epsilon, mu, T):
    """Fermi-Dirac 分布"""
    x = (epsilon - mu) / (k_B * T)
    return 1.0 / (np.exp(x) + 1.0)


def maxwell_boltzmann(epsilon, mu, T):
    """Maxwell-Boltzmann（经典极限）"""
    x = (epsilon - mu) / (k_B * T)
    return np.exp(-x)


# =============================================================================
# 自由费米气
# =============================================================================

def fermi_energy_3D(n, m=m_e):
    """3D 自由费米气的费米能 (J)"""
    return (hbar**2 / (2 * m)) * (3 * np.pi**2 * n)**(2/3)


def fermi_temperature(n, m=m_e):
    """费米温度 T_F = ε_F / k_B"""
    return fermi_energy_3D(n, m) / k_B


def fermi_velocity(n, m=m_e):
    """费米速度 v_F = ℏ k_F / m"""
    k_F = (3 * np.pi**2 * n)**(1/3)
    return hbar * k_F / m


def sommerfeld_specific_heat(T, n, m=m_e):
    """Sommerfeld 比热（电子贡献）
    
    C_V^el = (π²/2) (T/T_F) N k_B
    返回每电子的比热
    """
    T_F = fermi_temperature(n, m)
    return (np.pi**2 / 2) * (T / T_F) * k_B


# =============================================================================
# BEC 临界温度
# =============================================================================

def bec_critical_temperature(n, m):
    """3D 自由玻色气的 BEC 临界温度
    
    T_c = (2πℏ²)/(m k_B) × (n / ζ(3/2))^(2/3)
    """
    return (2 * np.pi * hbar**2 / (m * k_B)) * (n / ZETA_32)**(2/3)


def thermal_de_broglie(T, m):
    """热德布罗意波长 λ_T = h/√(2π m k_B T)"""
    return np.sqrt(2 * np.pi * hbar**2 / (m * k_B * T))


# =============================================================================
# Brillouin 函数（自旋系统）
# =============================================================================

def brillouin_function(x, J=0.5):
    """Brillouin 函数 B_J(x)
    
    自旋 J 系统在 x = g μ_B J B / (k_B T) 下的磁化
    特例：J = 1/2 → tanh(x)
    """
    if J == 0.5:
        return np.tanh(x)
    coth = lambda y: 1.0 / np.tanh(y)
    return ((2*J + 1) / (2*J)) * coth((2*J + 1) * x / (2*J)) - \
           (1.0 / (2*J)) * coth(x / (2*J))


def paramagnet_magnetization(B, T, n_spins, g=2.0, J=0.5):
    """顺磁体磁化（n_spins 个独立自旋 J）
    
    M = n_spins × g μ_B J × B_J(x), 其中 x = g μ_B J B / k_B T
    """
    x = g * mu_B * J * B / (k_B * T)
    return n_spins * g * mu_B * J * brillouin_function(x, J)


# =============================================================================
# Debye 模型
# =============================================================================

def debye_integrand(x):
    """Debye 积分的被积函数"""
    if x < 1e-10:
        return 0.0
    if x > 700:  # 防止溢出
        return 0.0
    e_x = np.exp(x)
    return x**4 * e_x / (e_x - 1.0)**2


def debye_specific_heat(T, T_Debye, n_atoms=6.022e23):
    """Debye 模型摩尔比热
    
    C_V = 9 N k_B (T/T_D)³ ∫₀^(T_D/T) x⁴eˣ/(eˣ-1)² dx
    
    极限：
    - T << T_D : C_V → 12π⁴/5 × N k_B (T/T_D)³  (T³ 律)
    - T >> T_D : C_V → 3 N k_B  (Dulong-Petit)
    """
    x_max = T_Debye / T
    integral, _ = quad(debye_integrand, 0, x_max, limit=200)
    return 9 * n_atoms * k_B * (T / T_Debye)**3 * integral


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("quantum_stat.py — 物理是活的 / 第 8 模块演示")
    print("Topic: 量子统计 —— 凝聚态的语言")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：三种分布对比
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】三种分布对比 (μ = 0)")
    print("-" * 70)
    
    T = 100  # K
    eps_ratios = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    print(f"  T = {T} K, μ = 0")
    print()
    print(f"  {'ε/kT':>6s}  {'Bose':>10s}  {'Fermi':>10s}  {'MB':>10s}  {'物理':>20s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*20}")
    
    for r in eps_ratios:
        eps = r * k_B * T
        n_be = bose_einstein(eps, 0, T)
        n_fd = fermi_dirac(eps, 0, T)
        n_mb = maxwell_boltzmann(eps, 0, T)
        
        if r < 1:
            phys = "量子区"
        elif r < 3:
            phys = "过渡"
        else:
            phys = "经典极限"
        
        print(f"  {r:>6.1f}  {n_be:>10.4f}  {n_fd:>10.4f}  {n_mb:>10.4f}  {phys:>20s}")
    
    print()
    print("  ✓ 高能极限 (ε/kT >> 1)：三种分布完全重合 (经典)")
    print("  ✓ 低能极限：BE >> MB >> FD (量子修正显著)")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：自由电子气 (铜)
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】自由电子气（铜，n = 8.49×10²⁸ /m³）")
    print("-" * 70)
    
    n_Cu = 8.49e28  # 铜的电子密度
    
    eps_F = fermi_energy_3D(n_Cu)
    T_F = fermi_temperature(n_Cu)
    v_F = fermi_velocity(n_Cu)
    
    print(f"  电子密度 n = {n_Cu:.2e} /m³")
    print(f"  费米能 ε_F = {eps_F/eV:.3f} eV")
    print(f"  费米温度 T_F = {T_F:.0f} K  ← 远高于室温！")
    print(f"  费米速度 v_F = {v_F:.2e} m/s")
    print(f"  v_F / c = {v_F/3e8:.4f}  (非相对论)")
    print()
    print(f"  室温 (300 K) 下：")
    print(f"    T / T_F = {300/T_F:.4f}  ← 高度简并的费米气")
    
    C_V_classical = 1.5 * k_B  # 经典预测每电子 3/2 k_B
    C_V_quantum = sommerfeld_specific_heat(300, n_Cu)
    
    print(f"    经典预测每电子 C_V = 3/2 k_B = {C_V_classical:.2e} J/K")
    print(f"    量子预测每电子 C_V = (π²/2)(T/T_F) k_B = {C_V_quantum:.2e} J/K")
    print(f"    量子/经典 = {C_V_quantum/C_V_classical:.4f}  ← 经典灾难被自然解决")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：BEC 临界温度
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】BEC 临界温度（铷-87 实验典型条件）")
    print("-" * 70)
    
    # 铷-87
    m_Rb87 = 87 * amu
    n_BEC = 1e20  # /m³ (典型激光冷却原子云密度)
    
    T_c_Rb = bec_critical_temperature(n_BEC, m_Rb87)
    
    print(f"  ⁸⁷Rb 原子，n = {n_BEC:.0e} /m³")
    print(f"  T_c = {T_c_Rb*1e6:.3f} μK")
    print(f"  T_c 比室温低 {300/T_c_Rb:.2e} 倍")
    print()
    
    # 钠
    m_Na23 = 23 * amu
    T_c_Na = bec_critical_temperature(n_BEC, m_Na23)
    print(f"  ²³Na 原子，n = {n_BEC:.0e} /m³")
    print(f"  T_c = {T_c_Na*1e6:.3f} μK")
    print()
    
    # 氦-4 液体
    n_He4_liquid = 2.18e28  # 液氦数密度
    m_He4 = 4 * amu
    T_c_He = bec_critical_temperature(n_He4_liquid, m_He4)
    print(f"  ⁴He 液体（n = {n_He4_liquid:.2e} /m³）：")
    print(f"  T_c 理论估算 = {T_c_He:.3f} K (理想气体)")
    print(f"  实验超流转变 T_λ = 2.17 K (强关联导致偏离理想)")
    print()
    print("  ✓ BEC 需要极低温 —— 1995 才实验实现")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：Brillouin 函数 + 顺磁磁化
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】自旋 1/2 顺磁体的磁化曲线")
    print("-" * 70)
    
    # 单位场强 = g μ_B B / k_B T 形式
    x_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    print(f"  自旋 J = 1/2，M/M_sat = B_J(x)")
    print()
    print(f"  {'x = gμ_B B/kT':>15s}  {'M/M_sat (B_1/2)':>18s}  {'物理区域':>15s}")
    print(f"  {'-'*15}  {'-'*18}  {'-'*15}")
    
    for x in x_values:
        M_norm = brillouin_function(x, J=0.5)
        if x < 0.5:
            region = "线性 (Curie)"
        elif x < 3:
            region = "弯曲"
        else:
            region = "接近饱和"
        print(f"  {x:>15.2f}  {M_norm:>18.4f}  {region:>15s}")
    
    print()
    print("  ✓ 小场极限：M ≈ x = (gμ_B B)/(kT)   ← Curie 定律 χ ∝ 1/T")
    print("  ✓ 大场极限：M → 1 (完全极化)")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 5：Debye 模型固体比热
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 5】Debye 模型 vs Dulong-Petit 经典极限")
    print("-" * 70)
    
    # 金刚石 T_D = 2230 K (典型高德拜温度)
    T_D = 2230
    
    print(f"  金刚石 Debye 温度 T_D = {T_D} K")
    print(f"  Dulong-Petit 经典极限 C_V = 3R = {3*8.314:.2f} J/(mol·K)")
    print()
    print(f"  {'T (K)':>8s}  {'C_V (J/mol·K)':>15s}  {'C_V / 3R':>10s}  {'低温 T³ 预测':>15s}")
    print(f"  {'-'*8}  {'-'*15}  {'-'*10}  {'-'*15}")
    
    T_test = [10, 50, 100, 300, 500, 1000, 2000, 5000]
    R = 8.314
    
    for T in T_test:
        C_V = debye_specific_heat(T, T_D, n_atoms=6.022e23)
        ratio = C_V / (3 * R)
        # 低温 T³ 律
        T3_pred = (12 * np.pi**4 / 5) * R * (T/T_D)**3 if T < T_D/10 else None
        if T3_pred is not None:
            t3_str = f"{T3_pred:.4f}"
        else:
            t3_str = "(失效)"
        print(f"  {T:>8d}  {C_V:>15.4f}  {ratio:>10.4f}  {t3_str:>15s}")
    
    print()
    print("  ✓ T << T_D : C_V ∝ T³ (Debye T³ 律)")
    print("  ✓ T >> T_D : C_V → 3R (Dulong-Petit 经典极限)")
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("第 8 章核心回顾：")
    print("  全同粒子 → 玻色 / 费米两类分布")
    print("  → 凝聚态准粒子统计 → 各种宏观现象")
    print()
    print("  实验 1: BE / FD / MB 对比 ✓")
    print("  实验 2: 铜的自由电子气 (高度简并) ✓")
    print("  实验 3: BEC 临界温度 (μK 量级) ✓")
    print("  实验 4: Brillouin 函数 (顺磁磁化) ✓")
    print("  实验 5: Debye T³ 律 + Dulong-Petit ✓")
    print()
    print("练习 / Exercises:")
    print("  1. 计算白矮星的简并压（n ~ 10³⁶ /m³, 相对论修正必要）")
    print("  2. 模拟铷 BEC 凝聚态原子比例 N₀/N vs T/T_c")
    print("  3. 不同 J 的 Brillouin 函数对比 (J = 1/2, 1, 3/2, 5/2)")
    print("  4. Einstein 模型 vs Debye 模型对比")
    print("=" * 70)
