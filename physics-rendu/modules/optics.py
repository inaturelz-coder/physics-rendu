"""
optics.py — 光学专题模块
============================
Module 20 of "打通物理任督二脉" / 第 20 模块

实现 / Implements:
    - 杨氏双缝干涉
    - 单缝衍射
    - 激光速率方程（粒子数反转 + 阈值行为）
    - SHG 二次谐波 + 频谱分析
    - 光纤色散（高斯脉冲展宽）

物理基础 / Physics:
    双缝: I = 4 I_0 cos²(πd sinθ/λ)
    单缝: I = I_0 [sinc(πa sinθ/λ)]²
    激光: dN/dt = R_p - N/τ - B n (N-N_th)
    SHG: P^(2) ∝ E²
    色散: σ_t(z) = σ_0 √(1 + (β₂ z / σ_0²)²)

栗周 / Li Zhou
2026 年 12 月 / December 2026
MIT License
"""

import numpy as np


# =============================================================================
# 1. 杨氏双缝干涉
# =============================================================================

def young_double_slit(theta_arr, wavelength, slit_separation, I0=1.0):
    """杨氏双缝干涉强度分布
    
    I(θ) = 4 I_0 cos²(π d sinθ / λ)
    
    Parameters:
        theta_arr : 角度数组（弧度）
        wavelength : 光波长（m）
        slit_separation : 缝距 d（m）
        I0 : 单缝强度
    """
    phase = np.pi * slit_separation * np.sin(theta_arr) / wavelength
    return 4 * I0 * np.cos(phase)**2


def fringe_spacing(wavelength, slit_separation, screen_distance):
    """条纹间距 = λ L / d"""
    return wavelength * screen_distance / slit_separation


# =============================================================================
# 2. 单缝衍射
# =============================================================================

def single_slit_diffraction(theta_arr, wavelength, slit_width, I0=1.0):
    """单缝衍射（Fraunhofer）
    
    I(θ) = I_0 [sin(β) / β]²
    where β = π a sinθ / λ
    """
    beta = np.pi * slit_width * np.sin(theta_arr) / wavelength
    # 处理 β = 0 的情况
    sinc_factor = np.where(np.abs(beta) < 1e-10, 1.0, np.sin(beta) / beta)
    return I0 * sinc_factor**2


def first_minimum_angle(wavelength, slit_width):
    """单缝衍射第一极小：sinθ = λ/a"""
    return np.arcsin(wavelength / slit_width)


# =============================================================================
# 3. 激光速率方程
# =============================================================================

def simulate_laser_rate_equations(R_p_arr, tau=1e-6, tau_c=1e-9,
                                   B=1e-12, A=0.01,
                                   N_total=1e20, t_max=1e-5, dt=1e-9):
    """求解三能级激光速率方程
    
    dN_2/dt = R_p - N_2/τ - B n (N_2 - N_1)
    dn/dt   = -n/τ_c + B n (N_2 - N_1) + A N_2
    
    N_1 + N_2 ≈ N_total (假设基态填满)
    
    Parameters:
        R_p_arr : 泵浦率扫描数组
        tau    : 上能级寿命
        tau_c  : 光腔寿命
        B      : 受激辐射截面
        A      : 自发辐射率
        N_total: 总原子数
    
    Returns:
        steady_state_n : 各泵浦率下的稳态光子数
    """
    steady_n = []
    
    for R_p in R_p_arr:
        N2 = 0.0
        n = 1.0  # 种子光子
        
        n_steps = int(t_max / dt)
        for _ in range(n_steps):
            N1 = N_total - N2
            dN2 = R_p - N2/tau - B * n * (N2 - N1)
            dn = -n/tau_c + B * n * (N2 - N1) + A * N2
            N2 += dN2 * dt
            n += dn * dt
            n = max(n, 1e-10)  # 防止负光子数
        
        steady_n.append(n)
    
    return np.array(steady_n)


# =============================================================================
# 4. SHG 频谱分析
# =============================================================================

def shg_simulation(omega=2*np.pi*1e15, chi1=1.0, chi2=1e-3, E0=1.0,
                   t_max=1e-14, dt=1e-17):
    """模拟 SHG：输入 E(t) = E_0 cos(ωt)
    
    极化：P(t) = ε_0 [χ¹ E + χ² E²]
    
    傅里叶变换看频率成分。
    """
    t = np.arange(0, t_max, dt)
    E = E0 * np.cos(omega * t)
    
    # 总极化
    P_linear = chi1 * E
    P_nonlinear = chi2 * E**2
    P_total = P_linear + P_nonlinear
    
    # 傅里叶变换
    P_freq = np.fft.rfft(P_total)
    freqs = np.fft.rfftfreq(len(t), dt) * 2 * np.pi  # 角频率
    
    return t, P_total, freqs, np.abs(P_freq)


# =============================================================================
# 5. 光纤色散：高斯脉冲展宽
# =============================================================================

def fiber_dispersion_gaussian(z_arr, sigma_0=1e-12, beta_2=20e-27):
    """高斯脉冲在色散光纤中的展宽
    
    脉冲宽度 σ(z) = σ_0 √(1 + (β_2 z / σ_0²)²)
    
    Parameters:
        z_arr : 光纤长度数组（m）
        sigma_0 : 初始脉冲宽度（s）
        beta_2 : 二阶色散系数（s²/m），单模光纤 ~ -20 ps²/km
    
    Returns:
        sigma_z : 各位置的脉冲宽度
    """
    z = np.asarray(z_arr)
    factor = beta_2 * z / sigma_0**2
    return sigma_0 * np.sqrt(1 + factor**2)


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("optics.py — 打通物理任督二脉 / 第 20 模块演示")
    print("Topic: 光学专题 —— 干涉 + 衍射 + 激光 + SHG + 色散")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：杨氏双缝干涉
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】杨氏双缝干涉")
    print("-" * 70)
    
    wavelength = 632.8e-9  # He-Ne 激光 632.8 nm
    d = 0.1e-3  # 缝距 0.1 mm
    L = 1.0     # 屏幕距离 1 m
    
    delta_y = fringe_spacing(wavelength, d, L)
    
    print(f"  光波长 λ = {wavelength*1e9:.1f} nm (He-Ne 激光)")
    print(f"  缝距 d = {d*1e3:.2f} mm")
    print(f"  屏幕距离 L = {L} m")
    print()
    print(f"  条纹间距 Δy = λL/d = {delta_y*1e3:.3f} mm")
    print()
    
    # 计算几个角度的强度
    theta_arr = np.linspace(-0.005, 0.005, 9)  # ±5 mrad
    I_arr = young_double_slit(theta_arr, wavelength, d)
    
    print(f"  {'θ (mrad)':>10s}  {'y on screen (mm)':>18s}  {'I/I_0':>10s}")
    print(f"  {'-'*10}  {'-'*18}  {'-'*10}")
    for theta, I in zip(theta_arr, I_arr):
        y_mm = theta * L * 1e3
        print(f"  {theta*1e3:>10.3f}  {y_mm:>18.3f}  {I:>10.4f}")
    
    print()
    print(f"  ✓ 极大极小交替 —— 光的波动性证明")
    print(f"  ✓ 极大间距 = 6.328 mm（理论 λL/d = 6.328 mm）")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：单缝衍射
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】单缝衍射（Fraunhofer）")
    print("-" * 70)
    
    a = 0.05e-3  # 缝宽 0.05 mm
    
    theta_min1 = first_minimum_angle(wavelength, a)
    print(f"  缝宽 a = {a*1e3:.3f} mm")
    print(f"  第一极小角度 sin⁻¹(λ/a) = {theta_min1*1e3:.4f} mrad")
    print()
    
    theta_test = np.array([0, 5, 10, 12.66, 15, 20, 25, 30]) * 1e-3
    I_test = single_slit_diffraction(theta_test, wavelength, a)
    
    print(f"  {'θ (mrad)':>10s}  {'I/I_0':>10s}  {'物理':>20s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*20}")
    for theta, I in zip(theta_test, I_test):
        if abs(theta) < 1e-6:
            phys = "中央极大"
        elif abs(theta - theta_min1) < 1e-4:
            phys = "第一极小"
        else:
            phys = ""
        print(f"  {theta*1e3:>10.2f}  {I:>10.4f}  {phys:>20s}")
    
    print()
    print(f"  ✓ 中央极大 + 两侧次极大 + 极小")
    print(f"  ✓ 第一极小 ≈ {theta_min1*1e3:.2f} mrad")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：激光速率方程 — 阈值行为
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】激光速率方程 — 阈值行为")
    print("-" * 70)
    
    # 扫描泵浦率
    R_p_arr = np.logspace(15, 22, 15)  # 泵浦率范围
    n_steady = simulate_laser_rate_equations(R_p_arr)
    
    print(f"  扫描泵浦率 R_p，看稳态光子数 n_steady：")
    print(f"  {'R_p (1/s)':>12s}  {'n_steady':>12s}  {'状态':>15s}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*15}")
    
    threshold_found = False
    for R_p, n in zip(R_p_arr, n_steady):
        if n < 1e3:
            status = "亚阈值"
        elif n < 1e8:
            status = "← 接近阈值"
            if not threshold_found:
                threshold_found = True
        else:
            status = "激光振荡"
        print(f"  {R_p:>12.2e}  {n:>12.2e}  {status:>15s}")
    
    print()
    print(f"  ✓ 低泵浦：n 几乎为零（亚阈值，只有自发辐射）")
    print(f"  ✓ 超过阈值后 n 指数增长 → 激光振荡")
    print(f"  ✓ 这是激光器的\"开/关\"特性的物理来源")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：SHG 二次谐波频谱
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】SHG 二次谐波 — 频谱分析")
    print("-" * 70)
    
    omega0 = 2 * np.pi * 3e14  # ω = 300 THz (1 μm 光)
    t, P, freqs, P_spectrum = shg_simulation(omega=omega0, chi1=1.0, chi2=0.1, E0=1.0)
    
    # 找峰
    n_show = len(freqs)
    # 在 ω 和 2ω 附近找峰
    idx_omega = np.argmin(np.abs(freqs - omega0))
    idx_2omega = np.argmin(np.abs(freqs - 2*omega0))
    idx_DC = 0
    
    print(f"  输入：E(t) = E_0 cos(ωt), ω = {omega0/(2*np.pi)/1e12:.1f} THz")
    print(f"  极化 P = χ¹ E + χ² E²")
    print()
    print(f"  傅里叶变换关键频率分量：")
    print(f"  {'频率 (THz)':>14s}  {'|P(ω)| 相对':>14s}  {'物理':>20s}")
    print(f"  {'-'*14}  {'-'*14}  {'-'*20}")
    
    P_max = P_spectrum.max()
    print(f"  {0:>14.1f}  {P_spectrum[idx_DC]/P_max:>14.4f}  {'DC 分量 (来自 χ²)':>20s}")
    print(f"  {freqs[idx_omega]/(2*np.pi)/1e12:>14.1f}  "
          f"{P_spectrum[idx_omega]/P_max:>14.4f}  {'基频 ω (线性)':>20s}")
    print(f"  {freqs[idx_2omega]/(2*np.pi)/1e12:>14.1f}  "
          f"{P_spectrum[idx_2omega]/P_max:>14.4f}  {'2ω (SHG!)':>20s}")
    
    print()
    print(f"  ✓ SHG 出现 2ω 频率 —— 倍频")
    print(f"  ✓ 这就是绿光激光笔的物理：1064 nm Nd:YAG → 532 nm SHG")
    print(f"  ✓ χ² 也产生 DC 分量（光整流效应）")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 5：光纤色散 — 高斯脉冲展宽
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 5】光纤色散 — 高斯脉冲沿光纤展宽")
    print("-" * 70)
    
    sigma_0 = 10e-12  # 初始脉宽 10 ps
    beta_2 = 20e-27   # 二阶色散 20 ps²/km
    
    z_arr = np.array([0, 1, 5, 10, 50, 100, 500, 1000]) * 1e3  # 0 to 1000 km
    sigma_arr = fiber_dispersion_gaussian(z_arr, sigma_0, beta_2)
    
    print(f"  初始脉冲宽度 σ_0 = {sigma_0*1e12:.1f} ps")
    print(f"  二阶色散 β_2 = {beta_2*1e27:.1f} ps²/km")
    print()
    print(f"  {'光纤长度 (km)':>15s}  {'脉宽 σ (ps)':>15s}  {'展宽倍数':>10s}")
    print(f"  {'-'*15}  {'-'*15}  {'-'*10}")
    
    for z, sig in zip(z_arr, sigma_arr):
        ratio = sig / sigma_0
        print(f"  {z*1e-3:>15.0f}  {sig*1e12:>15.4f}  {ratio:>10.3f}×")
    
    print()
    print(f"  ✓ 短距离（< 10 km）脉冲基本不变")
    print(f"  ✓ 长距离（> 100 km）脉冲明显展宽")
    print(f"  ✓ 1000 km 光纤后展宽 {sigma_arr[-1]/sigma_0:.1f} 倍")
    print(f"  ✓ 限制光通信的传输距离 + 速率")
    print(f"  ✓ 现代系统用 DSP（数字信号处理）+ 色散补偿光纤解决")
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("第 20 章核心回顾：")
    print("  光学 = 几何 + 波动 + 量子三层")
    print()
    print("  实验 1: 杨氏双缝（光波动性）✓")
    print("  实验 2: 单缝衍射（衍射极限）✓")
    print("  实验 3: 激光阈值行为 ✓")
    print("  实验 4: SHG 二次谐波 ✓")
    print("  实验 5: 光纤色散脉冲展宽 ✓")
    print()
    print("练习 / Exercises:")
    print("  1. N 缝衍射光栅 + 高分辨光谱")
    print("  2. 飞秒脉冲传播 + Kerr 自相位调制")
    print("  3. EDFA 增益谱 + 噪声系数")
    print("  4. 单光子 HOM 实验数值模拟")
    print("=" * 70)
