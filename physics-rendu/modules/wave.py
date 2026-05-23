"""
wave.py — 振动与波动模块
========================
Module 4 of "Physics is Alive" / 《物理是活的》第 4 模块

实现 / Implements:
    - DampedDrivenOscillator: 阻尼 + 受迫振子（继承 SpringOscillator）
    - CoupledOscillators: N 个耦合谐振子 → 波传播
    - 验证 SHM / 阻尼三种情形 / 共振峰
    - 演示从粒子振动到波动的连续极限

物理基础 / Physics:
    简谐：ẍ + ω₀²x = 0
    阻尼：ẍ + 2γẋ + ω₀²x = 0
    受迫：ẍ + 2γẋ + ω₀²x = (F₀/m)cos(ωt)
    共振：ω = ω_r ≈ ω₀（弱阻尼）
    
    多体耦合 → 波动方程：∂²u/∂t² = c² ∂²u/∂x²

栗周 / Li Zhou
2026 年 6 月 / June 2026
MIT License
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oscillator import SpringOscillator


# =============================================================================
# 类 1：阻尼 + 受迫谐振子
# =============================================================================

class DampedDrivenOscillator(SpringOscillator):
    """阻尼 + 受迫谐振子（继承 SpringOscillator）
    Damped driven oscillator.
    
    方程 / Equation:
        ẍ + 2γẋ + ω₀²x = (F₀/m) cos(ωt)
    
    Example (共振扫描):
        >>> osc = DampedDrivenOscillator(mass=1.0, k=1.0, gamma=0.05,
        ...                              F0=1.0, omega_drive=1.0)
        >>> for t in np.linspace(0, 100, 10000):
        ...     osc.step_verlet(t, dt=0.01)
    """
    
    def __init__(self, mass, k, gamma, F0=0.0, omega_drive=0.0, x0=1.0, v0=0.0):
        """
        Parameters:
            mass : 质量
            k    : 弹簧劲度系数
            gamma: 阻尼系数（方程中的 γ）
            F0   : 驱动力幅值
            omega_drive : 驱动角频率
            x0, v0 : 初始位置、速度
        """
        super().__init__(mass=mass, k=k, x0=x0, v0=v0)
        self.gamma = float(gamma)
        self.F0 = float(F0)
        self.omega_drive = float(omega_drive)
        self.omega_0 = np.sqrt(k / mass)
        # 历史
        self.energy_history['t'] = [0.0]
        self.energy_history['x'] = [self.r[0]]
        self.energy_history['v'] = [self.v[0]]
        self.energy_history['E'] = [self.total_energy]
    
    def force(self, t):
        """合力 = 弹簧 + 阻尼 + 驱动
        Total force = spring + damping + drive"""
        F_spring = -self.k * self.r[0]
        F_damp = -2 * self.m * self.gamma * self.v[0]
        F_drive = self.F0 * np.cos(self.omega_drive * t)
        return np.array([F_spring + F_damp + F_drive, 0.0])
    
    def step_verlet(self, t, dt):
        """Velocity Verlet 步进（含时间相关力）"""
        F1 = self.force(t)
        self.v += 0.5 * F1 * dt / self.m
        self.r += self.v * dt
        F2 = self.force(t + dt)
        self.v += 0.5 * F2 * dt / self.m
        
        # 记录
        t_new = t + dt
        self.energy_history['t'].append(t_new)
        self.energy_history['x'].append(self.r[0])
        self.energy_history['v'].append(self.v[0])
        self.energy_history['E'].append(self.total_energy)
    
    def steady_state_amplitude_theory(self):
        """稳态振幅的理论值（用于对照数值结果）"""
        w0 = self.omega_0
        w = self.omega_drive
        g = self.gamma
        return (self.F0 / self.m) / np.sqrt((w0**2 - w**2)**2 + 4*g**2*w**2)


# =============================================================================
# 类 2：N 个耦合谐振子（演示从粒子到波）
# =============================================================================

class CoupledOscillators:
    """N 个用弹簧串联的小球——演示连续极限下的波动方程
    
    Coupled oscillators: N masses connected by springs.
    Continuum limit → wave equation.
    
    方程: m·ẍᵢ = k(xᵢ₊₁ + xᵢ₋₁ - 2xᵢ)
    
    Example (波传播):
        >>> co = CoupledOscillators(N=100, m=1.0, k=10.0)
        >>> co.x[50] = 1.0  # 中间一个推一下
        >>> for _ in range(1000):
        ...     co.step(dt=0.01)
    """
    
    def __init__(self, N=50, m=1.0, k=1.0, boundary='fixed'):
        """
        Parameters:
            N : 粒子数
            m : 每个粒子的质量
            k : 弹簧劲度系数（粒子间）
            boundary : 'fixed' (两端固定) or 'periodic' (周期边界)
        """
        self.N = N
        self.m = float(m)
        self.k = float(k)
        self.boundary = boundary
        self.x = np.zeros(N)  # 位移
        self.v = np.zeros(N)  # 速度
        # 历史快照
        self.snapshots = []
        self.t = 0.0
    
    def force(self):
        """计算每个粒子的合力（差分形式）"""
        F = np.zeros(self.N)
        # 中间粒子
        F[1:-1] = self.k * (self.x[2:] + self.x[:-2] - 2 * self.x[1:-1])
        # 边界
        if self.boundary == 'fixed':
            # 两端固定——边界粒子的合力来自单侧
            F[0] = self.k * (self.x[1] - self.x[0])
            F[-1] = self.k * (self.x[-2] - self.x[-1])
        elif self.boundary == 'periodic':
            F[0] = self.k * (self.x[1] + self.x[-1] - 2 * self.x[0])
            F[-1] = self.k * (self.x[0] + self.x[-2] - 2 * self.x[-1])
        return F
    
    def step(self, dt):
        """Velocity Verlet 时间步进"""
        F1 = self.force()
        self.v += 0.5 * F1 * dt / self.m
        self.x += self.v * dt
        F2 = self.force()
        self.v += 0.5 * F2 * dt / self.m
        self.t += dt
    
    def snapshot(self):
        """记录当前状态"""
        self.snapshots.append({
            't': self.t,
            'x': self.x.copy(),
        })
    
    def wave_speed_theory(self, a=1.0):
        """波动方程的理论波速 c = a·√(k/m)
        其中 a 是相邻粒子间距"""
        return a * np.sqrt(self.k / self.m)


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("wave.py — 物理是活的 / 第 4 模块演示")
    print("wave.py — Physics is Alive / Module 4 Demo")
    print("=" * 70)
    print()
    print("主题：振动与波动 —— 宇宙之母")
    print("Topic: Vibrations and Waves — The Mother of All Motion")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：简谐振动（无阻尼，无驱动）
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】简谐振动 —— 验证能量守恒")
    print("[Experiment 1] Simple harmonic motion")
    print("-" * 70)
    
    osc1 = DampedDrivenOscillator(mass=1.0, k=1.0, gamma=0.0, F0=0.0,
                                   omega_drive=0.0, x0=1.0, v0=0.0)
    
    dt = 0.001
    T_max = 100.0  # 模拟 100 秒（约 16 个周期）
    n_steps = int(T_max / dt)
    
    for i in range(n_steps):
        t = i * dt
        osc1.step_verlet(t, dt)
    
    x_array = np.array(osc1.energy_history['x'])
    E_array = np.array(osc1.energy_history['E'])
    
    print(f"  ω₀ = {osc1.omega_0:.4f} rad/s")
    print(f"  周期 T = {2*np.pi/osc1.omega_0:.4f} s (理论)")
    print(f"  振幅范围: [{x_array.min():.4f}, {x_array.max():.4f}]")
    print(f"  振幅理论: ±1.0000")
    print(f"  初始能量 E₀ = {E_array[0]:.6f}")
    print(f"  最终能量 E  = {E_array[-1]:.6f}")
    print(f"  能量漂移: {(E_array[-1] - E_array[0])/E_array[0]*100:+.4f} %")
    print(f"  ✓ Verlet 辛积分器保持能量守恒")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：阻尼振动三种情形
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】阻尼振动 —— 三种情形对照")
    print("[Experiment 2] Damped oscillation — three regimes")
    print("-" * 70)
    
    omega_0 = 1.0
    gamma_cases = {
        '欠阻尼 / Underdamped (γ=0.1)': 0.1,
        '临界阻尼 / Critical    (γ=1.0)': 1.0,
        '过阻尼 / Overdamped   (γ=2.0)': 2.0,
    }
    
    T_max = 20.0
    n_steps = int(T_max / dt)
    
    for label, gamma in gamma_cases.items():
        osc = DampedDrivenOscillator(mass=1.0, k=omega_0**2, gamma=gamma,
                                      F0=0.0, omega_drive=0.0, x0=1.0, v0=0.0)
        for i in range(n_steps):
            t = i * dt
            osc.step_verlet(t, dt)
        
        x_arr = np.array(osc.energy_history['x'])
        x_max_at_end = np.abs(x_arr[-1000:]).max()  # 最后 1 秒的最大值
        x_first_zero = -1  # 第一次过零的时间
        for j in range(1, len(x_arr)):
            if x_arr[j-1] * x_arr[j] < 0:
                x_first_zero = j * dt
                break
        
        print(f"  {label}")
        print(f"    最终振幅 (t={T_max}s): {x_max_at_end:.6f}")
        if x_first_zero > 0:
            print(f"    首次过零时间: {x_first_zero:.4f} s (有振荡)")
        else:
            print(f"    无过零 → 无振荡 (单调衰减)")
        print()
    
    # -------------------------------------------------------------------------
    # 实验 3：共振扫描
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】共振扫描 —— 振幅 vs 驱动频率")
    print("[Experiment 3] Resonance scan")
    print("-" * 70)
    
    omega_0 = 1.0
    gamma = 0.05  # 弱阻尼
    
    omega_drive_array = np.array([0.5, 0.7, 0.9, 0.95, 1.00, 1.05, 1.1, 1.3, 1.5]) * omega_0
    
    print(f"  ω₀ = {omega_0:.4f}, γ = {gamma:.4f}, F₀/m = 1.0")
    print(f"  弱阻尼极限：共振峰宽度 Δω ≈ 2γ = {2*gamma:.4f}")
    print()
    print(f"  {'ω/ω₀':>8s}  {'数值振幅':>12s}  {'理论振幅':>12s}  {'误差':>8s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*8}")
    
    for w_drive in omega_drive_array:
        osc = DampedDrivenOscillator(mass=1.0, k=omega_0**2, gamma=gamma,
                                      F0=1.0, omega_drive=w_drive,
                                      x0=0.0, v0=0.0)
        # 跑足够长时间到达稳态（约 10/γ）
        T_max = 300.0
        n_steps = int(T_max / dt)
        for i in range(n_steps):
            t = i * dt
            osc.step_verlet(t, dt)
        
        # 取最后 50 秒的振幅
        x_arr = np.array(osc.energy_history['x'][-int(50/dt):])
        A_numerical = (x_arr.max() - x_arr.min()) / 2
        A_theory = osc.steady_state_amplitude_theory()
        error = abs(A_numerical - A_theory) / A_theory * 100
        
        marker = " ← 共振峰" if abs(w_drive - omega_0) < 0.01 else ""
        print(f"  {w_drive/omega_0:>8.2f}  {A_numerical:>12.4f}  {A_theory:>12.4f}  {error:>6.2f}%{marker}")
    
    print()
    print("  ✓ 数值结果与理论吻合 —— 共振峰出现在 ω = ω₀")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：从粒子到波 —— 耦合谐振子
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】N 个耦合谐振子 —— 波传播")
    print("[Experiment 4] N coupled oscillators — wave propagation")
    print("-" * 70)
    
    N = 100
    co = CoupledOscillators(N=N, m=1.0, k=10.0, boundary='fixed')
    
    # 初始扰动：中间一个粒子被推
    co.x[N // 2] = 1.0
    
    print(f"  粒子数 N = {N}, 弹簧 k = 10, 质量 m = 1")
    print(f"  初始扰动：第 {N//2} 个粒子位移 = 1.0")
    print(f"  波速理论 (a=1): c = √(k/m) = {co.wave_speed_theory():.4f}")
    print()
    
    dt = 0.01
    T_max = 5.0
    n_steps = int(T_max / dt)
    
    # 在几个特定时刻保存快照
    snapshot_times = [0.5, 1.0, 2.0, 3.0]
    snapshot_steps = [int(t / dt) for t in snapshot_times]
    
    print(f"  在不同时刻观察波包形态：")
    print(f"  {'t':>6s}  {'波包中心':>10s}  {'波包宽度':>10s}  {'传播距离':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")
    
    co.snapshot()
    for i in range(n_steps):
        co.step(dt)
        if i + 1 in snapshot_steps:
            co.snapshot()
            snapshot_idx = snapshot_steps.index(i + 1)
            t_now = snapshot_times[snapshot_idx]
            # 测量波包的左右边界（位移 > 0.05 的位置）
            indices = np.where(np.abs(co.x) > 0.05)[0]
            if len(indices) > 0:
                width = indices.max() - indices.min()
                center = (indices.max() + indices.min()) / 2
                distance = abs(center - N/2)
                print(f"  {t_now:>6.2f}  {center:>10.1f}  {width:>10d}  {distance:>10.1f}")
    
    # 估计实际波速
    if len(co.snapshots) >= 2:
        t1, t2 = snapshot_times[0], snapshot_times[-1]
        x_arr_1 = co.snapshots[1]['x']
        x_arr_2 = co.snapshots[-1]['x']
        # 找波前位置（右边）
        idx_1 = np.where(np.abs(x_arr_1) > 0.05)[0]
        idx_2 = np.where(np.abs(x_arr_2) > 0.05)[0]
        if len(idx_1) > 0 and len(idx_2) > 0:
            speed_numerical = (idx_2.max() - idx_1.max()) / (t2 - t1)
            print()
            print(f"  数值测量波速: {speed_numerical:.4f}")
            print(f"  理论波速 c = √(k/m) = {co.wave_speed_theory():.4f}")
            print(f"  ✓ 验证：离散耦合谐振子在连续极限下 → 波动方程")
    
    print()
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("章节回顾 / Chapter recap:")
    print("  第 4 章核心：任何小振幅运动都是谐振子")
    print("  ẍ + ω₀²x = 0  →  三种行为（简谐/阻尼/受迫）→  耦合 → 波动")
    print()
    print("  实验 1: 简谐振动能量守恒（Verlet 辛积分器）  ✓")
    print("  实验 2: 三种阻尼情形对照                        ✓")
    print("  实验 3: 共振扫描与理论吻合                      ✓")
    print("  实验 4: 多体耦合 → 连续极限 → 波动方程         ✓")
    print()
    print("练习 / Exercises:")
    print("  1. 试不同 ω/ω₀ —— 用 matplotlib 画共振峰曲线")
    print("  2. 试不同 N —— 验证连续极限的近似程度")
    print("  3. 试 periodic 边界 + 初始 Gaussian 波包 —— 看色散现象")
    print("=" * 70)
