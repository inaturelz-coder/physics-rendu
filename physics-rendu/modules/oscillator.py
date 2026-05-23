"""
oscillator.py — 谐振子动力学模块
================================
Module 2 of "Physics is Alive" / 《物理是活的》第 2 模块

实现 / Implements:
    - SpringOscillator 类：弹簧振子（线性回复力）
    - 多种数值积分器对比：Euler vs Velocity Verlet
    - 演示能量守恒（辛积分器 vs 非辛积分器）

物理基础 / Physics:
    弹簧振子：F = -kx
    总能量 E = (1/2)mv² + (1/2)kx²
    时间平移对称 → 能量守恒（Noether 定理）

数值意义 / Numerical Significance:
    Euler 方法：能量随时间漂移（数值伪影）
    Velocity Verlet：能量长期守恒（辛积分器）
    这是计算物理的经典教训——算法必须尊重对称性

栗周 / Li Zhou
2026 年 6 月 / June 2026
MIT License
"""

import numpy as np
import sys
import os

# 复用第 1 章的 Particle 模块 / Reuse Chapter 1's Particle module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from particle import Particle


class SpringOscillator(Particle):
    """弹簧振子（1D，但仍用 2D 矢量表示，便于扩展）
    Spring oscillator (1D, but using 2D vectors for extensibility)
    
    力：F = -kx（线性回复力）
    Force: F = -kx (linear restoring force)
    
    Attributes:
        k (float): 劲度系数 / spring constant (N/m)
        + Particle 的所有属性 / all Particle attributes
    
    Example:
        >>> osc = SpringOscillator(mass=1.0, k=4.0, x0=1.0, v0=0.0)
        >>> # 周期 T = 2π·sqrt(m/k) = π s
        >>> for _ in range(1000):
        ...     osc.step_verlet(dt=0.001)
        >>> print(osc.total_energy)  # 应保持 ≈ 2.0 J
    """
    
    def __init__(self, mass, k, x0, v0):
        super().__init__(mass=mass, position=[x0, 0], velocity=[v0, 0])
        self.k = float(k)
        # 能量历史 / Energy history
        self.energy_history = {
            't': [0.0],
            'KE': [self.kinetic_energy],
            'PE': [self.potential_energy],
            'E_total': [self.total_energy],
        }
    
    @property
    def potential_energy(self):
        """弹性势能 V = (1/2) k x²
        Elastic PE V = (1/2) k x²
        """
        return 0.5 * self.k * self.r[0]**2
    
    @property
    def total_energy(self):
        """总能量 E = KE + PE
        Total energy E = KE + PE
        
        理论上：dE/dt = 0（能量守恒）
        Theory: dE/dt = 0 (energy conservation)
        """
        return self.kinetic_energy + self.potential_energy
    
    @property
    def force(self):
        """胡克定律 F = -kx
        Hooke's law F = -kx
        """
        return np.array([-self.k * self.r[0], 0.0])
    
    def step_euler(self, dt):
        """显式 Euler 方法（一阶精度，非辛）
        Explicit Euler method (1st order, non-symplectic)
        
        ⚠️ 警告：能量会漂移！
        ⚠️ Warning: energy drifts!
        
        v_{n+1} = v_n + F(x_n)/m · dt
        x_{n+1} = x_n + v_n · dt
        """
        F = self.force
        self.v += F * dt / self.m
        self.r += self.v * dt
        self._record(dt)
    
    def step_verlet(self, dt):
        """Velocity Verlet 方法（二阶精度，辛积分器）
        Velocity Verlet method (2nd order, symplectic)
        
        ✓ 长期保持能量守恒（辛性质）
        ✓ Long-term energy conservation (symplectic property)
        
        步骤 / Steps:
            1. v_{n+1/2} = v_n + F(x_n)/2m · dt
            2. x_{n+1}   = x_n + v_{n+1/2} · dt
            3. v_{n+1}   = v_{n+1/2} + F(x_{n+1})/2m · dt
        """
        # 半步速度 / Half-step velocity
        F1 = self.force
        self.v += 0.5 * F1 * dt / self.m
        
        # 整步位置 / Full-step position
        self.r += self.v * dt
        
        # 第二半步速度（用新位置的力）/ Second half-step (with new force)
        F2 = self.force
        self.v += 0.5 * F2 * dt / self.m
        
        self._record(dt)
    
    def _record(self, dt):
        """记录历史 / Record history"""
        t_new = self.energy_history['t'][-1] + dt
        self.energy_history['t'].append(t_new)
        self.energy_history['KE'].append(self.kinetic_energy)
        self.energy_history['PE'].append(self.potential_energy)
        self.energy_history['E_total'].append(self.total_energy)
        # 也更新父类的 history（用于位置/速度跟踪）
        # Also update parent's history
        self.history['t'].append(t_new)
        self.history['r'].append(self.r.copy())
        self.history['v'].append(self.v.copy())
        self.history['m'].append(self.m)
    
    def __repr__(self):
        return (f"SpringOscillator(m={self.m:.3g}, k={self.k:.3g}, "
                f"x={self.r[0]:.3f}, v={self.v[0]:.3f}, "
                f"E={self.total_energy:.4f})")


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("oscillator.py — 物理是活的 / 第 2 模块演示")
    print("oscillator.py — Physics is Alive / Module 2 Demo")
    print("=" * 70)
    print()
    print("演示主题：能量守恒（Noether 定理）的数值验证")
    print("Demo: Numerical verification of energy conservation (Noether)")
    print()
    
    # 系统参数 / System parameters
    m = 1.0      # 质量 / mass (kg)
    k = 4.0      # 劲度系数 / spring constant (N/m)
    x0 = 1.0     # 初始位移 / initial displacement (m)
    v0 = 0.0     # 初始速度 / initial velocity (m/s)
    
    # 解析解参数 / Analytical parameters
    omega = np.sqrt(k / m)
    T = 2 * np.pi / omega
    E_exact = 0.5 * k * x0**2  # 初始时全是势能
    
    print(f"系统 / System: m={m} kg, k={k} N/m, x₀={x0} m, v₀={v0} m/s")
    print(f"角频率 / Angular frequency: ω = √(k/m) = {omega:.4f} rad/s")
    print(f"周期 / Period: T = 2π/ω = {T:.4f} s")
    print(f"初始总能量 / Initial total energy: E = {E_exact:.4f} J")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：Euler 方法——能量漂移
    # Experiment 1: Euler method—energy drifts
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】显式 Euler 方法 / Explicit Euler method")
    print("-" * 70)
    
    osc_euler = SpringOscillator(mass=m, k=k, x0=x0, v0=v0)
    dt = 0.01
    n_periods = 100
    n_steps = int(n_periods * T / dt)
    
    for _ in range(n_steps):
        osc_euler.step_euler(dt)
    
    E_final_euler = osc_euler.total_energy
    drift_euler = (E_final_euler - E_exact) / E_exact * 100
    
    print(f"  时间步 / dt:           {dt} s")
    print(f"  模拟 / Simulated:       {n_periods} 周期 / periods")
    print(f"  初始能量 / Initial E:   {E_exact:.4f} J")
    print(f"  最终能量 / Final E:     {E_final_euler:.4f} J")
    print(f"  能量漂移 / Drift:       {drift_euler:+.2f} %  ⚠️")
    print()
    print("  ⚠️ Euler 方法不保守能量——这是非辛积分器的通病。")
    print("  ⚠️ Euler doesn't conserve energy—common flaw of non-symplectic")
    print("     integrators.")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：Velocity Verlet——能量守恒
    # Experiment 2: Velocity Verlet—energy conserved
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】Velocity Verlet 方法 / Velocity Verlet")
    print("-" * 70)
    
    osc_verlet = SpringOscillator(mass=m, k=k, x0=x0, v0=v0)
    
    for _ in range(n_steps):
        osc_verlet.step_verlet(dt)
    
    E_final_verlet = osc_verlet.total_energy
    drift_verlet = (E_final_verlet - E_exact) / E_exact * 100
    
    # 取最大值 / Look at the max drift
    E_history = np.array(osc_verlet.energy_history['E_total'])
    max_drift_verlet = (E_history.max() - E_exact) / E_exact * 100
    min_drift_verlet = (E_history.min() - E_exact) / E_exact * 100
    
    print(f"  时间步 / dt:           {dt} s")
    print(f"  模拟 / Simulated:       {n_periods} 周期 / periods")
    print(f"  初始能量 / Initial E:   {E_exact:.4f} J")
    print(f"  最终能量 / Final E:     {E_final_verlet:.6f} J")
    print(f"  最终漂移 / Final drift: {drift_verlet:+.4f} %")
    print(f"  能量振荡范围 / E range: [{min_drift_verlet:+.4f}, "
          f"{max_drift_verlet:+.4f}] %")
    print()
    print("  ✓ Verlet 方法长期保持能量守恒——这是辛积分器的优点。")
    print("  ✓ Verlet conserves energy long-term—symplectic integrator wins.")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：对比 — 这就是 Noether 在数值上的表现
    # Experiment 3: Compare — Noether at the numerical level
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】对比 / Comparison")
    print("-" * 70)
    print(f"  方法 / Method        |  最终能量误差 / Final E drift")
    print(f"  Euler                |  {drift_euler:+8.2f} %")
    print(f"  Velocity Verlet      |  {drift_verlet:+8.4f} %")
    print()
    
    ratio = abs(drift_euler / drift_verlet) if drift_verlet != 0 else float('inf')
    print(f"  Verlet 比 Euler 准确约 {ratio:.0f} 倍")
    print(f"  Verlet ~{ratio:.0f}× more accurate than Euler")
    print()
    print("  Noether 定理告诉我们：能量守恒来自时间平移对称。")
    print("  Noether: energy conservation comes from time translation symmetry.")
    print("  辛积分器保留了这个对称——所以数值上也守恒。")
    print("  Symplectic integrators preserve this symmetry—conservation holds.")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：解析对照
    # Experiment 4: Analytical comparison
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】与解析解对照 / Compare with analytical solution")
    print("-" * 70)
    
    # x(t) = x₀ cos(ωt)（v₀ = 0 时）
    t_check = osc_verlet.history['t'][-1]
    x_analytical = x0 * np.cos(omega * t_check)
    v_analytical = -x0 * omega * np.sin(omega * t_check)
    
    print(f"  时间 t = {t_check:.3f} s ({n_periods} 周期后 / after)")
    print(f"  位置 / Position:")
    print(f"    Verlet:     x = {osc_verlet.r[0]:+.6f} m")
    print(f"    Analytical: x = {x_analytical:+.6f} m")
    print(f"    误差 / Error: {abs(osc_verlet.r[0] - x_analytical):.6f} m")
    print(f"  速度 / Velocity:")
    print(f"    Verlet:     v = {osc_verlet.v[0]:+.6f} m/s")
    print(f"    Analytical: v = {v_analytical:+.6f} m/s")
    print(f"    误差 / Error: {abs(osc_verlet.v[0] - v_analytical):.6f} m/s")
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("练习 / Exercises:")
    print("  1. 把 dt 改成 0.001 — 看 Euler 误差是否减小？")
    print("  2. 模拟 10000 周期 — 看 Verlet 能否依然守恒？")
    print("  3. 加阻尼 F = -kx - γv — 这时能量不守恒（时间对称破缺）")
    print("=" * 70)
