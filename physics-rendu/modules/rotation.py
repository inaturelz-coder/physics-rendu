"""
rotation.py — 转动与角动量模块
================================
Module 3 of "Physics is Alive" / 《物理是活的》第 3 模块

实现 / Implements:
    - RotatingParticle: 带角动量计算的粒子（继承 Particle）
    - OrbitingParticle: 中心力场（行星绕太阳）
    - 验证角动量守恒（Noether 第三例）
    - 验证开普勒第二定律（面积速度恒定）

物理基础 / Physics:
    L = r × p （角动量）
    τ = r × F （力矩）
    dL/dt = τ
    中心力 → τ = 0 → L 守恒
    L 守恒 ⟹ 开普勒第二定律（面积速度恒定）

栗周 / Li Zhou
2026 年 6 月 / June 2026
MIT License
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from particle import Particle


class RotatingParticle(Particle):
    """带角动量分析的粒子（基类）。
    Particle with angular momentum analysis (base class).
    
    Example:
        >>> p = RotatingParticle(mass=1.0, position=[1, 0], velocity=[0, 1])
        >>> print(p.angular_momentum())  # L_z = 1.0
    """
    
    def angular_momentum(self, origin=None):
        """角动量 z 分量（2D 情况）/ Angular momentum z-component (2D)
        
        L = r × p
        2D: L_z = x*p_y - y*p_x
        
        Parameters:
            origin: 参考点（默认原点）/ Reference point (default origin)
        
        Returns:
            float: L_z（角动量 z 分量）
        """
        if origin is None:
            origin = np.zeros_like(self.r)
        r_rel = self.r - origin
        p = self.momentum
        L_z = r_rel[0] * p[1] - r_rel[1] * p[0]
        return L_z


class OrbitingParticle(RotatingParticle):
    """中心力场中的粒子（行星绕太阳）。
    Particle in central force field (planet orbiting sun).
    
    F = -GMm r̂ / r²  （万有引力 / gravity）
    
    Conservation laws verified:
    - 角动量 L 守恒（中心力 → 力矩 = 0）
    - 总能量 E = KE + PE 守恒（保守力）
    
    Example (椭圆轨道 / Elliptical orbit):
        >>> orbit = OrbitingParticle(mass=1.0, position=[1, 0],
        ...                          velocity=[0, 0.8], GM=1.0)
        >>> for _ in range(10000):
        ...     orbit.step_verlet(dt=0.01)
    """
    
    def __init__(self, mass, position, velocity, GM):
        """
        Parameters:
            mass: 粒子质量 / Particle mass
            position: 初始位置 / Initial position
            velocity: 初始速度 / Initial velocity
            GM: G × 中心天体质量 / G × central body mass
        """
        super().__init__(mass=mass, position=position, velocity=velocity)
        self.GM = float(GM)
        # 历史记录（用于分析）
        self.energy_history = {
            't': [0.0],
            'KE': [self.kinetic_energy],
            'PE': [self.potential_energy()],
            'E_total': [self.kinetic_energy + self.potential_energy()],
            'L_z': [self.angular_momentum()],
        }
    
    def potential_energy(self):
        """引力势能 V = -GMm/r
        Gravitational PE V = -GMm/r"""
        r = np.linalg.norm(self.r)
        return -self.GM * self.m / r
    
    @property
    def total_energy(self):
        """总能量 = 动能 + 势能"""
        return self.kinetic_energy + self.potential_energy()
    
    def force(self):
        """中心引力 F = -GMm r̂ / r²
        Central gravity F = -GMm r̂ / r²"""
        r_vec = self.r
        r = np.linalg.norm(r_vec)
        if r < 1e-10:
            return np.zeros_like(r_vec)
        return -self.GM * self.m * r_vec / r**3
    
    def step_verlet(self, dt):
        """Velocity Verlet 辛积分器
        Velocity Verlet symplectic integrator
        
        辛积分器保持哈密顿动力学的结构 →
        长期能量和角动量都守恒
        Symplectic → long-term conservation of E and L
        """
        F1 = self.force()
        self.v += 0.5 * F1 * dt / self.m
        self.r += self.v * dt
        F2 = self.force()
        self.v += 0.5 * F2 * dt / self.m
        
        # 记录历史（同时更新位置和能量历史）
        t_new = self.energy_history['t'][-1] + dt
        self.energy_history['t'].append(t_new)
        self.energy_history['KE'].append(self.kinetic_energy)
        self.energy_history['PE'].append(self.potential_energy())
        self.energy_history['E_total'].append(self.total_energy)
        self.energy_history['L_z'].append(self.angular_momentum())
        # 也更新父类的 history
        self.history['t'].append(t_new)
        self.history['r'].append(self.r.copy())
        self.history['v'].append(self.v.copy())
        self.history['m'].append(self.m)
    
    def __repr__(self):
        return (f"OrbitingParticle(m={self.m:.3g}, "
                f"r={self.r}, v={self.v}, "
                f"L={self.angular_momentum():.4f}, "
                f"E={self.total_energy:.4f})")


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("rotation.py — 物理是活的 / 第 3 模块演示")
    print("rotation.py — Physics is Alive / Module 3 Demo")
    print("=" * 70)
    print()
    print("主题：角动量守恒（Noether 第三例：旋转对称性）")
    print("Topic: Angular momentum conservation (Noether trilogy ends)")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：基本角动量计算
    # Experiment 1: Basic angular momentum
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】基本角动量验证")
    print("[Experiment 1] Basic angular momentum")
    print("-" * 70)
    
    # 一个 1 kg 的粒子，位置 (1, 0)，速度 (0, 1) → L_z = 1*1 = 1
    p1 = RotatingParticle(mass=1.0, position=[1, 0], velocity=[0, 1])
    print(f"  粒子 / Particle: m=1.0 kg, r=[1,0] m, v=[0,1] m/s")
    print(f"  L_z = x·p_y - y·p_x = 1·1 - 0·0 = {p1.angular_momentum():.4f}")
    print(f"  (理论值 / Theory: 1.0)")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：椭圆轨道——角动量守恒
    # Experiment 2: Elliptical orbit—angular momentum conservation
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】椭圆轨道——角动量守恒验证")
    print("[Experiment 2] Elliptical orbit—L conservation")
    print("-" * 70)
    
    # 初始：近日点 r=1，速度 v_perp=0.8（小于圆轨道的 1.0 → 椭圆）
    # GM = 1（归一化单位）
    orbit = OrbitingParticle(mass=1.0, position=[1.0, 0],
                              velocity=[0, 0.8], GM=1.0)
    
    L_initial = orbit.angular_momentum()
    E_initial = orbit.total_energy
    
    # 模拟一个完整轨道周期（约 5 个单位时间）
    dt = 0.001
    n_steps = 50000
    
    for _ in range(n_steps):
        orbit.step_verlet(dt)
    
    L_final = orbit.angular_momentum()
    E_final = orbit.total_energy
    
    # 提取轨道范围
    r_history = np.array([np.linalg.norm(r) for r in orbit.history['r']])
    r_min = r_history.min()  # 近日点
    r_max = r_history.max()  # 远日点
    
    L_history = np.array(orbit.energy_history['L_z'])
    L_max_dev = np.abs(L_history - L_initial).max()
    
    print(f"  初始角动量 / Initial L:  {L_initial:.6f}")
    print(f"  最终角动量 / Final L:    {L_final:.6f}")
    print(f"  L 最大偏差 / Max drift:  {L_max_dev:.2e}")
    print(f"  相对漂移 / Relative:     {L_max_dev/abs(L_initial)*100:.2e} %")
    print()
    print(f"  初始能量 / Initial E:    {E_initial:.6f}")
    print(f"  最终能量 / Final E:      {E_final:.6f}")
    print(f"  E 漂移 / E drift:        {(E_final-E_initial)/abs(E_initial)*100:+.4f} %")
    print()
    print(f"  轨道范围 / Orbit range:")
    print(f"    近日点 / Perihelion: r_min = {r_min:.4f}")
    print(f"    远日点 / Aphelion:   r_max = {r_max:.4f}")
    print(f"    椭圆离心率 / Eccentricity ≈ {(r_max-r_min)/(r_max+r_min):.4f}")
    print()
    print("  ✓ Verlet 辛积分器精确保持角动量——这就是 Noether 在数值上的实现。")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：开普勒第二定律
    # Experiment 3: Kepler's second law
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】开普勒第二定律——面积速度恒定")
    print("[Experiment 3] Kepler's 2nd law—constant areal velocity")
    print("-" * 70)
    
    # 面积速度 dA/dt = L / (2m)
    # 这是 L 守恒 → dA/dt 守恒 的直接推论
    areal_velocity = L_initial / (2 * orbit.m)
    
    print(f"  L 守恒 → 面积速度恒定：")
    print(f"  dA/dt = L / (2m) = {L_initial:.4f} / (2·{orbit.m}) = {areal_velocity:.4f}")
    print()
    print("  采样三个时刻 / Sample 3 time points:")
    
    # 在轨道的三个不同点采样 dA/dt
    sample_steps = [n_steps // 4, n_steps // 2, 3 * n_steps // 4]
    for idx, step in enumerate(sample_steps):
        r = orbit.history['r'][step]
        v = orbit.history['v'][step]
        r_mag = np.linalg.norm(r)
        # dA/dt = (1/2) |r × v| 
        dA_dt = 0.5 * abs(r[0]*v[1] - r[1]*v[0])
        t_sample = orbit.energy_history['t'][step]
        print(f"    t={t_sample:.2f}s, r={r_mag:.3f}: dA/dt = {dA_dt:.4f}")
    
    print()
    print("  三个不同位置——面积速度完全相同 ✓")
    print("  Kepler 1609 年用第谷·布拉赫数据归纳出——你用代码 1 秒重现。")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：中子星——角动量守恒在宇宙学
    # Experiment 4: Neutron star—angular momentum on cosmic scales
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】中子星——角动量守恒在宇宙尺度")
    print("[Experiment 4] Neutron star—L on cosmic scales")
    print("-" * 70)
    
    # 太阳参数
    R_sun = 7e8          # 太阳半径 (m)
    T_sun = 25 * 86400   # 自转周期 25 天 (s)
    omega_sun = 2 * np.pi / T_sun
    
    # 中子星参数
    R_ns = 10e3          # 中子星半径 10 km
    
    # I ∝ R²，假设质量不变
    # I_sun ω_sun = I_ns ω_ns
    # (2/5) M R_sun² ω_sun = (2/5) M R_ns² ω_ns
    # ω_ns = ω_sun × (R_sun/R_ns)²
    
    omega_ns = omega_sun * (R_sun / R_ns)**2
    T_ns = 2 * np.pi / omega_ns
    rotations_per_sec = omega_ns / (2 * np.pi)
    
    print(f"  太阳 / Sun: R = {R_sun:.2e} m, T_rot = {T_sun/86400:.1f} 天")
    print(f"  中子星 / Neutron star: R = {R_ns:.2e} m")
    print()
    print(f"  角动量守恒预测 (I ∝ R²):")
    print(f"    ω_ns = ω_sun · (R_sun/R_ns)² = ω_sun × {(R_sun/R_ns)**2:.2e}")
    print(f"    中子星自转周期 T = {T_ns*1000:.2f} ms")
    print(f"    每秒自转 = {rotations_per_sec:.0f} 圈")
    print()
    print(f"  对照：最快毫秒脉冲星 PSR J1748-2446ad = 716 圈/秒")
    print(f"  Reality: PSR J1748-2446ad spins at 716 Hz")
    print()
    print("  纯角动量守恒的简单估算——量级吻合宇宙真实数据。")
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("Noether 三部曲 / Noether Trilogy:")
    print("  Ch 1: 空间平移 → 动量守恒")
    print("  Ch 2: 时间平移 → 能量守恒")
    print("  Ch 3: 旋转    → 角动量守恒 ← 本章")
    print()
    print("练习 / Exercises:")
    print("  1. 试不同初始速度——画出轨道（matplotlib）")
    print("  2. 验证：圆轨道下 v² = GM/r")
    print("  3. 试用 step_euler 替换 step_verlet——观察 L 是否还守恒")
    print("=" * 70)
