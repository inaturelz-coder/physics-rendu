"""
particle.py — 单粒子动力学模块
================================
Module 1 of "Physics is Alive" / 《物理是活的》第 1 模块

实现 / Implements:
    - Particle 类：恒定质量粒子（F=ma 适用）
    - Rocket 类：变质量粒子（F=ma 失效，F=dp/dt 仍正确）

物理基础 / Physics:
    F = dp/dt = d(mv)/dt = m·dv/dt + v·dm/dt
    当 dm/dt = 0 时，退化为 F = ma
    当 dm/dt ≠ 0 时，必须用 F = dp/dt（如火箭）

栗周 / Li Zhou
2026 年 6 月 / June 2026
MIT License
"""

import numpy as np


class Particle:
    """单粒子的牛顿动力学求解器。
    A solver for single-particle Newtonian dynamics.
    
    适用于质量恒定的情况——此时 F = ma 与 F = dp/dt 等价。
    For constant mass—where F = ma and F = dp/dt are equivalent.
    
    Attributes:
        m (float): 质量 / Mass
        r (ndarray): 位置矢量 / Position vector
        v (ndarray): 速度矢量 / Velocity vector
        history (dict): 时间演化的轨迹 / Trajectory history
    
    Example:
        >>> p = Particle(mass=1.0, position=[0, 0], velocity=[0, 0])
        >>> for _ in range(100):
        ...     p.apply_force([1.0, 0], dt=0.01)
        >>> print(p.r, p.v)  # 应得 [0.5, 0], [1, 0]
    """
    
    def __init__(self, mass, position, velocity):
        """
        Parameters:
            mass (float): 粒子质量 (kg) / Particle mass (kg)
            position (array_like): 初始位置 (m) / Initial position (m)
            velocity (array_like): 初始速度 (m/s) / Initial velocity (m/s)
        """
        self.m = float(mass)
        self.r = np.array(position, dtype=float)
        self.v = np.array(velocity, dtype=float)
        
        # 历史轨迹——用于绘图和分析
        # Trajectory history—for plotting and analysis
        self.history = {
            't': [0.0],
            'r': [self.r.copy()],
            'v': [self.v.copy()],
            'm': [self.m],
        }
    
    @property
    def momentum(self):
        """动量 p = mv
        Momentum p = mv
        
        Returns:
            ndarray: 动量矢量 / Momentum vector
        """
        return self.m * self.v
    
    @property
    def kinetic_energy(self):
        """动能 KE = (1/2) m v²
        Kinetic energy KE = (1/2) m v²
        
        Returns:
            float: 动能 (J) / Kinetic energy (J)
        """
        return 0.5 * self.m * np.dot(self.v, self.v)
    
    def apply_force(self, F, dt):
        """施加力 F 持续时间 dt，使用 Euler 方法更新状态。
        Apply force F for time dt using Euler method.
        
        基本方程：F = dp/dt
        当 m 恒定时：dv = F·dt / m
        Basic equation: F = dp/dt
        When m is constant: dv = F·dt / m
        
        ⚠️ 注意：只在 m 恒定时正确！
        ⚠️ Note: Only correct when m is constant!
        变质量情况请用 Rocket.burn_fuel() / 
        For variable mass, use Rocket.burn_fuel()
        
        Parameters:
            F (array_like): 力矢量 / Force vector (N)
            dt (float): 时间步长 / Time step (s)
        """
        F = np.asarray(F, dtype=float)
        
        # Euler 积分（一阶精度——简单但够用）
        # Euler integration (first-order—simple but enough)
        self.v += F * dt / self.m
        self.r += self.v * dt
        
        # 记录历史 / Record history
        t_new = self.history['t'][-1] + dt
        self.history['t'].append(t_new)
        self.history['r'].append(self.r.copy())
        self.history['v'].append(self.v.copy())
        self.history['m'].append(self.m)
    
    def __repr__(self):
        return (f"Particle(m={self.m:.3g}, "
                f"r={self.r}, v={self.v}, "
                f"p={self.momentum})")


class Rocket(Particle):
    """变质量粒子——火箭。
    Variable-mass particle—a rocket.
    
    这是 F=ma 失效、F=dp/dt 仍然成立的经典例子。
    Classic example where F=ma fails but F=dp/dt still works.
    
    通过 burn_fuel() 方法消耗燃料并产生推力，
    自然实现齐奥尔科夫斯基火箭方程。
    Use burn_fuel() to consume fuel and generate thrust,
    naturally realizing the Tsiolkovsky rocket equation.
    
    Example (Falcon 9 第一级 / Falcon 9 first stage):
        >>> rocket = Rocket(mass=549000, position=[0,0], velocity=[0,0])
        >>> for _ in range(1620):
        ...     if rocket.m > 130000:
        ...         rocket.burn_fuel(2400, [-2800, 0], dt=0.1)
        >>> print(f"最终速度: {rocket.v[0]:.0f} m/s")  # ~4000 m/s
    """
    
    def burn_fuel(self, dm_dt, v_exhaust, dt):
        """烧燃料 dt 时间：消耗质量 + 喷出燃料 → 产生反推力
        Burn fuel for dt: consume mass + expel propellant → recoil thrust
        
        物理 / Physics:
            动量守恒 + 燃料喷出速度 v_exhaust（相对火箭）
            → 推力 F_thrust = -v_exhaust · (dm/dt)
            
        Tsiolkovsky 方程的微观推导：
            Δv = u · ln(M₀ / M_f)
            其中 u = |v_exhaust|, M₀ = 初始质量, M_f = 最终质量
        
        Parameters:
            dm_dt (float): 燃料消耗速率 (kg/s)，必须 > 0
                          Fuel consumption rate (kg/s), must be > 0
            v_exhaust (array_like): 燃料喷射速度矢量（相对火箭，
                                    通常朝后） (m/s)
                                    Exhaust velocity vector relative to
                                    rocket (usually backward) (m/s)
            dt (float): 时间步长 (s) / Time step (s)
        
        Notes:
            - 这里我们假设 v_exhaust 是相对火箭的喷射速度
            - 推力方向 = 反喷射方向
            - We assume v_exhaust is the exhaust velocity rel. to rocket
            - Thrust direction = opposite to exhaust direction
        """
        v_exhaust = np.asarray(v_exhaust, dtype=float)
        
        if dm_dt <= 0:
            raise ValueError("dm_dt 必须 > 0 / dm_dt must be > 0")
        if self.m <= 0:
            raise ValueError("质量必须 > 0 / Mass must be > 0")
        
        # 推力 = -v_exhaust · dm/dt
        # Thrust = -v_exhaust · dm/dt
        F_thrust = -v_exhaust * dm_dt
        
        # 用 F = dp/dt 更新（此时 m 仍是当前值，下一行才更新）
        # Update using F = dp/dt (m is still current value, updated below)
        self.v += F_thrust * dt / self.m
        self.r += self.v * dt
        
        # 消耗燃料 / Consume fuel
        self.m -= dm_dt * dt
        
        # 记录 / Record
        t_new = self.history['t'][-1] + dt
        self.history['t'].append(t_new)
        self.history['r'].append(self.r.copy())
        self.history['v'].append(self.v.copy())
        self.history['m'].append(self.m)
    
    def __repr__(self):
        return (f"Rocket(m={self.m:.3g} kg, "
                f"v={self.v} m/s)")


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("particle.py — 物理是活的 / 第 1 模块演示")
    print("particle.py — Physics is Alive / Module 1 Demo")
    print("=" * 60)
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：恒定质量粒子——F=ma 工作良好
    # Experiment 1: Constant-mass particle—F=ma works
    # -------------------------------------------------------------------------
    print("【实验 1】恒定质量粒子在恒力下运动")
    print("[Experiment 1] Constant-mass particle under constant force")
    print()
    
    p1 = Particle(mass=1.0, position=[0, 0], velocity=[0, 0])
    F = [1.0, 0]  # 1 N 沿 x 方向
    dt = 0.01
    n_steps = 100
    
    for _ in range(n_steps):
        p1.apply_force(F, dt)
    
    # 应得结果（解析解）/ Expected (analytical):
    # x(t) = (1/2) a t² = 0.5 · 1 · 1² = 0.5 m
    # v(t) = a t = 1 · 1 = 1 m/s
    
    print(f"  时间 / Time:     1.0 s")
    print(f"  位置 / Position: {p1.r}  (应得 / expected: [0.5, 0])")
    print(f"  速度 / Velocity: {p1.v}  (应得 / expected: [1.0, 0])")
    print(f"  动量 / Momentum: {p1.momentum}")
    print(f"  动能 / KE:       {p1.kinetic_energy:.3f} J")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：火箭——变质量
    # Experiment 2: Rocket—variable mass
    # -------------------------------------------------------------------------
    print("【实验 2】火箭推进——以 Falcon 9 第一级为例")
    print("[Experiment 2] Rocket propulsion—Falcon 9 first stage")
    print()
    
    # Falcon 9 第一级参数（接近公开数据）
    # Falcon 9 first stage (approximate public data)
    M_0 = 549_000  # 初始总质量 (kg)
    M_f = 130_000  # 第一级燃尽后质量 (kg)
    dm_dt = 2400   # 燃料消耗率 (kg/s)
    u = 2800       # 燃料喷射速度 (m/s)
    
    rocket = Rocket(mass=M_0, position=[0, 0], velocity=[0, 0])
    v_exhaust = np.array([-u, 0])  # 向后喷射 / exhaust backward
    
    dt = 0.1
    duration = 200  # 秒（足够长，但会在燃尽时停止）
    n_steps = int(duration / dt)
    
    for _ in range(n_steps):
        if rocket.m > M_f:
            rocket.burn_fuel(dm_dt, v_exhaust, dt)
        else:
            break
    
    # 理论值（齐奥尔科夫斯基方程）/ Theoretical (Tsiolkovsky)
    delta_v_theory = u * np.log(M_0 / M_f)
    
    print(f"  初始质量 / M₀:       {M_0:>8.0f} kg")
    print(f"  最终质量 / M_f:      {rocket.m:>8.0f} kg "
          f"(目标 / target: {M_f})")
    print(f"  最终速度 / v_final:  {rocket.v[0]:>8.0f} m/s")
    print(f"  理论值 / Theoretical: {delta_v_theory:>8.0f} m/s")
    print(f"  误差 / Error:         "
          f"{(rocket.v[0] - delta_v_theory) / delta_v_theory * 100:>+8.2f} %")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：动量守恒验证（爆炸）
    # Experiment 3: Verify momentum conservation (explosion)
    # -------------------------------------------------------------------------
    print("【实验 3】爆炸 — 验证动量守恒")
    print("[Experiment 3] Explosion — verifying momentum conservation")
    print()
    
    # 一个 10 kg 物体爆炸成 3 块——总动量必须守恒
    # A 10 kg object explodes into 3 pieces—total momentum must conserve
    
    # 初始静止
    p_initial = np.array([0.0, 0.0])
    
    # 爆炸后 3 块（人为设定，保证总动量 = 0）
    pieces = [
        {'m': 3.0, 'v': np.array([5.0, 0.0])},   # 向 +x 飞
        {'m': 4.0, 'v': np.array([-3.0, 2.0])},  # 向 -x +y 飞
        {'m': 3.0, 'v': np.array([-1.0, -2.67])},
    ]
    
    p_total = sum(pc['m'] * pc['v'] for pc in pieces)
    print(f"  初始总动量 / Initial total momentum: {p_initial}")
    print(f"  爆炸后总动量 / Final total momentum:  {p_total}")
    print(f"  ⇒ {'守恒 ✓' if np.allclose(p_total, p_initial, atol=0.1) else '不守恒 ✗'} "
          f"(误差 / err: {np.linalg.norm(p_total):.3f})")
    print()
    print("  注：内力（爆炸）不改变系统总动量——这是动量守恒律的力量。")
    print("  Note: Internal forces (explosion) don't change total momentum—")
    print("        the power of momentum conservation.")
    print()
    
    print("=" * 60)
    print("演示完毕 / Demo complete.")
    print("练习 / Exercise: 改变参数试试 / Try changing parameters")
    print("=" * 60)
