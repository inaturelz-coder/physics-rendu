"""
chaos.py — 混沌系统模块
========================
Module 5 of "Physics is Alive" / 《物理是活的》第 5 模块

实现 / Implements:
    - LorenzSystem: 经典 Lorenz 系统（蝴蝶吸引子）
    - DoublePendulum: 双摆（最简单的混沌力学系统）
    - LogisticMap: 离散映射 + 周期倍化路径
    - 数值验证：Lyapunov 指数、Feigenbaum 常数

物理基础 / Physics:
    Lorenz (1963): ẋ=σ(y-x), ẏ=x(ρ-z)-y, ż=xy-βz
    双摆: 4 自由度强非线性
    Logistic: x_{n+1} = r x_n (1-x_n)
    
    敏感依赖：|δ(t)| ≈ |δ(0)| e^{λt}
    Feigenbaum 常数：δ = 4.6692...

栗周 / Li Zhou
2026 年 6 月 / June 2026
MIT License
"""

import numpy as np
from scipy.integrate import odeint


# =============================================================================
# 类 1：Lorenz 系统
# =============================================================================

class LorenzSystem:
    """经典 Lorenz 1963 系统
    
    方程 / Equations:
        ẋ = σ(y - x)
        ẏ = x(ρ - z) - y
        ż = xy - βz
    
    经典参数（混沌）/ Classic chaotic parameters:
        σ = 10, ρ = 28, β = 8/3
    
    Example:
        >>> sys = LorenzSystem(initial=[1.0, 1.0, 1.0])
        >>> sys.evolve(t_max=50, dt=0.01)
        >>> trajectory = sys.solution  # shape (N, 3)
    """
    
    def __init__(self, initial=None, sigma=10.0, rho=28.0, beta=8/3):
        if initial is None:
            initial = [1.0, 1.0, 1.0]
        self.state0 = np.array(initial, dtype=float)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.beta = float(beta)
        self.solution = None
        self.t = None
    
    def rhs(self, state, t):
        """ODE 右端"""
        x, y, z = state
        return [
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z,
        ]
    
    def evolve(self, t_max=50.0, dt=0.01):
        """积分求解"""
        self.t = np.arange(0, t_max, dt)
        self.solution = odeint(self.rhs, self.state0, self.t)
        return self.solution
    
    def is_chaotic_regime(self):
        """判断参数是否在混沌区"""
        # Lorenz 系统的混沌阈值
        return self.rho > 24.74


# =============================================================================
# 类 2：双摆
# =============================================================================

class DoublePendulum:
    """双摆 —— 最简单的混沌力学系统
    
    4 个自由度 / 4 degrees of freedom:
        θ1, ω1, θ2, ω2
    
    强非线性来自 sin(θ1-θ2) 等耦合项
    """
    
    def __init__(self, theta1=np.pi/2, theta2=np.pi/2, omega1=0, omega2=0,
                 L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
        """
        默认初始条件：两个摆都在水平位置（接近不稳定平衡上方）—— 强混沌
        """
        self.state0 = np.array([theta1, omega1, theta2, omega2], dtype=float)
        self.L1, self.L2 = float(L1), float(L2)
        self.m1, self.m2 = float(m1), float(m2)
        self.g = float(g)
        self.solution = None
        self.t = None
    
    def rhs(self, state, t):
        """双摆 ODE（推导见 Goldstein 经典力学）"""
        theta1, omega1, theta2, omega2 = state
        L1, L2, m1, m2, g = self.L1, self.L2, self.m1, self.m2, self.g
        
        delta = theta2 - theta1
        sin_d, cos_d = np.sin(delta), np.cos(delta)
        
        den = (2*m1 + m2 - m2*np.cos(2*theta1 - 2*theta2))
        
        domega1 = (-g*(2*m1 + m2)*np.sin(theta1)
                   - m2*g*np.sin(theta1 - 2*theta2)
                   - 2*sin_d*m2*(omega2**2*L2 + omega1**2*L1*cos_d)) / (L1*den)
        
        domega2 = (2*sin_d*(omega1**2*L1*(m1 + m2)
                            + g*(m1 + m2)*np.cos(theta1)
                            + omega2**2*L2*m2*cos_d)) / (L2*den)
        
        return [omega1, domega1, omega2, domega2]
    
    def evolve(self, t_max=20.0, dt=0.01):
        self.t = np.arange(0, t_max, dt)
        self.solution = odeint(self.rhs, self.state0, self.t)
        return self.solution
    
    def cartesian_positions(self):
        """转换到 (x, y) 坐标——便于画图"""
        if self.solution is None:
            return None
        theta1 = self.solution[:, 0]
        theta2 = self.solution[:, 2]
        x1 = self.L1 * np.sin(theta1)
        y1 = -self.L1 * np.cos(theta1)
        x2 = x1 + self.L2 * np.sin(theta2)
        y2 = y1 - self.L2 * np.cos(theta2)
        return x1, y1, x2, y2


# =============================================================================
# 类 3：Logistic 映射 + 周期倍化分析
# =============================================================================

class LogisticMap:
    """离散 Logistic 映射 —— 周期倍化路径
    
    x_{n+1} = r x_n (1 - x_n)
    
    周期倍化分岔 → 混沌
    Feigenbaum 常数 δ = 4.6692... 是普适的
    """
    
    def __init__(self, r=3.5, x0=0.5):
        self.r = float(r)
        self.x0 = float(x0)
    
    def iterate(self, n_skip=1000, n_keep=500):
        """跑到稳态——返回稳态序列"""
        x = self.x0
        # 丢弃瞬态
        for _ in range(n_skip):
            x = self.r * x * (1 - x)
        # 保留稳态
        seq = np.empty(n_keep)
        for i in range(n_keep):
            x = self.r * x * (1 - x)
            seq[i] = x
        return seq
    
    @staticmethod
    def bifurcation_diagram(r_min=2.5, r_max=4.0, n_r=1000,
                             n_skip=1000, n_keep=200):
        """生成分岔图所需的数据"""
        r_values = np.linspace(r_min, r_max, n_r)
        all_x = []
        all_r = []
        for r in r_values:
            lm = LogisticMap(r=r, x0=0.5)
            seq = lm.iterate(n_skip, n_keep)
            all_x.extend(seq)
            all_r.extend([r] * n_keep)
        return np.array(all_r), np.array(all_x)


# =============================================================================
# Lyapunov 指数估算
# =============================================================================

def estimate_lyapunov_lorenz(sigma=10, rho=28, beta=8/3, t_max=30, dt=0.01, eps=1e-9):
    """
    用"两条轨迹分离"法估算 Lorenz 系统的最大 Lyapunov 指数
    
    Returns:
        lambda_estimate : 数值估算值
        sol1, sol2      : 两条轨迹（用于画图）
    """
    state1 = np.array([1.0, 1.0, 1.0])
    state2 = state1 + np.array([eps, 0, 0])
    
    sys1 = LorenzSystem(initial=state1, sigma=sigma, rho=rho, beta=beta)
    sys2 = LorenzSystem(initial=state2, sigma=sigma, rho=rho, beta=beta)
    
    sys1.evolve(t_max, dt)
    sys2.evolve(t_max, dt)
    
    dist = np.linalg.norm(sys1.solution - sys2.solution, axis=1)
    # 取对数 —— 线性区拟合斜率
    log_dist = np.log(dist + 1e-30)
    
    # 取线性增长区（避开初始稳态 + 饱和段）
    t = sys1.t
    # 取 t ∈ [1, t_max*0.4] 区间
    fit_mask = (t > 1) & (t < 0.4 * t_max) & (dist > eps * 10) & (dist < 10)
    if fit_mask.sum() < 10:
        return None, sys1.solution, sys2.solution
    
    coeffs = np.polyfit(t[fit_mask], log_dist[fit_mask], 1)
    return coeffs[0], sys1.solution, sys2.solution


# =============================================================================
# Feigenbaum 常数估算
# =============================================================================

def estimate_feigenbaum_constant():
    """
    估算 Feigenbaum 常数 δ
    
    用周期倍化分岔点 r_n 的几何收敛性：
        δ = lim (r_n - r_{n-1}) / (r_{n+1} - r_n)
    
    已知前几个分岔点（数值精确）：
    """
    # 周期倍化的关键 r 值
    r_bifurcations = [
        3.0,           # 周期 1 → 周期 2
        3.4494897,     # 周期 2 → 周期 4
        3.5440903,     # 周期 4 → 周期 8
        3.5644073,     # 周期 8 → 周期 16
        3.5687594,     # 周期 16 → 周期 32
        3.5696916,     # 周期 32 → 周期 64
    ]
    
    deltas = []
    for i in range(len(r_bifurcations) - 2):
        delta = (r_bifurcations[i+1] - r_bifurcations[i]) / \
                (r_bifurcations[i+2] - r_bifurcations[i+1])
        deltas.append(delta)
    
    return deltas, r_bifurcations


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("chaos.py — 物理是活的 / 第 5 模块演示")
    print("chaos.py — Physics is Alive / Module 5 Demo")
    print("=" * 70)
    print()
    print("主题：流体与混沌 —— 从可预测到不可预测")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：Lorenz 吸引子
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】Lorenz 蝴蝶吸引子")
    print("-" * 70)
    
    lorenz = LorenzSystem(initial=[1.0, 1.0, 1.0])
    lorenz.evolve(t_max=50, dt=0.01)
    
    x, y, z = lorenz.solution[:, 0], lorenz.solution[:, 1], lorenz.solution[:, 2]
    
    print(f"  σ = {lorenz.sigma}, ρ = {lorenz.rho}, β = {lorenz.beta:.4f}")
    print(f"  在混沌区: {lorenz.is_chaotic_regime()}")
    print(f"  轨迹长度: {len(lorenz.solution)} 点")
    print(f"  x 范围: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  y 范围: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  z 范围: [{z.min():.2f}, {z.max():.2f}]")
    
    # 统计两个"翅膀"的访问次数
    left_wing = np.sum(x < -5)
    right_wing = np.sum(x > 5)
    print(f"  左翼访问: {left_wing} 步 ({100*left_wing/len(x):.1f}%)")
    print(f"  右翼访问: {right_wing} 步 ({100*right_wing/len(x):.1f}%)")
    print(f"  ✓ 轨迹在两翼间随机切换 —— 混沌特征")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：蝴蝶效应（Lyapunov 估算）
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】蝴蝶效应 + Lyapunov 指数估算")
    print("-" * 70)
    
    epsilons = [1e-6, 1e-9, 1e-12]
    print(f"  初始差异 ε     →   t=30s 时距离")
    print(f"  {'-'*40}")
    
    for eps in epsilons:
        lam, sol1, sol2 = estimate_lyapunov_lorenz(eps=eps, t_max=30)
        # 取最终距离
        dist_final = np.linalg.norm(sol1[-1] - sol2[-1])
        # 取 t=10s 时距离
        dist_t10 = np.linalg.norm(sol1[1000] - sol2[1000])
        print(f"  ε = {eps:.0e}      t=10s: {dist_t10:.4e}   t=30s: {dist_final:.4f}")
    
    lam, _, _ = estimate_lyapunov_lorenz(eps=1e-9, t_max=30)
    print()
    print(f"  数值 Lyapunov 指数估算: λ ≈ {lam:.4f}")
    print(f"  文献值（Sprott 2003）:  λ ≈ 0.9056")
    print(f"  ✓ 蝴蝶效应 —— 初始差异 1e-9 也能在 30 秒内放大到 O(1)")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：双摆
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】双摆 —— 最简单的混沌力学系统")
    print("-" * 70)
    
    dp1 = DoublePendulum(theta1=np.pi/2, theta2=np.pi/2)
    dp2 = DoublePendulum(theta1=np.pi/2 + 1e-8, theta2=np.pi/2)  # 差 1e-8
    
    dp1.evolve(t_max=20, dt=0.01)
    dp2.evolve(t_max=20, dt=0.01)
    
    # 两个双摆末端位置的距离
    x1a, y1a, x2a, y2a = dp1.cartesian_positions()
    x1b, y1b, x2b, y2b = dp2.cartesian_positions()
    
    end_distance = np.sqrt((x2a - x2b)**2 + (y2a - y2b)**2)
    
    print(f"  两个双摆初始 θ1 差异: 1e-8 弧度")
    print(f"  双摆末端距离演化：")
    for t_check in [1, 5, 10, 15, 20]:
        idx = int(t_check / 0.01)
        if idx < len(end_distance):
            print(f"    t = {t_check:>2}s : 末端距离 = {end_distance[idx]:.4e} m")
    print(f"  ✓ 微小初始差异 → 宏观分离 —— 经典混沌特征")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：Logistic 映射与 Feigenbaum 常数
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】Logistic 映射 —— 周期倍化与 Feigenbaum 常数")
    print("-" * 70)
    
    # 不同 r 的稳态行为
    test_r = [2.5, 3.0, 3.2, 3.5, 3.55, 3.567, 3.7, 3.9]
    print(f"  {'r':>6s}  {'稳态行为':>18s}  {'周期':>6s}")
    print(f"  {'-'*6}  {'-'*18}  {'-'*6}")
    
    for r in test_r:
        lm = LogisticMap(r=r, x0=0.5)
        seq = lm.iterate(n_skip=2000, n_keep=100)
        unique = len(np.unique(np.round(seq, 4)))
        if unique == 1:
            behavior = "固定点"
            period = 1
        elif unique <= 32:
            behavior = f"周期 {unique}"
            period = unique
        else:
            behavior = "混沌"
            period = "∞"
        print(f"  {r:>6.3f}  {behavior:>18s}  {str(period):>6s}")
    print()
    
    # Feigenbaum 常数
    print(f"  Feigenbaum 常数估算：")
    deltas, r_bif = estimate_feigenbaum_constant()
    print(f"  {'分岔区间':>14s}  {'δ 估算':>10s}")
    print(f"  {'-'*14}  {'-'*10}")
    for i, d in enumerate(deltas):
        print(f"  {f'r_{i+1}/r_{i+2}':>14s}  {d:>10.4f}")
    print()
    print(f"  理论值: δ = 4.6692... (Feigenbaum 1978)")
    print(f"  收敛趋势：随着 n 增大 → 接近 4.6692")
    print(f"  ✓ 这就是混沌的普适性 —— 不同系统同一常数")
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("第 5 章核心回顾：")
    print("  非线性 + 多维 + 敏感依赖 → 混沌")
    print()
    print("  实验 1: Lorenz 蝴蝶吸引子 —— 确定性混沌      ✓")
    print("  实验 2: 蝴蝶效应 —— Lyapunov λ ≈ 0.9         ✓")
    print("  实验 3: 双摆 —— 1e-8 差异 → 宏观分离          ✓")
    print("  实验 4: Logistic 周期倍化 —— Feigenbaum 普适 ✓")
    print()
    print("练习 / Exercises:")
    print("  1. 用 matplotlib 画 Lorenz 3D 吸引子")
    print("  2. 用 matplotlib 画 Logistic 分岔图")
    print("  3. 试不同的 ρ 值：ρ=15 → 稳定，ρ=24 → 边界，ρ=28 → 混沌")
    print("  4. 双摆能量守恒检验（应当严格守恒）")
    print("=" * 70)
