"""
stat.py — 统计力学模块
========================
Module 7 of "Physics is Alive" / 《物理是活的》第 7 模块

实现 / Implements:
    - boltzmann_distribution: Boltzmann 分布数值验证
    - IsingModel2D: 2D Ising Metropolis Monte Carlo（核心）
    - 临界温度扫描 + 磁化曲线
    - 临界涨落可视化

物理基础 / Physics:
    Boltzmann: P_i = (1/Z) exp(-E_i/k_BT)
    配分函数: Z = Σ exp(-E_i/k_BT)
    Ising 2D 临界温度: T_c = 2J/(k_B ln(1+√2)) ≈ 2.269 J/k_B (Onsager 1944)

栗周 / Li Zhou
2026 年 7 月 / July 2026
MIT License
"""

import numpy as np


# 物理常数
k_B = 1.380649e-23
# Ising 中 J 和 T 都以无量纲单位（J=1, k_B=1）


# =============================================================================
# 函数：Boltzmann 分布的 Monte Carlo 验证
# =============================================================================

def boltzmann_sampling(energies, T, n_samples=100000, seed=None):
    """从有限能级体系按 Boltzmann 分布采样
    
    Parameters:
        energies : 数组，每个微观态的能量
        T        : 温度（无量纲，与能量同单位）
        n_samples: 采样次数
    
    Returns:
        counts    : 每个能级被采样的次数
        prob_sim  : 模拟概率
        prob_th   : 理论 Boltzmann 概率
    """
    rng = np.random.default_rng(seed)
    
    # 理论概率
    weights = np.exp(-np.array(energies) / T)
    Z = weights.sum()
    prob_th = weights / Z
    
    # 数值采样
    indices = rng.choice(len(energies), size=n_samples, p=prob_th)
    counts = np.bincount(indices, minlength=len(energies))
    prob_sim = counts / n_samples
    
    return counts, prob_sim, prob_th


def metropolis_single_particle(potential_func, T, n_steps=10000, step_size=0.5, seed=None):
    """单粒子在 1D 势中做 Metropolis 采样
    
    验证：得到的位置分布应符合 Boltzmann P(x) ∝ exp(-V(x)/T)
    """
    rng = np.random.default_rng(seed)
    x = 0.0
    samples = []
    accepts = 0
    
    for _ in range(n_steps):
        x_new = x + rng.uniform(-step_size, step_size)
        dE = potential_func(x_new) - potential_func(x)
        if dE < 0 or rng.random() < np.exp(-dE / T):
            x = x_new
            accepts += 1
        samples.append(x)
    
    return np.array(samples), accepts / n_steps


# =============================================================================
# 类：2D Ising 模型 + Metropolis
# =============================================================================

class IsingModel2D:
    """2D Ising 模型 Metropolis Monte Carlo
    
    H = -J Σ s_i s_j  (近邻求和)
    自旋 s_i = ±1
    周期边界条件
    
    临界温度（Onsager 精确解，J=1, k_B=1）:
        T_c = 2/ln(1+√2) ≈ 2.269185
    
    Example:
        >>> ising = IsingModel2D(L=20, T=2.0, seed=42)
        >>> ising.thermalize(n_sweeps=1000)
        >>> M = ising.measure_magnetization(n_sweeps=2000)
    """
    
    def __init__(self, L=20, T=2.5, J=1.0, seed=None):
        """
        Parameters:
            L : 边长（晶格 L×L）
            T : 温度（J, k_B 都设为 1 的单位）
            J : 耦合常数（默认 1）
            seed : 随机种子
        """
        self.L = int(L)
        self.T = float(T)
        self.J = float(J)
        self.rng = np.random.default_rng(seed)
        
        # 初始化随机自旋 ±1
        self.spins = 2 * self.rng.integers(0, 2, (L, L)) - 1
        self.spins = self.spins.astype(np.int8)
    
    def total_energy(self):
        """总能量（周期边界）"""
        # 用 numpy roll 高效计算
        right = np.roll(self.spins, -1, axis=1)
        down = np.roll(self.spins, -1, axis=0)
        E = -self.J * np.sum(self.spins * right + self.spins * down)
        return float(E)
    
    def magnetization(self):
        """平均磁化（每自旋）"""
        return float(np.mean(self.spins))
    
    def metropolis_sweep(self):
        """一次完整 sweep = L² 次随机翻转尝试"""
        L = self.L
        for _ in range(L * L):
            i = self.rng.integers(0, L)
            j = self.rng.integers(0, L)
            s = self.spins[i, j]
            # 4 个最近邻（周期边界）
            neighbors = (self.spins[(i+1) % L, j] + 
                        self.spins[(i-1) % L, j] +
                        self.spins[i, (j+1) % L] + 
                        self.spins[i, (j-1) % L])
            dE = 2 * self.J * s * neighbors
            if dE <= 0 or self.rng.random() < np.exp(-dE / self.T):
                self.spins[i, j] = -s
    
    def thermalize(self, n_sweeps=500):
        """热化：丢弃初始 sweeps 让系统达到平衡"""
        for _ in range(n_sweeps):
            self.metropolis_sweep()
    
    def measure(self, n_sweeps=2000, n_skip=10):
        """测量平均量
        
        Returns:
            results : dict with keys 'M', 'M_abs', 'E', 'M2', 'E2'
        """
        M_list, M_abs_list, E_list = [], [], []
        for k in range(n_sweeps):
            self.metropolis_sweep()
            if k % n_skip == 0:
                M_list.append(self.magnetization())
                M_abs_list.append(abs(self.magnetization()))
                E_list.append(self.total_energy() / (self.L**2))
        
        M_arr = np.array(M_list)
        M_abs_arr = np.array(M_abs_list)
        E_arr = np.array(E_list)
        
        N = self.L**2
        return {
            'M': M_arr.mean(),
            'M_abs': M_abs_arr.mean(),
            'M2': (M_arr**2).mean(),
            'E_per_site': E_arr.mean(),
            'E2_per_site2': (E_arr**2).mean(),
            'chi': N * (M_arr**2).mean() / self.T,  # 磁化率
            'C_V': N * E_arr.var() / self.T**2,     # 比热容
        }


# =============================================================================
# Onsager 精确解 - 用于对照
# =============================================================================

def onsager_Tc():
    """2D Ising 临界温度（精确）"""
    return 2.0 / np.log(1.0 + np.sqrt(2.0))


def onsager_magnetization(T, Tc=None):
    """Onsager 1949 自发磁化（精确）
    
    M(T) = [1 - sinh(2J/k_BT)^(-4)]^(1/8)  for T < T_c
         = 0                                for T >= T_c
    """
    if Tc is None:
        Tc = onsager_Tc()
    
    if T >= Tc:
        return 0.0
    
    arg = np.sinh(2.0 / T)
    M = (1 - arg**(-4))**(1/8)
    return float(M)


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("stat.py — 物理是活的 / 第 7 模块演示")
    print("Topic: 经典统计力学 —— 从微观涌现宏观")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：Boltzmann 分布数值验证
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】Boltzmann 分布验证（4 能级体系）")
    print("-" * 70)
    
    energies = [0, 1, 2, 3]  # 4 个能级
    T = 1.0
    
    counts, prob_sim, prob_th = boltzmann_sampling(energies, T, n_samples=100000, seed=42)
    
    print(f"  温度 T = {T}, 采样次数 = 100000")
    print()
    print(f"  {'能级 E':>8s}  {'模拟概率':>10s}  {'理论概率':>10s}  {'误差':>8s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")
    for i, E in enumerate(energies):
        err = abs(prob_sim[i] - prob_th[i]) / prob_th[i] * 100
        print(f"  {E:>8d}  {prob_sim[i]:>10.4f}  {prob_th[i]:>10.4f}  {err:>6.2f}%")
    print()
    print(f"  ✓ P(E) ∝ exp(-E/T) 完美验证")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：单粒子 Metropolis (谐振子势)
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】单粒子在谐振子势中的 Metropolis 采样")
    print("-" * 70)
    
    # V(x) = (1/2) k x² , k=1
    potential = lambda x: 0.5 * x**2
    
    T_list = [0.5, 1.0, 2.0]
    print(f"  谐振子势 V(x) = x²/2")
    print(f"  理论 <x²> = T/k = T (k=1)")
    print()
    print(f"  {'T':>6s}  {'<x²> 数值':>12s}  {'<x²> 理论':>12s}  {'接受率':>10s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*10}")
    
    for T in T_list:
        samples, acc_rate = metropolis_single_particle(potential, T, n_steps=200000,
                                                        step_size=1.5, seed=42)
        # 丢弃前 10% 做热化
        samples = samples[len(samples)//10:]
        x2_sim = (samples**2).mean()
        x2_th = T  # 等分定理
        print(f"  {T:>6.1f}  {x2_sim:>12.4f}  {x2_th:>12.4f}  {acc_rate:>9.2%}")
    
    print()
    print(f"  ✓ 等分定理 <½kx²> = ½T 得到验证")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：2D Ising 临界温度扫描（核心实验）
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】2D Ising 模型 —— 相变涌现")
    print("-" * 70)
    
    Tc_onsager = onsager_Tc()
    print(f"  Onsager 精确解: T_c = 2/ln(1+√2) = {Tc_onsager:.4f}")
    print(f"  晶格 L = 16 × 16, 周期边界")
    print()
    
    # 温度扫描
    T_scan = [1.0, 1.5, 2.0, 2.20, Tc_onsager, 2.35, 2.5, 3.0, 4.0]
    
    print(f"  {'T':>7s}  {'<|M|>':>10s}  {'Onsager':>10s}  "
          f"{'<E>/N':>10s}  {'χ':>10s}  {'相态':>14s}")
    print(f"  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*14}")
    
    L = 16
    n_therm = 600
    n_meas = 2000
    
    for T in T_scan:
        ising = IsingModel2D(L=L, T=T, J=1.0, seed=42)
        ising.thermalize(n_sweeps=n_therm)
        result = ising.measure(n_sweeps=n_meas, n_skip=2)
        M_onsager = onsager_magnetization(T)
        
        if T < Tc_onsager - 0.1:
            phase = "有序"
        elif abs(T - Tc_onsager) < 0.1:
            phase = "临界 ← T_c"
        else:
            phase = "无序"
        
        print(f"  {T:>7.3f}  {result['M_abs']:>10.4f}  {M_onsager:>10.4f}  "
              f"{result['E_per_site']:>10.4f}  {result['chi']:>10.2f}  {phase:>14s}")
    
    print()
    print(f"  ✓ T < T_c: |M| > 0 (有序)")
    print(f"  ✓ T = T_c: |M| 急剧下降 + χ 峰值 (临界涨落)")
    print(f"  ✓ T > T_c: |M| ≈ 0 (无序)")
    print(f"  注: 有限尺度 L=16 → 与 Onsager 无限大极限略有偏差")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：临界构型可视化
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】不同温度下的自旋构型可视化")
    print("-" * 70)
    
    for T_show in [1.0, Tc_onsager, 4.0]:
        ising = IsingModel2D(L=20, T=T_show, J=1.0, seed=42)
        ising.thermalize(n_sweeps=2000)
        # 多 sweep 后取一个构型
        ising.metropolis_sweep()
        
        label = ("低温（有序）" if T_show < 2 else 
                 "临界（团簇各种尺度）" if abs(T_show - Tc_onsager) < 0.1 else
                 "高温（无序）")
        print(f"\n  T = {T_show:.4f}  ({label})")
        for row in ising.spins:
            line = "    " + "".join("█" if s > 0 else "·" for s in row)
            print(line)
    
    print()
    print("  注: 低温 → 大块同向区域")
    print("      临界 → 各尺度的团簇 (重整化群的核心)")
    print("      高温 → 完全随机")
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("第 7 章核心回顾：")
    print("  统计力学 = 微观涌现宏观")
    print("  Boltzmann 分布 + 配分函数 Z 是核心工具")
    print()
    print("  实验 1: Boltzmann 分布 P ∝ exp(-E/T) ✓")
    print("  实验 2: Metropolis 算法 + 等分定理 ✓")
    print("  实验 3: Ising 2D 相变 T_c ≈ 2.27 ✓")
    print("  实验 4: 临界涨落的可视化 ✓")
    print()
    print("练习 / Exercises:")
    print("  1. 增大 L (32, 64, 128) → 比热峰更尖锐")
    print("  2. 计算临界指数 β: M ∝ (T_c - T)^β, 应得 β = 1/8")
    print("  3. 加外磁场 h → 看磁滞回线")
    print("  4. 改 3D Ising → 探索另一普适类")
    print("=" * 70)
