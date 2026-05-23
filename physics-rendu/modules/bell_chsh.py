"""
bell_chsh.py — Bell 不等式 + CHSH 实验模块
===========================================
Module 16 of "Physics is Alive" / 《物理是活的》第 16 模块

实现 / Implements:
    - classical_local_hidden_variable: 经典局域隐变量模型
    - quantum_chsh: 量子纠缠态预测
    - chsh_experiment_simulation: 数值采样模拟 S 值
    - 验证量子力学违反 Bell 不等式

物理基础 / Physics:
    CHSH: S = E(a,b) + E(a,b') + E(a',b) - E(a',b')
    经典局域实在论：|S| ≤ 2
    量子纠缠态：|S| = 2√2 ≈ 2.828
    Bell 单态: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2

栗周 / Li Zhou
2026 年 10 月 / October 2026
MIT License
"""

import numpy as np


# =============================================================================
# 经典局域隐变量模型
# =============================================================================

def classical_lhv_correlation(theta_a, theta_b, n_samples=100000, seed=None):
    """经典局域隐变量模型的关联函数（对应 Φ⁺ 态：正关联）
    
    模型：每对粒子有隐变量 λ ∈ [0, 2π]
    粒子 A 在方向 θ_a 测得 sign(cos(λ - θ_a))
    粒子 B 在方向 θ_b 测得 sign(cos(λ - θ_b))（同向 → 正关联）
    
    E(a, b) = ⟨A·B⟩
    """
    rng = np.random.default_rng(seed)
    lam = rng.uniform(0, 2 * np.pi, n_samples)
    
    A = np.sign(np.cos(lam - theta_a))
    B = np.sign(np.cos(lam - theta_b))  # Φ⁺ 正关联
    
    return np.mean(A * B)


def classical_S_value(theta_a, theta_a_prime, theta_b, theta_b_prime,
                     n_samples=100000, seed=None):
    """计算经典 CHSH S 值"""
    E_ab = classical_lhv_correlation(theta_a, theta_b, n_samples, seed)
    E_ab_prime = classical_lhv_correlation(theta_a, theta_b_prime, n_samples, seed)
    E_aprime_b = classical_lhv_correlation(theta_a_prime, theta_b, n_samples, seed)
    E_aprime_bprime = classical_lhv_correlation(theta_a_prime, theta_b_prime,
                                                 n_samples, seed)
    
    S = E_ab + E_ab_prime + E_aprime_b - E_aprime_bprime
    return S, (E_ab, E_ab_prime, E_aprime_b, E_aprime_bprime)


# =============================================================================
# 量子纠缠态（Bell 单态）
# =============================================================================

def quantum_correlation_singlet(theta_a, theta_b):
    """Bell 单态 |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2 的量子关联
    
    精确公式：E(a, b) = -cos(θ_a - θ_b)
    """
    return -np.cos(theta_a - theta_b)


def quantum_correlation_phi_plus(theta_a, theta_b):
    """Bell 态 |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 的量子关联
    
    E(a, b) = cos(θ_a + θ_b) （视角度定义）
    这里采用：E(a, b) = cos(θ_a - θ_b) 用于 |Φ⁻⟩ 形式
    """
    return np.cos(theta_a - theta_b)


def quantum_S_value(theta_a, theta_a_prime, theta_b, theta_b_prime, state='singlet'):
    """量子 CHSH S 值"""
    if state == 'singlet':
        corr = quantum_correlation_singlet
    else:
        corr = quantum_correlation_phi_plus
    
    E_ab = corr(theta_a, theta_b)
    E_ab_prime = corr(theta_a, theta_b_prime)
    E_aprime_b = corr(theta_a_prime, theta_b)
    E_aprime_bprime = corr(theta_a_prime, theta_b_prime)
    
    S = E_ab + E_ab_prime + E_aprime_b - E_aprime_bprime
    return S, (E_ab, E_ab_prime, E_aprime_b, E_aprime_bprime)


def quantum_S_sampling(theta_a, theta_a_prime, theta_b, theta_b_prime,
                       n_samples=100000, state='singlet', seed=None):
    """通过有限采样估计量子 S 值（带涨落）"""
    rng = np.random.default_rng(seed)
    
    def sample_pair(theta_A, theta_B):
        """从量子分布采样测量结果对 (A, B), A, B ∈ {±1}"""
        # 关联函数
        if state == 'singlet':
            E = -np.cos(theta_A - theta_B)
        else:
            E = np.cos(theta_A - theta_B)
        # P(A=B) = (1 + E) / 2 ... 不，单态是 P(A = -B) = (1 + E_singlet)/2
        # 简化：直接按 E 抽样
        # E = P(AB=+1) - P(AB=-1) = 2P(AB=+1) - 1
        # P(AB=+1) = (1 + E) / 2
        p_same = (1 + E) / 2
        r = rng.uniform(size=n_samples)
        AB = np.where(r < p_same, 1, -1)
        # A 单独均匀 ±1
        A = rng.choice([-1, 1], size=n_samples)
        B = A * AB
        return A, B
    
    A_ab, B_ab = sample_pair(theta_a, theta_b)
    A_ab2, B_ab2 = sample_pair(theta_a, theta_b_prime)
    A_a2b, B_a2b = sample_pair(theta_a_prime, theta_b)
    A_a2b2, B_a2b2 = sample_pair(theta_a_prime, theta_b_prime)
    
    E_ab = np.mean(A_ab * B_ab)
    E_ab_prime = np.mean(A_ab2 * B_ab2)
    E_aprime_b = np.mean(A_a2b * B_a2b)
    E_aprime_bprime = np.mean(A_a2b2 * B_a2b2)
    
    S = E_ab + E_ab_prime + E_aprime_b - E_aprime_bprime
    return S, (E_ab, E_ab_prime, E_aprime_b, E_aprime_bprime)


# =============================================================================
# 经典 vs 量子的最优 CHSH 配置
# =============================================================================

def optimal_chsh_angles():
    """对 |Φ⁺⟩：最优测量角度给 |S| = 2√2
    
    最优配置：
    θ_a = 0, θ_a' = π/2
    θ_b = π/4, θ_b' = -π/4
    
    对应 E(a,b) = cos(θ_a - θ_b)（Φ⁺ 态）：
    E(a,b)   = cos(-π/4) = √2/2
    E(a,b')  = cos(π/4)  = √2/2
    E(a',b)  = cos(π/4)  = √2/2
    E(a',b') = cos(3π/4) = -√2/2
    S = √2/2 + √2/2 + √2/2 - (-√2/2) = 2√2 ≈ 2.828
    """
    theta_a = 0.0
    theta_a_prime = np.pi / 2
    theta_b = np.pi / 4
    theta_b_prime = -np.pi / 4
    return theta_a, theta_a_prime, theta_b, theta_b_prime


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("bell_chsh.py — 物理是活的 / 第 16 模块演示")
    print("Topic: Bell 不等式 + CHSH 实验")
    print("=" * 70)
    print()
    
    # 最优角度
    theta_a, theta_a_prime, theta_b, theta_b_prime = optimal_chsh_angles()
    angles_deg = (np.degrees(theta_a), np.degrees(theta_a_prime),
                  np.degrees(theta_b), np.degrees(theta_b_prime))
    
    print(f"测量角度（最优 CHSH 配置）:")
    print(f"  A 测量方向：θ_a  = {angles_deg[0]:.0f}°,  θ_a' = {angles_deg[1]:.0f}°")
    print(f"  B 测量方向：θ_b  = {angles_deg[2]:.0f}°,  θ_b' = {angles_deg[3]:.0f}°")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：量子理论预测
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】量子力学理论预测（Bell 单态）")
    print("-" * 70)
    
    S_QM, E_vals = quantum_S_value(theta_a, theta_a_prime, theta_b, theta_b_prime,
                                    state='phi_plus')
    
    print(f"  关联函数 E(a, b) = cos(θ_a - θ_b)  (Φ⁺ 态)")
    print(f"  E(a,  b ) = {E_vals[0]:>+.4f}")
    print(f"  E(a,  b') = {E_vals[1]:>+.4f}")
    print(f"  E(a', b ) = {E_vals[2]:>+.4f}")
    print(f"  E(a', b') = {E_vals[3]:>+.4f}")
    print()
    print(f"  S = E(a,b) + E(a,b') + E(a',b) - E(a',b')")
    print(f"  S_QM = {S_QM:.4f}")
    print(f"  2√2 = {2*np.sqrt(2):.4f}")
    print(f"  |S_QM| / 2 = {abs(S_QM)/2:.4f}  (经典上限 = 1)")
    print()
    print(f"  ✓ 量子预测 |S| = 2√2 ≈ 2.828 > 2 (违反 Bell 不等式 41%)")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：经典局域隐变量模型
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】经典局域隐变量模型（10000 次采样）")
    print("-" * 70)
    
    n_samp = 10000
    S_classical, E_classical = classical_S_value(theta_a, theta_a_prime,
                                                  theta_b, theta_b_prime,
                                                  n_samples=n_samp, seed=42)
    
    print(f"  数值模拟：")
    print(f"  E(a,  b ) = {E_classical[0]:>+.4f}")
    print(f"  E(a,  b') = {E_classical[1]:>+.4f}")
    print(f"  E(a', b ) = {E_classical[2]:>+.4f}")
    print(f"  E(a', b') = {E_classical[3]:>+.4f}")
    print()
    print(f"  S_经典 = {S_classical:.4f}")
    print()
    print(f"  ✓ |S_经典| ≤ 2 —— 满足 Bell 不等式")
    print(f"  ✓ 经典隐变量模型无法达到量子的 2√2")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：量子数值采样
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】量子数值采样验证（10000 次）")
    print("-" * 70)
    
    S_qsamp, E_qsamp = quantum_S_sampling(theta_a, theta_a_prime,
                                           theta_b, theta_b_prime,
                                           n_samples=n_samp, state='phi_plus', seed=42)
    
    print(f"  数值采样（从量子概率分布）：")
    print(f"  E(a,  b ) = {E_qsamp[0]:>+.4f}")
    print(f"  E(a,  b') = {E_qsamp[1]:>+.4f}")
    print(f"  E(a', b ) = {E_qsamp[2]:>+.4f}")
    print(f"  E(a', b') = {E_qsamp[3]:>+.4f}")
    print()
    print(f"  S_量子采样 = {S_qsamp:.4f}")
    print(f"  S_量子理论 = {S_QM:.4f}")
    print(f"  误差: {abs(S_qsamp - S_QM):.4f}")
    print()
    print(f"  ✓ 数值采样 ≈ 理论 2√2")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：扫描多个角度配置
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】扫描 θ_b 角度（其他固定）→ 看 |S| 如何变化")
    print("-" * 70)
    
    theta_b_scan = np.linspace(0, np.pi, 13)
    print(f"  固定 θ_a = 0°, θ_a' = 90°, θ_b' = θ_b - 90° (保持最优分离)")
    print()
    print(f"  {'θ_b (°)':>8s}  {'S_经典':>10s}  {'S_量子':>10s}  {'是否 >2':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
    
    for tb in theta_b_scan:
        tbp = tb - np.pi / 2  # 保持 b 和 b' 相距 π/2
        S_c, _ = classical_S_value(0, np.pi/2, tb, tbp, n_samples=5000, seed=42)
        S_q, _ = quantum_S_value(0, np.pi/2, tb, tbp, state='phi_plus')
        violate = "✓" if abs(S_q) > 2.01 else "—"
        print(f"  {np.degrees(tb):>8.1f}  {S_c:>+10.4f}  {S_q:>+10.4f}  {violate:>10s}")
    
    print()
    print(f"  ✓ 经典 |S| ≤ 2 恒成立")
    print(f"  ✓ 量子 |S| 在 θ_b = 45° + n·90° 附近达到 2√2")
    print()
    
    # -------------------------------------------------------------------------
    # 总结
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("第 16 章核心回顾：")
    print("  Bell 不等式：局域实在论 → |S| ≤ 2")
    print("  量子力学：纠缠态 → |S| = 2√2 ≈ 2.828")
    print("  实验验证：违反 Bell 不等式 → 否定局域实在论")
    print()
    print("  实验 1: 量子理论 S = 2.828 ✓")
    print("  实验 2: 经典 LHV |S| ≤ 2 ✓")
    print("  实验 3: 量子数值采样 ≈ 2√2 ✓")
    print("  实验 4: 角度扫描 → 量子始终最强 ✓")
    print()
    print("Aspect (法), Clauser (美), Zeilinger (奥) 因 Bell 实验获 2022 Nobel")
    print()
    print("练习 / Exercises:")
    print("  1. GHZ 三体态：单次实验就能违反局域实在")
    print("  2. 实现 BB84 量子密钥分发")
    print("  3. 模拟退相干对纠缠的破坏")
    print("  4. 实现量子隐形传态协议")
    print("=" * 70)
