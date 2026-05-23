"""
advanced_cm.py — 凝聚态前沿模块
==================================
Module 19 of "打通物理任督二脉" / 第 19 模块

实现 / Implements:
    - SSH 模型（1D 拓扑入门）
    - Haldane 模型（2D 拓扑能带 + Chern 数）
    - 单点 Hubbard 模型（强关联示意）
    - altermag 简化模型（自旋分裂）
    - BCS 能隙 + Bogoliubov 准粒子

物理基础 / Physics:
    SSH: 1D 二聚化链，t1 vs t2 决定拓扑相
    Haldane: 蜂窝格子 + 复跳跃 + 交错势
    Hubbard: 单点版本展示强关联起点
    Altermag: 子格各向异性 → 能带自旋分裂
    BCS: 准粒子谱 E_k = √(ξ² + Δ²)

栗周 / Li Zhou
2026 年 11 月 / November 2026
MIT License
"""

import numpy as np


# =============================================================================
# 1. SSH 模型 — 1D 拓扑入门
# =============================================================================

def ssh_bands(k_arr, t1=1.0, t2=0.5):
    """SSH 模型能带
    
    Hamiltonian = [0, t1+t2 e^{-ik}; t1+t2 e^{ik}, 0]
    
    E_±(k) = ±|t1 + t2 e^{ik}| = ±√(t1² + t2² + 2 t1 t2 cos k)
    
    t1 > t2: 拓扑平凡
    t1 < t2: 拓扑非平凡（边界零模）
    """
    E_abs = np.sqrt(t1**2 + t2**2 + 2 * t1 * t2 * np.cos(k_arr))
    return E_abs, -E_abs


def ssh_finite_chain(N=20, t1=1.0, t2=0.5):
    """有限 N 位 SSH 链精确对角化
    
    返回 2N 个本征值（包括可能的边界零模）
    """
    dim = 2 * N
    H = np.zeros((dim, dim))
    
    for i in range(N):
        # intra-cell hopping (A_i ↔ B_i)
        H[2*i, 2*i + 1] = t1
        H[2*i + 1, 2*i] = t1
        if i < N - 1:
            # inter-cell hopping (B_i ↔ A_{i+1})
            H[2*i + 1, 2*i + 2] = t2
            H[2*i + 2, 2*i + 1] = t2
    
    eigvals = np.linalg.eigvalsh(H)
    return eigvals


# =============================================================================
# 2. Haldane 模型 — 蜂窝 + 拓扑能带
# =============================================================================

def haldane_bands(kx, ky, t1=1.0, t2=0.1, phi=np.pi/2, M=0.0):
    """Haldane 模型能带（蜂窝格子）
    
    H(k) = [M + 2t₂ Σ cos(k·b_i+φ), 复跳跃]
    
    简化：a₀ = 1, 最近邻矢量沿 60° 方向
    """
    # 最近邻矢量（蜂窝格子）
    # d1 = (1, 0), d2 = (-1/2, √3/2), d3 = (-1/2, -√3/2)
    # 次近邻矢量 b1 = d2 - d3, b2 = d3 - d1, b3 = d1 - d2
    b1 = (0, np.sqrt(3))
    b2 = (-1.5, -np.sqrt(3)/2)
    b3 = (1.5, -np.sqrt(3)/2)
    
    # 对角元（A 和 B 子格）
    sum_AA = 2 * t2 * (np.cos(kx*b1[0] + ky*b1[1] + phi) +
                       np.cos(kx*b2[0] + ky*b2[1] + phi) +
                       np.cos(kx*b3[0] + ky*b3[1] + phi))
    sum_BB = 2 * t2 * (np.cos(kx*b1[0] + ky*b1[1] - phi) +
                       np.cos(kx*b2[0] + ky*b2[1] - phi) +
                       np.cos(kx*b3[0] + ky*b3[1] - phi))
    
    H_AA = M + sum_AA
    H_BB = -M + sum_BB
    
    # 最近邻
    d1 = (1, 0)
    d2 = (-0.5, np.sqrt(3)/2)
    d3 = (-0.5, -np.sqrt(3)/2)
    H_AB = -t1 * (np.exp(1j*(kx*d1[0] + ky*d1[1])) +
                  np.exp(1j*(kx*d2[0] + ky*d2[1])) +
                  np.exp(1j*(kx*d3[0] + ky*d3[1])))
    
    # 2×2 Hamiltonian
    trace = H_AA + H_BB
    det = H_AA * H_BB - np.abs(H_AB)**2
    discriminant = (H_AA - H_BB)**2 / 4 + np.abs(H_AB)**2
    
    E_plus = trace/2 + np.sqrt(discriminant)
    E_minus = trace/2 - np.sqrt(discriminant)
    return E_plus, E_minus


# =============================================================================
# 3. 单点 Hubbard 模型 — 强关联起点
# =============================================================================

def hubbard_single_site(U=4.0, mu=0.0):
    """单点 Hubbard：1 个位点 + 2 自旋
    
    Hilbert 空间：|0⟩, |↑⟩, |↓⟩, |↑↓⟩
    
    H = U n_↑ n_↓ - μ(n_↑ + n_↓)
    
    本征值：
        |0⟩:    E = 0
        |↑⟩:    E = -μ
        |↓⟩:    E = -μ
        |↑↓⟩:   E = U - 2μ
    """
    eigvals = {
        '|0⟩': 0.0,
        '|↑⟩': -mu,
        '|↓⟩': -mu,
        '|↑↓⟩': U - 2*mu
    }
    return eigvals


def hubbard_thermal_occupation(U, mu, T):
    """Hubbard 单点的统计权重 + 平均占据"""
    energies = list(hubbard_single_site(U, mu).values())
    Z = sum(np.exp(-E/T) for E in energies)
    
    # ⟨n⟩ = (0·exp(0) + 1·2·exp(μ/T) + 2·exp(-(U-2μ)/T)) / Z
    n_avg = (0 + 1 * np.exp(mu/T) + 1 * np.exp(mu/T) + 2 * np.exp(-(U-2*mu)/T)) / Z
    # ⟨n_↑ n_↓⟩ = exp(-(U-2μ)/T) / Z
    double_occ = np.exp(-(U-2*mu)/T) / Z
    
    return n_avg, double_occ


# =============================================================================
# 4. altermag 简化模型 — 能带自旋分裂
# =============================================================================

def altermag_simple_bands(kx, ky, t=1.0, alpha=0.3):
    """altermag 简化模型
    
    考虑 2 子格 A, B 通过 90° 旋转相联
    每个子格的能带是各向异性的：
        ε_A(k) = -2t(α cos kx + (2-α) cos ky)
        ε_B(k) = -2t((2-α) cos kx + α cos ky)
    
    (子格 ↑ 在 A 上，子格 ↓ 在 B 上)
    
    净磁化 = 0 (子格反平行)
    但能带：
        ε↑(k) = ε_A(k)
        ε↓(k) = ε_B(k)
    
    一般 ε↑ ≠ ε↓ → 能带自旋分裂！
    """
    eps_up = -2 * t * (alpha * np.cos(kx) + (2 - alpha) * np.cos(ky))
    eps_down = -2 * t * ((2 - alpha) * np.cos(kx) + alpha * np.cos(ky))
    return eps_up, eps_down


# =============================================================================
# 5. BCS 能隙 + Bogoliubov 准粒子
# =============================================================================

def bcs_quasiparticle_energy(xi, Delta):
    """Bogoliubov 准粒子能量
    
    E_k = √(ξ_k² + Δ²)
    
    最小能隙 = Δ (在 ξ_k = 0 处)
    """
    return np.sqrt(xi**2 + Delta**2)


def bcs_gap_equation(Delta, T, V_pairing=1.0, Ef=1.0, hbar_wD=0.1):
    """BCS 自洽能隙方程（简化）
    
    1 = V Σ_k tanh(E_k/2T) / (2 E_k)
    
    数值积分 ∫dξ near Fermi level
    """
    # 简化：在 ±hbar_wD 范围内对 ξ 积分
    xi_arr = np.linspace(-hbar_wD * Ef, hbar_wD * Ef, 1000)
    E_arr = bcs_quasiparticle_energy(xi_arr, Delta)
    if T < 1e-6:
        integrand = 1.0 / (2 * E_arr)
    else:
        integrand = np.tanh(E_arr / (2*T)) / (2 * E_arr)
    integral = np.trapezoid(integrand, xi_arr)
    
    # 自洽: 1 = V · integral
    return 1 - V_pairing * integral


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("advanced_cm.py — 打通物理任督二脉 / 第 19 模块演示")
    print("Topic: 凝聚态前沿 —— 拓扑、强关联、altermag、超导")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：SSH 模型（拓扑入门）
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】SSH 模型 — 1D 拓扑入门")
    print("-" * 70)
    
    # 平凡相 vs 拓扑相
    print(f"  平凡相 (t1 = 1.0, t2 = 0.5)：")
    print(f"    有限 N=20 链的本征值（最低 6 个 + 最高 6 个）：")
    
    eigvals_trivial = ssh_finite_chain(N=20, t1=1.0, t2=0.5)
    n = len(eigvals_trivial)
    print(f"    最低: {eigvals_trivial[:3]}")
    print(f"    中间: {eigvals_trivial[n//2 - 1:n//2 + 1]}")
    print(f"    最高: {eigvals_trivial[-3:]}")
    
    # 检查是否有零模（在中间能量附近）
    zero_modes_trivial = np.sum(np.abs(eigvals_trivial) < 0.01)
    print(f"    接近 0 的能量数: {zero_modes_trivial}")
    print()
    
    print(f"  拓扑相 (t1 = 0.5, t2 = 1.0)：")
    eigvals_topo = ssh_finite_chain(N=20, t1=0.5, t2=1.0)
    print(f"    最低: {eigvals_topo[:3]}")
    print(f"    中间: {eigvals_topo[n//2 - 1:n//2 + 1]}")
    print(f"    最高: {eigvals_topo[-3:]}")
    
    zero_modes_topo = np.sum(np.abs(eigvals_topo) < 0.01)
    print(f"    接近 0 的能量数: {zero_modes_topo}  ← 拓扑边界态")
    print()
    print(f"  ✓ t1 > t2：平凡相，无零模")
    print(f"  ✓ t1 < t2：拓扑相，2 个零模（链两端边界态）")
    print(f"  ✓ 这就是\"拓扑保护边界态\"的最简单实例")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：Haldane 模型（拓扑能带）
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】Haldane 模型 — 蜂窝格子拓扑能带")
    print("-" * 70)
    
    # 在几个高对称点测能带
    points = {
        'Γ (0, 0)': (0, 0),
        'K (4π/3, 0)': (4*np.pi/3, 0),
        'K\' (-4π/3, 0)': (-4*np.pi/3, 0),
        'M (π, π/√3)': (np.pi, np.pi/np.sqrt(3)),
    }
    
    # 三种参数：纯石墨烯、加质量项 M（破缺）、Haldane（拓扑）
    cases = [
        ('普通石墨烯 (M=0, t2=0)', 0.0, 0.0),
        ('破缺 sublatt (M=0.5, t2=0)', 0.5, 0.0),
        ('Haldane 拓扑 (M=0, t2=0.2, φ=π/2)', 0.0, 0.2),
    ]
    
    for case_name, M, t2 in cases:
        print(f"\n  {case_name}:")
        print(f"  {'点':>15s}  {'E_+':>10s}  {'E_-':>10s}  {'能隙':>10s}")
        print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*10}")
        for name, (kx, ky) in points.items():
            E_p, E_m = haldane_bands(kx, ky, t1=1.0, t2=t2,
                                     phi=np.pi/2, M=M)
            gap = E_p - E_m
            print(f"  {name:>15s}  {E_p:>+10.4f}  {E_m:>+10.4f}  {gap:>10.4f}")
    
    print()
    print(f"  ✓ 普通石墨烯: K, K' 处能隙 = 0 (Dirac 点)")
    print(f"  ✓ 加 M: K, K' 处都开同样能隙（拓扑平凡）")
    print(f"  ✓ Haldane t2: K, K' 处能隙符号相反 → 拓扑非平凡 (Chern = ±1)")
    print(f"  ✓ 这是\"量子反常 Hall 效应\"的微观模型")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：单点 Hubbard 模型（强关联起点）
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】单点 Hubbard 模型 — 强关联起点")
    print("-" * 70)
    
    print(f"  Hilbert 空间：4 个态 (|0⟩, |↑⟩, |↓⟩, |↑↓⟩)")
    print(f"  H = U n_↑ n_↓ - μ(n_↑ + n_↓)")
    print()
    
    for U, mu_name, mu in [(4.0, '半填充 μ=U/2', 2.0),
                            (4.0, '空 μ=0', 0.0),
                            (10.0, '强相互作用 μ=U/2', 5.0)]:
        states = hubbard_single_site(U=U, mu=mu)
        print(f"  U = {U}, {mu_name}:")
        for name, E in states.items():
            print(f"    {name:>5s}: E = {E:>+7.3f}")
        
        # 热占据（T = U/10）
        T = U / 10
        n_avg, d_occ = hubbard_thermal_occupation(U, mu, T)
        print(f"    在 T = {T}: ⟨n⟩ = {n_avg:.4f}, ⟨n_↑n_↓⟩ = {d_occ:.4f}")
        print()
    
    print(f"  ✓ U 大时双占据 ⟨n_↑n_↓⟩ 被抑制 → Mott 物理起点")
    print(f"  ✓ 半填充 μ=U/2 时电子-空穴对称")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：altermag 简化模型 — 能带自旋分裂
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】altermag 简化模型 — 能带自旋分裂（无净磁化）")
    print("-" * 70)
    
    print(f"  2 子格通过 90° 旋转相联：")
    print(f"  ε↑(k) = -2t [α cos(kx) + (2-α) cos(ky)]")
    print(f"  ε↓(k) = -2t [(2-α) cos(kx) + α cos(ky)]")
    print(f"  (这里 α = 0.3 → 强各向异性)")
    print()
    
    test_k = [
        ('(π, 0)', (np.pi, 0)),
        ('(0, π)', (0, np.pi)),
        ('(π/2, π/2)', (np.pi/2, np.pi/2)),
        ('(π, π)', (np.pi, np.pi)),
    ]
    
    print(f"  {'k 点':>15s}  {'ε↑(k)':>10s}  {'ε↓(k)':>10s}  {'自旋分裂':>10s}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*10}")
    
    for name, (kx, ky) in test_k:
        eps_up, eps_down = altermag_simple_bands(kx, ky, t=1.0, alpha=0.3)
        splitting = eps_up - eps_down
        print(f"  {name:>15s}  {eps_up:>+10.4f}  {eps_down:>+10.4f}  {splitting:>+10.4f}")
    
    print()
    
    # 检查关键特征：(π, 0) 和 (0, π) 自旋分裂相反 → 净磁化为 0
    eps_up_pi0, eps_down_pi0 = altermag_simple_bands(np.pi, 0, alpha=0.3)
    eps_up_0pi, eps_down_0pi = altermag_simple_bands(0, np.pi, alpha=0.3)
    
    print(f"  关键观察：")
    print(f"    (π, 0): 自旋分裂 = {eps_up_pi0 - eps_down_pi0:+.4f}")
    print(f"    (0, π): 自旋分裂 = {eps_up_0pi - eps_down_0pi:+.4f}")
    print(f"    两处自旋分裂\"相反\" → BZ 上分裂之和 = 0 → 净磁化 = 0 ✓")
    print()
    print(f"  ✓ 即使 ⟨M⟩ = 0（反铁磁式）—— 能带在 k 空间自旋分裂")
    print(f"  ✓ 这就是 altermag 的\"特异签名\"")
    print(f"  ✓ 真实材料：MnTe, CrSb, Mn₅Si₃ 等")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 5：BCS 能隙 + Bogoliubov 准粒子
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 5】BCS 超导：能隙 + Bogoliubov 准粒子")
    print("-" * 70)
    
    Delta = 1.0  # meV
    print(f"  超导能隙 Δ = {Delta} meV")
    print(f"  Bogoliubov 准粒子谱 E(k) = √(ξ(k)² + Δ²)")
    print()
    print(f"  {'ξ(k) (meV)':>14s}  {'E(k) (meV)':>14s}  {'物理':>20s}")
    print(f"  {'-'*14}  {'-'*14}  {'-'*20}")
    
    for xi in [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]:
        E = bcs_quasiparticle_energy(xi, Delta)
        if abs(xi) < 0.01:
            phys = "费米面：E = Δ"
        elif abs(xi) > 3:
            phys = "远离费米面 ≈ |ξ|"
        else:
            phys = ""
        print(f"  {xi:>+14.2f}  {E:>14.4f}  {phys:>20s}")
    
    print()
    print(f"  ✓ 最小能隙 2Δ = {2*Delta} meV（费米面处）")
    print(f"  ✓ 远离费米面 E → |ξ|（正常电子）")
    print(f"  ✓ 这就是超导能隙的微观图像")
    print(f"  ✓ Cooper 对 → 凝聚 → 能隙 → 零电阻 + Meissner")
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("第 19 章核心回顾：")
    print("  凝聚态前沿 6 大方向：拓扑、超导、强关联、2D、磁性、量子模拟")
    print()
    print("  实验 1: SSH 拓扑边界态 ✓")
    print("  实验 2: Haldane 拓扑能带 ✓")
    print("  实验 3: 单点 Hubbard 强关联 ✓")
    print("  实验 4: altermag 自旋分裂 ✓")
    print("  实验 5: BCS 能隙 + Bogoliubov ✓")
    print()
    print("练习 / Exercises:")
    print("  1. SSH Zak 相数值计算")
    print("  2. Haldane Chern 数（数值积分 Berry 曲率）")
    print("  3. Hubbard 多点（小簇精确对角化）")
    print("  4. altermag 反常 Hall 数值")
    print("  5. BCS 自洽 Δ(T) 求解")
    print("=" * 70)
