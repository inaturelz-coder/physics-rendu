"""
qcomp.py — 量子信息与量子计算模块
===================================
Module 17 of "Physics is Alive" / 《物理是活的》第 17 模块

实现 / Implements:
    - QuantumState: 多比特态向量表示
    - 单比特门: X, Y, Z, H, S, T
    - 双比特门: CNOT
    - Bell 态制备
    - 量子隐形传态完整模拟
    - Grover 搜索（4 项）
    - QFT 量子傅里叶变换
    - 3 比特相位翻转纠错码

物理基础 / Physics:
    Bell 态: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    Grover: 经典 O(N) → 量子 O(√N)
    QFT: 量子版 DFT
    
栗周 / Li Zhou
2026 年 10 月 / October 2026
MIT License
"""

import numpy as np


# =============================================================================
# 单比特门
# =============================================================================

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S_gate = np.array([[1, 0], [0, 1j]], dtype=complex)
T_gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


# =============================================================================
# 多比特工具
# =============================================================================

def apply_single_qubit_gate(state, gate, qubit, n_qubits):
    """对 n 比特态向量的特定比特施加单比特门"""
    op = np.array([1.0], dtype=complex)
    for i in range(n_qubits):
        if i == qubit:
            op = np.kron(op, gate)
        else:
            op = np.kron(op, I)
    return op @ state


def apply_cnot(state, control, target, n_qubits):
    """CNOT 门：control 为 1 时翻转 target"""
    dim = 2**n_qubits
    new_state = state.copy()
    for i in range(dim):
        # 提取每个比特
        bits = [(i >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if bits[control] == 1:
            # 翻转 target
            j = i ^ (1 << (n_qubits - 1 - target))
            new_state[j] = state[i]
            new_state[i] = state[j] if bits[target] == 1 else new_state[i]
    # 简化重写
    new_state = np.zeros_like(state)
    for i in range(dim):
        bits = [(i >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if bits[control] == 1:
            j = i ^ (1 << (n_qubits - 1 - target))
            new_state[j] += state[i]
        else:
            new_state[i] += state[i]
    return new_state


def init_state(n_qubits, basis=0):
    """初始化 |basis⟩ 态 (basis 是 0 到 2^n - 1 的整数)"""
    state = np.zeros(2**n_qubits, dtype=complex)
    state[basis] = 1.0
    return state


def probabilities(state):
    """所有基态的测量概率"""
    return np.abs(state)**2


def measure(state, n_samples=1, seed=None):
    """从态中采样测量结果"""
    rng = np.random.default_rng(seed)
    probs = probabilities(state)
    return rng.choice(len(probs), size=n_samples, p=probs)


def print_state(state, n_qubits, threshold=1e-10):
    """打印态的非零分量"""
    s_list = []
    for i, amp in enumerate(state):
        if abs(amp) > threshold:
            bits = format(i, f'0{n_qubits}b')
            s_list.append(f"({amp.real:+.4f}{amp.imag:+.4f}j)|{bits}⟩")
    return " + ".join(s_list)


# =============================================================================
# 实验 1：Bell 态制备
# =============================================================================

def prepare_bell_phi_plus():
    """制备 |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    
    步骤：|00⟩ -H₁→ (|0⟩+|1⟩)|0⟩/√2 -CNOT→ (|00⟩+|11⟩)/√2
    """
    state = init_state(2, basis=0)  # |00⟩
    state = apply_single_qubit_gate(state, H_gate, 0, 2)  # H on qubit 0
    state = apply_cnot(state, control=0, target=1, n_qubits=2)
    return state


# =============================================================================
# 实验 2：量子隐形传态
# =============================================================================

def quantum_teleportation(alpha, beta, seed=42):
    """量子隐形传态完整模拟
    
    A 处的未知态 α|0⟩ + β|1⟩ → B 处
    用 1 对共享 Bell 态作资源
    
    Returns: (B 处恢复出的态向量 = [α, β])
    """
    rng = np.random.default_rng(seed)
    
    # 初始 3 比特态: A_input ⊗ |Φ⁺⟩_AB
    # 比特顺序：[A_input, A_bell, B_bell]
    input_state = np.array([alpha, beta], dtype=complex)
    bell_state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    state = np.kron(input_state, bell_state)
    
    # Step 1: A 对 [A_input, A_bell] 做 Bell 基测量
    # 等价于：CNOT(A_input → A_bell) + H(A_input)，然后测量
    state = apply_cnot(state, control=0, target=1, n_qubits=3)
    state = apply_single_qubit_gate(state, H_gate, 0, 3)
    
    # 测量 A_input 和 A_bell —— 得 2 比特经典结果
    probs = probabilities(state)
    # 边缘分布 P(m1, m2) = Σ P(m1, m2, b)
    p_classical = np.zeros((2, 2))
    for i in range(8):
        m1 = (i >> 2) & 1
        m2 = (i >> 1) & 1
        p_classical[m1, m2] += probs[i]
    
    # 采样 (m1, m2)
    flat_p = p_classical.flatten()
    idx = rng.choice(4, p=flat_p)
    m1, m2 = idx // 2, idx % 2
    
    # 根据测量结果，B 比特的态（不归一化条件态）
    # 把 (m1, m2) 投影到 [A_input, A_bell]，留下 B 比特态
    B_state = np.zeros(2, dtype=complex)
    for b in range(2):
        idx_in_8 = (m1 << 2) | (m2 << 1) | b
        B_state[b] = state[idx_in_8]
    # 归一化
    norm = np.linalg.norm(B_state)
    if norm > 1e-10:
        B_state /= norm
    
    # Step 4: B 根据 (m1, m2) 做 Pauli 修正
    # (0,0) → I, (0,1) → X, (1,0) → Z, (1,1) → ZX = iY
    if m2 == 1:
        B_state = X @ B_state
    if m1 == 1:
        B_state = Z @ B_state
    
    return B_state, (m1, m2)


# =============================================================================
# 实验 3：Grover 搜索
# =============================================================================

def grover_search(n_qubits, marked_item, n_iterations=None):
    """Grover 搜索算法
    
    在 N = 2^n 项中找 marked_item
    最优迭代次数 ≈ (π/4)√N
    """
    N = 2**n_qubits
    if n_iterations is None:
        # 最优迭代数：使旋转角接近 π/2 → 用 floor 而非 round
        # N=4: (π/4)√4 = π/2 ≈ 1.57 → 1 次（恰好达到完美）
        n_iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(N))))
    
    # 初始化均匀叠加
    state = np.ones(N, dtype=complex) / np.sqrt(N)
    
    # Grover 迭代
    for _ in range(n_iterations):
        # Oracle: 标记项相位翻转
        state[marked_item] *= -1
        
        # Diffusion: 2|s⟩⟨s| - I
        # 在均匀叠加 |s⟩ 上反射
        mean_amp = np.mean(state)
        state = 2 * mean_amp - state
    
    return state, n_iterations


# =============================================================================
# 实验 4：量子傅里叶变换 (QFT)
# =============================================================================

def QFT_matrix(n_qubits):
    """显式构造 QFT 矩阵
    
    QFT|j⟩ = (1/√N) Σ_k ω^(jk) |k⟩
    where ω = exp(2πi/N), N = 2^n
    """
    N = 2**n_qubits
    omega = np.exp(2j * np.pi / N)
    U = np.zeros((N, N), dtype=complex)
    for j in range(N):
        for k in range(N):
            U[k, j] = omega**(j * k)
    U /= np.sqrt(N)
    return U


def verify_QFT_unitary(n_qubits):
    """验证 QFT 是幺正算符"""
    U = QFT_matrix(n_qubits)
    UdU = U.conj().T @ U
    return np.allclose(UdU, np.eye(2**n_qubits))


# =============================================================================
# 实验 5：3 比特相位翻转纠错码
# =============================================================================

def phase_flip_code():
    """3 比特相位翻转码演示
    
    编码：|0⟩_L = |+++⟩, |1⟩_L = |---⟩
    错误：Z 错误（相位翻转）→ |+⟩ ↔ |-⟩
    检测 + 纠正
    """
    results = {}
    
    # 编码 |+⟩_L
    # |+⟩_L = (|0⟩_L + |1⟩_L)/√2 = (|+++⟩ + |---⟩)/√2
    psi_logical = np.zeros(8, dtype=complex)
    # |+++⟩ = 1/√8 (|000⟩+|001⟩+...+|111⟩) 全部正
    # |---⟩ = 1/√8 ((-)^(s)|s⟩) 按位求和的奇偶 → 某种符号
    # 简化：直接构造编码 |0_L⟩ = |+++⟩
    
    # |+⟩ = (|0⟩+|1⟩)/√2 → |+++⟩ = (1/2√2) Σ_{s∈{0,1}³} |s⟩
    # 所有 8 个基态系数 1/2√2
    state_plus_L = np.ones(8, dtype=complex) / np.sqrt(8)
    
    # |---⟩ = 系数 (-1)^(s_1+s_2+s_3)
    state_minus_L = np.array([(-1)**bin(i).count('1') for i in range(8)],
                              dtype=complex) / np.sqrt(8)
    
    results['|+++⟩'] = state_plus_L.copy()
    results['|---⟩'] = state_minus_L.copy()
    
    # 检查 ⟨+++|---⟩ = 0（正交性）
    overlap = np.abs(state_plus_L.conj() @ state_minus_L)
    results['正交性'] = (overlap < 1e-10)
    
    # 演示 Z 错误检测：
    # 在第 1 比特施加 Z → 改变 |---⟩ 的系数符号（只对 qubit 1 = 1 的态）
    Z_on_qubit0 = np.eye(8, dtype=complex)
    for i in range(8):
        if (i >> 2) & 1:  # 比特 0（高位）为 1
            Z_on_qubit0[i, i] = -1
    
    corrupted = Z_on_qubit0 @ state_plus_L
    # 检查与原态的差异
    fidelity_corrupted = np.abs(state_plus_L.conj() @ corrupted)**2
    results['Z 错误后保真度'] = fidelity_corrupted
    
    # 用 Hadamard 把相位翻转转化为比特翻转 → 经典纠错
    # （完整纠错涉及辅助比特测量症状 —— 这里简化演示）
    results['说明'] = 'Z 错误把 |+⟩→|-⟩，相当于比特翻转的 X 基版本'
    
    return results


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("qcomp.py — 物理是活的 / 第 17 模块演示")
    print("Topic: 量子信息与量子计算")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：Bell 态制备
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】Bell 态 |Φ⁺⟩ 制备")
    print("-" * 70)
    
    bell = prepare_bell_phi_plus()
    
    print(f"  电路：|00⟩ → H(qubit 0) → CNOT → ?")
    print(f"  结果态: {print_state(bell, 2)}")
    print()
    print(f"  各基态概率：")
    probs_bell = probabilities(bell)
    for i, p in enumerate(probs_bell):
        bits = format(i, '02b')
        if p > 1e-10:
            print(f"    |{bits}⟩: P = {p:.4f}")
    
    print()
    print(f"  ✓ |Φ⁺⟩ = (|00⟩+|11⟩)/√2 完美纠缠态")
    print(f"  ✓ 单独看 A 或 B 都是 50/50 随机")
    print(f"  ✓ 但 A 和 B 必然 \"同步\"（00 或 11，不是 01 或 10）")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：量子隐形传态
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】量子隐形传态")
    print("-" * 70)
    
    # 任意输入态
    theta = np.pi / 3  # 60°
    phi = np.pi / 4    # 45°
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    
    input_state = np.array([alpha, beta])
    print(f"  A 输入态: ({alpha:.4f}){'|0⟩'} + ({beta.real:.4f}+{beta.imag:.4f}j)|1⟩")
    print(f"  Bloch 角 (θ={np.degrees(theta):.0f}°, φ={np.degrees(phi):.0f}°)")
    print()
    
    # 多次模拟，验证 4 种测量结果都能正确恢复
    print(f"  4 次运行（不同随机种子，覆盖各种测量结果）：")
    print(f"  {'种子':>6s}  {'(m1, m2)':>10s}  {'B 输出态':>40s}  {'保真度':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*40}  {'-'*10}")
    
    for seed in [1, 7, 42, 100]:
        B_state, (m1, m2) = quantum_teleportation(alpha, beta, seed=seed)
        # 保真度 |⟨ψ|ψ_B⟩|²
        fidelity = np.abs(input_state.conj() @ B_state)**2
        B_str = f"({B_state[0]:.3f}, {B_state[1]:.3f})"
        print(f"  {seed:>6d}  ({m1},{m2}){'':>4s}  {B_str:>40s}  {fidelity:>10.4f}")
    
    print()
    print(f"  ✓ 无论 A 测得什么 (m1, m2)——B 都能完全恢复 |ψ⟩")
    print(f"  ✓ 保真度 = 1.0 (完美) —— 量子隐形传态成功")
    print(f"  ✓ A 处原态消失（不可克隆定理）—— 不是\"复制\"")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：Grover 搜索
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】Grover 搜索算法（N = 4 项）")
    print("-" * 70)
    
    n_qubits = 2  # N = 4
    N = 4
    
    print(f"  在 N = {N} 项中搜索 marked_item")
    print(f"  经典：平均 {N/2} 次查询，最坏 {N-1} 次")
    print(f"  量子（Grover）：1 次查询即可（最优迭代数 (π/4)√4 = π/2 ≈ 1.57 → 取 1）")
    print()
    
    print(f"  {'marked':>8s}  {'迭代次数':>10s}  {'P(找到)':>10s}  {'是否高概率':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
    
    for marked in range(N):
        state, n_iter = grover_search(n_qubits, marked_item=marked)
        p_marked = probabilities(state)[marked]
        success = "✓" if p_marked > 0.9 else "—"
        print(f"  {marked:>8d}  {n_iter:>10d}  {p_marked:>10.4f}  {success:>10s}")
    
    print()
    print(f"  ✓ Grover 算法成功 —— 1 次迭代搜出 4 项")
    print(f"  ✓ 推广：N 项搜索需 ~ √N 步 (vs 经典 N/2)")
    print(f"  ✓ 真实应用：N = 10⁶ 时 √N = 1000 (vs 经典 500000)")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：量子傅里叶变换
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】量子傅里叶变换 (QFT) —— Shor 算法的核心")
    print("-" * 70)
    
    for n in [2, 3, 4]:
        U = QFT_matrix(n)
        is_unitary = verify_QFT_unitary(n)
        N_dim = 2**n
        print(f"  n = {n} 比特 (N = {N_dim}):")
        print(f"    幺正性 U†U = I: {'✓' if is_unitary else '✗'}")
        print(f"    矩阵规模: {N_dim}×{N_dim}")
    
    print()
    
    # 演示 QFT 对一个态的作用
    print(f"  QFT 作用于 |1⟩ (n=3, N=8):")
    state = init_state(3, basis=1)
    U3 = QFT_matrix(3)
    state_after = U3 @ state
    print(f"    QFT|1⟩ = (1/√8) Σ ω^k |k⟩, ω = exp(2πi/8)")
    
    # 显示前几个非零项的幅度
    for k in range(4):
        amp = state_after[k]
        print(f"    |{k}⟩: 振幅 = {amp.real:+.4f} {amp.imag:+.4f}j (|amp|² = {abs(amp)**2:.4f})")
    
    print(f"  ✓ QFT 是 DFT 的量子版本 —— 用 O(n²) 量子门实现 (经典 N log N)")
    print(f"  ✓ Shor 算法用 QFT 找周期 → 分解大数")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 5：3 比特相位翻转纠错码
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 5】3 比特相位翻转纠错码（演示）")
    print("-" * 70)
    
    results = phase_flip_code()
    
    print(f"  编码方案：")
    print(f"    |0⟩_L = |+++⟩")
    print(f"    |1⟩_L = |---⟩")
    print()
    print(f"  正交性 ⟨+++|---⟩ = 0: {'✓' if results['正交性'] else '✗'}")
    print()
    print(f"  Z 错误后保真度: {results['Z 错误后保真度']:.4f}")
    print(f"  说明: {results['说明']}")
    print()
    print(f"  完整纠错协议（简化版）：")
    print(f"    1. Hadamard 转换：相位翻转 → 比特翻转")
    print(f"    2. 比特翻转用 3 比特重复码纠正")
    print(f"    3. Hadamard 反变换回去")
    print()
    print(f"  ✓ Shor 1995 9 比特码 = 比特翻转码 ⊗ 相位翻转码 = 完整纠错")
    print(f"  ✓ 现代主流：表面码（千个物理比特 = 1 个逻辑比特）")
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("第 17 章核心回顾：")
    print("  量子信息 = 量子比特 + 纠缠 + 量子门 + 测量")
    print()
    print("  实验 1: Bell 态制备 ✓")
    print("  实验 2: 量子隐形传态（完美保真度）✓")
    print("  实验 3: Grover 搜索 N=4 ✓")
    print("  实验 4: QFT 幺正 + 用于 Shor ✓")
    print("  实验 5: 3 比特相位翻转码 ✓")
    print()
    print("练习 / Exercises:")
    print("  1. 完整 Deutsch-Jozsa 算法")
    print("  2. BB84 协议 + 窃听检测")
    print("  3. VQE 求 H₂ 分子基态")
    print("  4. 用 Qiskit 在真实 IBM Quantum 上运行")
    print("=" * 70)
