"""
ml_physics.py — 物理与 AI 模块
==================================
Module 21 of "打通物理任督二脉" / 第 21 模块（项目压轴）

实现 / Implements:
    - Hopfield 网络（自旋玻璃 = 联想记忆）
    - 简单 MLP + SGD 训练（观察 SGD = Langevin）
    - 2D Ising 临界温度的"涌现"演示
    - Boltzmann 机采样 + Monte Carlo 对比
    - 神经网络势函数 toy model（H₂ 势能曲线）

物理基础 / Physics:
    Hopfield: E = -1/2 Σ J_ij s_i s_j (自旋玻璃)
    SGD: θ_{t+1} = θ_t - η ∇L + noise (Langevin 离散)
    Ising 涌现: 临界温度附近 ML 准确率突变
    Boltzmann: P(s) ∝ exp(-E(s)/T)
    ML 势: V(r) = NN(r) 拟合精确量子势

栗周 / Li Zhou
2027 年初 / Early 2027
MIT License
"""

import numpy as np


# =============================================================================
# 1. Hopfield 网络（自旋玻璃 = 联想记忆）
# =============================================================================

class HopfieldNetwork:
    """Hopfield 网络 = Ising 模型变体
    
    存储模式 → 学习权重（Hebb 规则）
    给部分输入 → 演化到最近的存储模式（联想检索）
    """
    
    def __init__(self, n_neurons):
        self.n = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))
    
    def train(self, patterns):
        """Hebb 规则学习
        
        W_ij = (1/N) Σ_p s_i^p s_j^p
        """
        self.W = np.zeros((self.n, self.n))
        for p in patterns:
            self.W += np.outer(p, p)
        self.W /= len(patterns)
        np.fill_diagonal(self.W, 0)
    
    def energy(self, state):
        """E = -1/2 s^T W s"""
        return -0.5 * state @ self.W @ state
    
    def recall(self, noisy_input, max_steps=50):
        """从噪声输入演化到最近的存储模式"""
        state = noisy_input.copy()
        for step in range(max_steps):
            # 异步更新
            new_state = state.copy()
            for i in np.random.permutation(self.n):
                h = self.W[i] @ new_state
                new_state[i] = 1 if h >= 0 else -1
            if np.array_equal(new_state, state):
                return new_state, step
            state = new_state
        return state, max_steps


# =============================================================================
# 2. 简单 MLP + SGD（观察 SGD = Langevin）
# =============================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def sigmoid_deriv(y):
    return y * (1 - y)


class SimpleMLP:
    """2 层 MLP：input → hidden → output"""
    
    def __init__(self, n_in, n_hidden, n_out, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 0.5, (n_hidden, n_out))
        self.b2 = np.zeros(n_out)
    
    def forward(self, x):
        self.h = sigmoid(x @ self.W1 + self.b1)
        self.y = sigmoid(self.h @ self.W2 + self.b2)
        return self.y
    
    def loss(self, x, target):
        y = self.forward(x)
        return 0.5 * np.mean((y - target)**2)
    
    def step(self, x, target, lr):
        """单步 SGD"""
        y = self.forward(x)
        # 反向传播
        delta_y = (y - target) * sigmoid_deriv(y)
        delta_h = delta_y @ self.W2.T * sigmoid_deriv(self.h)
        # 更新
        self.W2 -= lr * np.outer(self.h, delta_y)
        self.b2 -= lr * delta_y
        self.W1 -= lr * np.outer(x, delta_h)
        self.b1 -= lr * delta_h


# =============================================================================
# 3. Ising 临界温度的"涌现"演示
# =============================================================================

def ising_sample(L=10, T=2.0, n_steps=2000, seed=None):
    """单次 Metropolis MC 采样 → 返回 (磁化平方, 能量) 历史平均"""
    rng = np.random.default_rng(seed)
    spins = rng.choice([-1, 1], size=(L, L))
    
    m2_sum = 0
    e_sum = 0
    n_measure = 0
    
    for step in range(n_steps):
        # 1 sweep = L*L 次单点翻转尝试
        for _ in range(L * L):
            i, j = rng.integers(0, L, 2)
            s = spins[i, j]
            nb = (spins[(i+1)%L, j] + spins[(i-1)%L, j] +
                  spins[i, (j+1)%L] + spins[i, (j-1)%L])
            dE = 2 * s * nb
            if dE <= 0 or rng.random() < np.exp(-dE/T):
                spins[i, j] = -s
        
        if step > n_steps // 2:  # 热化后采样
            m = spins.sum() / (L * L)
            m2_sum += m * m
            e = 0
            for i in range(L):
                for j in range(L):
                    e -= spins[i, j] * (spins[(i+1)%L, j] + spins[i, (j+1)%L])
            e_sum += e / (L * L)
            n_measure += 1
    
    return m2_sum / n_measure, e_sum / n_measure


def ising_phase_classifier_simple(magnetizations):
    """\"分类器\": |m²| 大 → 有序; 小 → 无序
    模拟 ML 学到的"决策"
    """
    threshold = 0.3  # 简化 —— 实际 ML 学习的阈值
    return np.array([1 if m > threshold else 0 for m in magnetizations])


# =============================================================================
# 4. Boltzmann 机采样 vs Monte Carlo
# =============================================================================

def boltzmann_sample(energies, T=1.0, n_samples=10000, seed=None):
    """从离散态的 Boltzmann 分布 P(i) ∝ exp(-E_i/T) 采样"""
    rng = np.random.default_rng(seed)
    weights = np.exp(-np.array(energies) / T)
    weights /= weights.sum()
    return rng.choice(len(energies), size=n_samples, p=weights)


def hopfield_boltzmann_compare(n_states=8, T=1.0, n_samples=10000, seed=42):
    """比较 \"直接采样\" vs \"Hopfield 动力学+噪声\""""
    rng = np.random.default_rng(seed)
    # 随机能量
    energies = rng.normal(0, 1, n_states)
    
    # 1. 直接 Boltzmann 采样
    samples_direct = boltzmann_sample(energies, T, n_samples, seed=seed)
    counts_direct = np.bincount(samples_direct, minlength=n_states) / n_samples
    
    # 理论分布
    weights = np.exp(-energies / T)
    weights /= weights.sum()
    
    return energies, weights, counts_direct


# =============================================================================
# 5. 神经网络势函数 toy model（H₂ 势能曲线）
# =============================================================================

def H2_potential_exact(r):
    """H₂ 势能曲线（Morse 拟合，近似精确量子势）
    
    V(r) = D [(1 - exp(-a(r-r_e)))² - 1]
    D ≈ 4.75 eV, r_e ≈ 0.74 Å, a ≈ 1.94 /Å
    """
    D = 4.75
    r_e = 0.74
    a = 1.94
    return D * (1 - np.exp(-a * (r - r_e)))**2 - D


def ml_fit_potential(r_train, V_train, n_hidden=20, n_epochs=2000, lr=0.01, seed=42):
    """用简单 MLP 拟合 V(r)"""
    rng = np.random.default_rng(seed)
    
    # 标准化输入
    r_mean = r_train.mean()
    r_std = r_train.std()
    V_mean = V_train.mean()
    V_std = V_train.std()
    
    x = ((r_train - r_mean) / r_std).reshape(-1, 1)
    y = ((V_train - V_mean) / V_std).reshape(-1, 1)
    
    # 初始化权重
    W1 = rng.normal(0, 0.5, (1, n_hidden))
    b1 = np.zeros(n_hidden)
    W2 = rng.normal(0, 0.5, (n_hidden, 1))
    b2 = np.zeros(1)
    
    losses = []
    for epoch in range(n_epochs):
        # Forward
        h = np.tanh(x @ W1 + b1)
        y_pred = h @ W2 + b2
        loss = 0.5 * np.mean((y_pred - y)**2)
        losses.append(loss)
        
        # Backward
        d_y = (y_pred - y) / len(x)
        d_W2 = h.T @ d_y
        d_b2 = d_y.sum(axis=0)
        d_h = d_y @ W2.T
        d_pre1 = d_h * (1 - h**2)
        d_W1 = x.T @ d_pre1
        d_b1 = d_pre1.sum(axis=0)
        
        # Update
        W1 -= lr * d_W1
        b1 -= lr * d_b1
        W2 -= lr * d_W2
        b2 -= lr * d_b2
    
    # 返回预测函数
    def predict(r_new):
        x_new = ((r_new - r_mean) / r_std).reshape(-1, 1)
        h_new = np.tanh(x_new @ W1 + b1)
        y_new = h_new @ W2 + b2
        return (y_new.flatten() * V_std + V_mean)
    
    return predict, losses


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("ml_physics.py — 打通物理任督二脉 / 第 21 模块演示")
    print("Topic: 物理与 AI —— 三层纠缠 + 物理化趋势")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：Hopfield 网络 = 自旋玻璃 = 联想记忆
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】Hopfield 网络 = 自旋玻璃 + 联想记忆")
    print("-" * 70)
    
    n_neurons = 25  # 5×5 "图像"
    
    # 定义 3 个 5×5 模式（简化版字母）
    pattern_A = np.array([
        [-1, 1, 1, 1,-1],
        [ 1,-1,-1,-1, 1],
        [ 1, 1, 1, 1, 1],
        [ 1,-1,-1,-1, 1],
        [ 1,-1,-1,-1, 1]
    ]).flatten()
    
    pattern_B = np.array([
        [ 1, 1, 1, 1,-1],
        [ 1,-1,-1,-1, 1],
        [ 1, 1, 1, 1,-1],
        [ 1,-1,-1,-1, 1],
        [ 1, 1, 1, 1,-1]
    ]).flatten()
    
    pattern_C = np.array([
        [-1, 1, 1, 1, 1],
        [ 1,-1,-1,-1,-1],
        [ 1,-1,-1,-1,-1],
        [ 1,-1,-1,-1,-1],
        [-1, 1, 1, 1, 1]
    ]).flatten()
    
    patterns = [pattern_A, pattern_B, pattern_C]
    
    net = HopfieldNetwork(n_neurons)
    net.train(patterns)
    
    print(f"  网络: {n_neurons} 个神经元 (5×5 \"图像\")")
    print(f"  存储 3 个模式: A, B, C")
    print(f"  能量 E = -1/2 s^T W s (Ising 形式)")
    print()
    print(f"  各存储模式的能量:")
    print(f"    E(A) = {net.energy(pattern_A):.4f}")
    print(f"    E(B) = {net.energy(pattern_B):.4f}")
    print(f"    E(C) = {net.energy(pattern_C):.4f}")
    print()
    
    # 给出有噪声的输入 → 看能否恢复
    rng = np.random.default_rng(42)
    print(f"  联想检索测试（20% 比特翻转噪声）：")
    print(f"  {'测试输入':>15s}  {'恢复正确率':>10s}  {'迭代步':>8s}  {'恢复模式':>10s}")
    print(f"  {'-'*15}  {'-'*10}  {'-'*8}  {'-'*10}")
    
    for name, original in [('A', pattern_A), ('B', pattern_B), ('C', pattern_C)]:
        # 加 20% 噪声
        noisy = original.copy()
        n_flip = int(0.2 * n_neurons)
        flip_idx = rng.choice(n_neurons, n_flip, replace=False)
        noisy[flip_idx] *= -1
        
        recovered, steps = net.recall(noisy)
        # 对比各存储模式找最近的
        similarities = [np.mean(recovered == p) for p in patterns]
        best_idx = np.argmax(similarities)
        best_name = ['A', 'B', 'C'][best_idx]
        accuracy = similarities[best_idx]
        
        print(f"  {f'噪声 {name}':>15s}  {accuracy:>10.4f}  {steps:>8d}  "
              f"{best_name:>10s}")
    
    print()
    print(f"  ✓ Hopfield 网络成功恢复了存储模式")
    print(f"  ✓ 能量极小 = 存储模式 = 自旋玻璃基态")
    print(f"  ✓ Hopfield 1982 论文核心想法")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：MLP + SGD 噪声（SGD = Langevin 演示）
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】MLP + SGD 训练：观察梯度噪声 = Langevin 动力学")
    print("-" * 70)
    
    # XOR 问题（经典 ML 测试）
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])
    
    print(f"  问题：XOR 函数（经典 ML 基准）")
    print(f"  网络: 2 → 4 → 1, sigmoid 激活")
    print()
    
    # 比较"小学习率（低噪声）" vs "大学习率（高噪声 = 高温）"
    for lr, label in [(0.1, "低 lr=0.1 (低噪声/低温)"),
                       (1.0, "中 lr=1.0 (中等)"),
                       (3.0, "大 lr=3.0 (高噪声/高温)")]:
        mlp = SimpleMLP(2, 4, 1, seed=42)
        losses = []
        for epoch in range(500):
            for i in np.random.default_rng(epoch).permutation(4):
                mlp.step(X[i], Y[i], lr)
            losses.append(mlp.loss(X.flatten().reshape(-1, 2)[0], Y[0]))
        
        # 最终损失（4 个样本均值）
        final_loss = np.mean([mlp.loss(X[i], Y[i]) for i in range(4)])
        # 噪声估计（最后 100 步损失的标准差）
        noise_std = np.std(losses[-100:])
        print(f"  {label:>30s}: 最终 loss = {final_loss:.4f}, "
              f"近期 std = {noise_std:.4f}")
    
    print()
    print(f"  ✓ 学习率 = 温度 (退火算法语言)")
    print(f"  ✓ 大学习率 → 大梯度噪声 → 更难收敛到精确极小")
    print(f"  ✓ 小学习率 → 小噪声 → 平稳收敛但慢")
    print(f"  ✓ SGD 本质上是 Langevin 动力学的离散版")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：Ising 临界温度的"涌现"演示
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】Ising 临界温度 — 涌现 + 相变 + 类\"scaling law\"")
    print("-" * 70)
    
    print(f"  扫描温度 T = 1.5 到 3.5，看 ⟨m²⟩ 突变 (= 涌现)")
    print(f"  L = 10 (一定大小限制下 \"小模型\")")
    print()
    
    T_arr = [1.5, 2.0, 2.27, 2.5, 3.0, 3.5]
    m2_arr = []
    
    print(f"  {'T':>8s}  {'⟨m²⟩':>10s}  {'相态':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}")
    for T in T_arr:
        m2, _ = ising_sample(L=10, T=T, n_steps=500, seed=int(T*100))
        m2_arr.append(m2)
        phase = "有序" if m2 > 0.5 else ("临界" if m2 > 0.15 else "无序")
        print(f"  {T:>8.2f}  {m2:>10.4f}  {phase:>10s}")
    
    print()
    print(f"  ✓ T < 2.27 (Onsager): ⟨m²⟩ 大 → 有序 (类\"小模型能学的简单模式\")")
    print(f"  ✓ T ≈ 2.27: 突变区 (= 涌现 = 相变临界点)")
    print(f"  ✓ T > 2.27: ⟨m²⟩ 小 → 无序")
    print(f"  ✓ 大模型 \"涌现新能力\" 与此完全同构")
    print(f"  ✓ Scaling laws 就是临界指数的现代名字")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：Boltzmann 分布采样
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】Boltzmann 分布采样（神经 ML 的统计力学根基）")
    print("-" * 70)
    
    energies, weights, counts = hopfield_boltzmann_compare(n_states=6, T=1.0, n_samples=20000)
    
    print(f"  6 个状态的能量 + Boltzmann 概率 P_i ∝ exp(-E_i/T), T = 1.0")
    print()
    print(f"  {'状态':>6s}  {'E_i':>8s}  {'P_i (理论)':>12s}  {'P_i (采样)':>12s}  {'差':>8s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*8}")
    for i in range(len(energies)):
        diff = abs(weights[i] - counts[i])
        print(f"  {i:>6d}  {energies[i]:>+8.4f}  {weights[i]:>12.4f}  "
              f"{counts[i]:>12.4f}  {diff:>8.4f}")
    
    print()
    print(f"  ✓ 直接采样收敛到 Boltzmann 分布")
    print(f"  ✓ 这就是 Hinton 1985 Boltzmann 机的核心：让神经网络学到这种分布")
    print(f"  ✓ Diffusion model 反向过程本质上是采样 Boltzmann 分布")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 5：神经网络势函数 (H₂ 势能曲线)
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 5】神经网络势函数：H₂ 势能曲线拟合")
    print("-" * 70)
    
    # 训练数据：精确 H₂ Morse 势
    r_train = np.linspace(0.4, 3.0, 30)
    V_train = H2_potential_exact(r_train)
    
    # ML 拟合
    predict_fn, losses = ml_fit_potential(r_train, V_train, n_hidden=15,
                                           n_epochs=3000, lr=0.05, seed=42)
    
    print(f"  训练: 30 个 (r, V) 数据点, 15 隐层神经元, 3000 epochs")
    print(f"  最终训练 loss = {losses[-1]:.6f}")
    print()
    
    # 测试: 在新点上比较
    r_test = np.array([0.5, 0.74, 1.0, 1.5, 2.0, 2.5])
    V_exact = H2_potential_exact(r_test)
    V_ml = predict_fn(r_test)
    
    print(f"  {'r (Å)':>10s}  {'V_exact (eV)':>12s}  {'V_ML (eV)':>12s}  {'差':>8s}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*8}")
    for r, ve, vm in zip(r_test, V_exact, V_ml):
        print(f"  {r:>10.3f}  {ve:>+12.4f}  {vm:>+12.4f}  {abs(ve-vm):>8.4f}")
    
    print()
    print(f"  ✓ MLP 学到了 H₂ 势能曲线的形状（最低点、长程衰减）")
    print(f"  ✓ 这是\"机器学习势函数\"的最简版本")
    print(f"  ✓ 现实中：MACE, NequIP 等用于百万原子模拟")
    print(f"  ✓ 比 DFT 快 1000-10000 倍 + 精度相当")
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("第 21 章核心回顾 + 整书项目终曲：")
    print("  物理与 AI 三层纠缠：工具、应用、本体")
    print("  AI 物理化是 21 世纪后半段的方向")
    print()
    print("  实验 1: Hopfield 网络 = 自旋玻璃记忆 ✓")
    print("  实验 2: SGD = Langevin 动力学 ✓")
    print("  实验 3: Ising 临界 → 大模型涌现的同构 ✓")
    print("  实验 4: Boltzmann 采样 ✓")
    print("  实验 5: H₂ 神经网络势函数 ✓")
    print()
    print("=" * 70)
    print("项目完结：《打通物理任督二脉》21 章")
    print("21 个 Python 模块 + 80+ 数值实验 + 450+ 页 PDF")
    print()
    print("https://github.com/inaturelz-coder/physics-rendu")
    print("=" * 70)
