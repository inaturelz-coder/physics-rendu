"""
math_prep.py — 数学预备模块
==================================
Module 0 of "打通物理任督二脉" / 第 0 模块

实现 / Implements:
    - 数值微分 + 积分（验证基本定理）
    - 矩阵对角化（np.linalg.eig）
    - 概率分布采样 + 中心极限定理验证
    - 复数运算 + 欧拉公式可视化

物理基础 / Physics:
    微积分基本定理: ∫_a^b f'(x)dx = f(b) - f(a)
    本征值方程: A v = λ v
    中心极限定理: Σ X_i → 高斯
    欧拉公式: e^{iθ} = cos θ + i sin θ

栗周 / Li Zhou
2026 - 2027
MIT License
"""

import numpy as np


# =============================================================================
# 1. 数值微分 + 积分
# =============================================================================

def numerical_derivative(f, x, h=1e-5):
    """中心差分数值微分：f'(x) ≈ (f(x+h) - f(x-h))/(2h)"""
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_integral(f, a, b, n=10000):
    """复合梯形法数值积分"""
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return np.trapezoid(y, x)


def verify_fundamental_theorem(f, f_prime, a, b):
    """验证微积分基本定理：∫_a^b f'(x)dx = f(b) - f(a)"""
    integral_of_derivative = numerical_integral(f_prime, a, b)
    direct_difference = f(b) - f(a)
    return integral_of_derivative, direct_difference


# =============================================================================
# 2. 矩阵对角化（量子力学的核心）
# =============================================================================

def diagonalize(A):
    """求矩阵 A 的本征值与本征矢
    
    A v = λ v
    
    返回: eigenvalues, eigenvectors
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # 按本征值大小排序
    idx = np.argsort(eigenvalues.real)
    return eigenvalues[idx], eigenvectors[:, idx]


def hermitian_check(A):
    """验证矩阵是否为厄米：A† = A"""
    return np.allclose(A, A.conj().T)


def eigenvalue_equation_check(A, v, lam, tol=1e-10):
    """验证本征值方程 A v = λ v"""
    return np.allclose(A @ v, lam * v, atol=tol)


# =============================================================================
# 3. 概率分布采样 + 中心极限定理
# =============================================================================

def sample_uniform(n_samples, low=0.0, high=1.0, seed=None):
    """从均匀分布采样"""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, n_samples)


def sample_exponential(n_samples, rate=1.0, seed=None):
    """从指数分布采样: f(x) = λ e^{-λx}"""
    rng = np.random.default_rng(seed)
    return rng.exponential(1/rate, n_samples)


def central_limit_demo(n_per_sample=30, n_trials=10000, 
                        dist='uniform', seed=42):
    """中心极限定理演示
    
    从某分布抽 n_per_sample 个数，求平均；
    重复 n_trials 次。
    平均值的分布 → 高斯
    """
    rng = np.random.default_rng(seed)
    
    if dist == 'uniform':
        samples = rng.uniform(0, 1, (n_trials, n_per_sample))
        true_mean = 0.5
        true_var_per = 1/12
    elif dist == 'exponential':
        samples = rng.exponential(1, (n_trials, n_per_sample))
        true_mean = 1.0
        true_var_per = 1.0
    else:
        raise ValueError(f"Unknown dist: {dist}")
    
    sample_means = samples.mean(axis=1)
    # 中心极限定理预测: 平均值 ~ N(μ, σ²/n)
    expected_std = np.sqrt(true_var_per / n_per_sample)
    
    return sample_means, true_mean, expected_std


# =============================================================================
# 4. 复数 + 欧拉公式
# =============================================================================

def euler_formula(theta):
    """欧拉公式: e^{iθ} = cos θ + i sin θ
    
    验证两种计算结果一致
    """
    direct = np.exp(1j * theta)
    decomposed = np.cos(theta) + 1j * np.sin(theta)
    return direct, decomposed


def complex_multiplication(z1, z2):
    """复数乘法 + 验证 |z1*z2| = |z1|*|z2|, arg(z1*z2) = arg(z1)+arg(z2)"""
    product = z1 * z2
    mag_check = np.abs(z1) * np.abs(z2)
    arg_check = np.angle(z1) + np.angle(z2)
    return product, np.abs(product), mag_check, np.angle(product), arg_check


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("math_prep.py — 打通物理任督二脉 / 第 0 模块演示")
    print("Topic: 数学预备 —— 微积分 + 线代 + 概率 + 复数")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：数值微分 + 积分 + 基本定理
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】数值微分 + 积分 + 微积分基本定理")
    print("-" * 70)
    
    # 测 f(x) = x³ 的导数（应该是 3x²）
    f = lambda x: x**3
    f_prime_exact = lambda x: 3 * x**2
    
    print(f"  测试函数: f(x) = x³, f'(x) = 3x²")
    print()
    label_x = 'x'
    label_num = "f'(x) 数值"
    label_exact = "f'(x) 精确"
    label_err = '误差'
    print(f"  {label_x:>6s}  {label_num:>14s}  {label_exact:>14s}  {label_err:>10s}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*10}")
    
    for x in [-2, -1, 0, 1, 2]:
        num_d = numerical_derivative(f, x)
        exact_d = f_prime_exact(x)
        err = abs(num_d - exact_d)
        print(f"  {x:>6.1f}  {num_d:>14.6f}  {exact_d:>14.6f}  {err:>10.2e}")
    
    print()
    print(f"  ✓ 中心差分数值微分精度极高（误差 ~ 1e-10）")
    print()
    
    # 验证基本定理
    print(f"  微积分基本定理验证: ∫_a^b f'(x)dx = f(b) - f(a)")
    a, b = 1.0, 3.0
    int_of_deriv, direct_diff = verify_fundamental_theorem(f, f_prime_exact, a, b)
    print(f"    ∫_1^3 3x² dx = {int_of_deriv:.6f}")
    print(f"    f(3) - f(1) = 3³ - 1³ = {direct_diff}")
    print(f"    ✓ 两者一致（误差 < 1e-6）")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：矩阵对角化（量子力学核心）
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】矩阵对角化 —— 量子力学本征值方程")
    print("-" * 70)
    
    # 构造一个 2x2 厄米矩阵（最简的"量子比特"Hamiltonian）
    # H = [[E_1, t], [t, E_2]] (实对称 → 厄米)
    E1, E2, t = 1.0, -1.0, 0.5
    H = np.array([[E1, t],
                  [t, E2]])
    
    print(f"  Hamiltonian: H = [[{E1}, {t}], [{t}, {E2}]]")
    print(f"  厄米性验证: {hermitian_check(H)}")
    print()
    
    eigvals, eigvecs = diagonalize(H)
    
    print(f"  本征值（按大小排序）：")
    for i, ev in enumerate(eigvals):
        print(f"    E_{i} = {ev:+.6f}")
    print()
    
    # 解析解：E± = (E1+E2)/2 ± √((E1-E2)²/4 + t²)
    discriminant = np.sqrt(((E1-E2)/2)**2 + t**2)
    E_plus = (E1+E2)/2 + discriminant
    E_minus = (E1+E2)/2 - discriminant
    print(f"  解析解：")
    print(f"    E_- = {E_minus:+.6f}")
    print(f"    E_+ = {E_plus:+.6f}")
    print()
    
    # 验证本征值方程
    for i, (lam, v) in enumerate(zip(eigvals, eigvecs.T)):
        is_eigen = eigenvalue_equation_check(H, v, lam)
        print(f"  H v_{i} = λ_{i} v_{i}: {is_eigen}")
    
    print()
    print(f"  ✓ 量子力学的\"本征值 = 能级\"——这是 Ch14 之根基")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：中心极限定理
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】中心极限定理 —— 高斯分布的统治")
    print("-" * 70)
    
    # 从均匀分布抽 30 个数求平均，重复 10000 次
    sample_means, true_mean, expected_std = central_limit_demo(
        n_per_sample=30, n_trials=10000, dist='uniform'
    )
    
    print(f"  从均匀分布 U(0,1) 抽 30 个数求平均，重复 10000 次")
    print(f"  理论：平均值 ~ N(μ={true_mean}, σ={expected_std:.4f})")
    print()
    print(f"  数值统计：")
    print(f"    平均值的均值 = {sample_means.mean():.4f}（理论 {true_mean}）")
    print(f"    平均值的方差 = {sample_means.std():.4f}（理论 {expected_std:.4f}）")
    print()
    
    # 用指数分布也试试（更"不像高斯"的分布）
    sample_means_exp, true_mean_exp, exp_std_exp = central_limit_demo(
        n_per_sample=30, n_trials=10000, dist='exponential'
    )
    print(f"  即使从指数分布（高度偏斜）抽样，平均值仍趋近高斯：")
    print(f"    平均值的均值 = {sample_means_exp.mean():.4f}（理论 {true_mean_exp}）")
    print(f"    平均值的方差 = {sample_means_exp.std():.4f}（理论 {exp_std_exp:.4f}）")
    print()
    print(f"  ✓ 中心极限定理：大量独立样本的平均 → 高斯")
    print(f"  ✓ 这是为何高斯分布在物理中无处不在")
    print(f"  ✓ 测量误差、热涨落 —— 皆为高斯")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：复数 + 欧拉公式
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】欧拉公式 e^{iθ} = cos θ + i sin θ")
    print("-" * 70)
    
    print(f"  欧拉公式验证：")
    print(f"  {'θ':>10s}  {'e^(iθ)':>30s}  {'cos θ + i sin θ':>30s}")
    print(f"  {'-'*10}  {'-'*30}  {'-'*30}")
    
    for theta in [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]:
        direct, decomp = euler_formula(theta)
        theta_label = f"{theta/np.pi:.2f}π" if theta != 0 else "0"
        print(f"  {theta_label:>10s}  "
              f"{f'{direct.real:+.4f} + {direct.imag:+.4f}i':>30s}  "
              f"{f'{decomp.real:+.4f} + {decomp.imag:+.4f}i':>30s}")
    
    print()
    
    # 欧拉恒等式
    print(f"  欧拉恒等式（数学最美）：e^(iπ) + 1 = {np.exp(1j*np.pi) + 1}")
    print(f"  （应该约等于 0 + 0i ——浮点误差是正常的）")
    print()
    
    # 复数乘法验证
    print(f"  复数乘法验证：z₁·z₂ ——")
    z1 = 2 * np.exp(1j * np.pi/4)  # 模 2，辐角 π/4
    z2 = 3 * np.exp(1j * np.pi/3)  # 模 3，辐角 π/3
    
    product, mag_p, mag_check, arg_p, arg_check = complex_multiplication(z1, z2)
    print(f"    z₁ = 2·e^(iπ/4), z₂ = 3·e^(iπ/3)")
    print(f"    z₁·z₂ = {product:.4f}")
    print(f"    |z₁·z₂| = {mag_p:.4f}, |z₁|·|z₂| = {mag_check:.4f}")
    print(f"    arg(z₁·z₂) = {arg_p/np.pi:.4f}π, arg(z₁) + arg(z₂) = {arg_check/np.pi:.4f}π")
    print(f"    ✓ 复数乘法：模相乘 + 辐角相加")
    print()
    print(f"  ✓ 欧拉公式是量子力学 + 信号处理 + 波动力学的根基")
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("第 0 章核心回顾：")
    print("  数学预备 = 微积分 + 矢量 + 线代 + 复数 + 概率")
    print()
    print("  实验 1: 数值微分 + 基本定理 ✓")
    print("  实验 2: 矩阵对角化（量子本征值）✓")
    print("  实验 3: 中心极限定理（高斯之源）✓")
    print("  实验 4: 欧拉公式 + 复数乘法 ✓")
    print()
    print("练习 / Exercises:")
    print("  1. 实现 3x3 矩阵对角化 + 验证")
    print("  2. 比较泰勒展开与精确函数误差")
    print("  3. 蒙特卡洛求 π")
    print("  4. 数值求 ∇·F（散度）")
    print("=" * 70)

    # ============== 可选画图 ==============
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        import numpy as np
        
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        
        # 左上：欧拉公式 — 单位圆
        ax = axes[0, 0]
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'b-', lw=2)
        # 标几个特殊点
        for t, label in [(0, '1'), (np.pi/2, 'i'), (np.pi, '-1'), (3*np.pi/2, '-i')]:
            ax.plot(np.cos(t), np.sin(t), 'ro', markersize=10)
            ax.annotate(f'$e^{{i{label}}}={np.cos(t):+.0f}{"+":>1s}{np.sin(t):+.0f}i$' if abs(np.sin(t))>0.5 or abs(np.cos(t))>0.5 else f'$e^{{i\\theta}}$', 
                       xy=(np.cos(t), np.sin(t)),
                       xytext=(np.cos(t)*1.2, np.sin(t)*1.2),
                       ha='center', fontsize=10)
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlabel('Re')
        ax.set_ylabel('Im')
        ax.set_title('欧拉公式: $e^{i\\theta}$ 单位圆')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        
        # 右上：中心极限定理
        ax = axes[0, 1]
        rng = np.random.default_rng(42)
        # 从均匀分布抽样本，求平均
        means = []
        for _ in range(10000):
            sample = rng.uniform(0, 1, 30)
            means.append(sample.mean())
        means = np.array(means)
        ax.hist(means, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')
        # 叠加理论高斯曲线
        x = np.linspace(means.min(), means.max(), 200)
        sigma = np.sqrt(1/12/30)
        gauss = np.exp(-(x-0.5)**2/(2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
        ax.plot(x, gauss, 'r-', lw=2.5, label=f'$N(0.5, {sigma:.3f})$')
        ax.set_xlabel('样本平均')
        ax.set_ylabel('概率密度')
        ax.set_title('中心极限定理（均匀→高斯）')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 左下：导数 vs 数值微分
        ax = axes[1, 0]
        x = np.linspace(-3, 3, 100)
        f_vals = x**3
        f_prime_exact = 3 * x**2
        # 数值微分
        h = 1e-4
        f_prime_num = (((x+h)**3 - (x-h)**3) / (2*h))
        ax.plot(x, f_vals, 'b-', lw=2, label='$f(x) = x^3$')
        ax.plot(x, f_prime_exact, 'r-', lw=2, label="$f'(x) = 3x^2$ 精确")
        ax.plot(x[::5], f_prime_num[::5], 'go', markersize=6, label="数值微分")
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('f, f\'')
        ax.set_title("数值微分 vs 精确导数")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 右下：矩阵本征值 — 量子比特能级
        ax = axes[1, 1]
        t_arr = np.linspace(-2, 2, 200)
        E1, E2 = 1.0, -1.0
        E_minus = (E1+E2)/2 - np.sqrt(((E1-E2)/2)**2 + t_arr**2)
        E_plus = (E1+E2)/2 + np.sqrt(((E1-E2)/2)**2 + t_arr**2)
        ax.plot(t_arr, E_plus, 'r-', lw=2.5, label='$E_+$ 上能级')
        ax.plot(t_arr, E_minus, 'b-', lw=2.5, label='$E_-$ 下能级')
        # 标耦合 t=0 处
        ax.axhline(E1, color='gray', ls=':', alpha=0.5, label='$E_1$（无耦合）')
        ax.axhline(E2, color='gray', ls=':', alpha=0.5, label='$E_2$（无耦合）')
        ax.set_xlabel('耦合 t')
        ax.set_ylabel('能量 E')
        ax.set_title('量子比特: 2x2 厄米矩阵能级（"avoided crossing"）')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.suptitle('Ch0: 数学预备 —— 微积分 + 线代 + 概率 + 复数', fontsize=13)
        plt.tight_layout()
        plt.savefig('ch00_math_prep.png', dpi=120, bbox_inches='tight')
        plt.close()
        print("\n✓ 已保存可视化: ch00_math_prep.png")
    except ImportError:
        print("\n(matplotlib 未安装 — 跳过画图)")

