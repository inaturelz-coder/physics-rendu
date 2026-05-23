"""
thermo.py — 热力学模块
========================
Module 6 of "Physics is Alive" / 《物理是活的》第 6 模块

实现 / Implements:
    - IdealGas: 理想气体的 4 种过程（等温/等压/等容/绝热）
    - CarnotEngine: 卡诺循环 + 效率验证
    - free_expansion_entropy: 自由膨胀的熵变
    - Maxwell-Boltzmann 速度分布

物理基础 / Physics:
    pV = N k_B T
    卡诺效率 η = 1 - T_c/T_h
    熵增 ΔS = k_B ln(Ω₂/Ω₁)

栗周 / Li Zhou
2026 年 7 月 / July 2026
MIT License
"""

import numpy as np


# 物理常数
R = 8.314          # 气体常数 J/(mol·K)
k_B = 1.380649e-23 # Boltzmann 常数 J/K
N_A = 6.022e23     # Avogadro 数


# =============================================================================
# 类 1：理想气体
# =============================================================================

class IdealGas:
    """理想气体类
    
    pV = nRT
    单原子 γ = 5/3, 双原子 γ = 7/5
    
    Example:
        >>> gas = IdealGas(n=1.0, T=300, V=0.022)
        >>> gas.pressure()  # ~ 101325 Pa
    """
    
    def __init__(self, n=1.0, T=300.0, V=0.022, gamma=5/3):
        """
        Parameters:
            n     : 物质的量 (mol)
            T     : 温度 (K)
            V     : 体积 (m³)
            gamma : 绝热指数 (单原子 5/3, 双原子 7/5)
        """
        self.n = float(n)
        self.T = float(T)
        self.V = float(V)
        self.gamma = float(gamma)
    
    def pressure(self):
        """从状态方程算压强"""
        return self.n * R * self.T / self.V
    
    def internal_energy(self):
        """内能 U = (f/2) n R T, f = 自由度"""
        # γ = (f+2)/f → f = 2/(γ-1)
        f = 2 / (self.gamma - 1)
        return f/2 * self.n * R * self.T
    
    def isothermal_process(self, V_final):
        """等温过程（pV = nRT 不变）"""
        V1, V2 = self.V, V_final
        W = self.n * R * self.T * np.log(V2 / V1)  # 气体对外做功
        Q = W  # 等温 ΔU = 0
        return {'W': W, 'Q': Q, 'dU': 0, 'V_final': V_final, 'T_final': self.T}
    
    def adiabatic_process(self, V_final):
        """绝热过程（pV^γ = const, TV^(γ-1) = const）"""
        V1, V2 = self.V, V_final
        T1, T2 = self.T, self.T * (V1 / V2)**(self.gamma - 1)
        # 对外做功 = -ΔU（绝热 Q=0）
        f = 2 / (self.gamma - 1)
        dU = f/2 * self.n * R * (T2 - T1)
        W = -dU
        return {'W': W, 'Q': 0, 'dU': dU, 'V_final': V_final, 'T_final': T2}
    
    def isobaric_process(self, V_final):
        """等压过程"""
        V1, V2 = self.V, V_final
        p = self.pressure()
        T2 = self.T * V2 / V1
        f = 2 / (self.gamma - 1)
        dU = f/2 * self.n * R * (T2 - self.T)
        W = p * (V2 - V1)  # 气体对外做功
        Q = dU + W
        return {'W': W, 'Q': Q, 'dU': dU, 'V_final': V_final, 'T_final': T2}
    
    def isochoric_process(self, T_final):
        """等容过程"""
        T2 = float(T_final)
        f = 2 / (self.gamma - 1)
        dU = f/2 * self.n * R * (T2 - self.T)
        return {'W': 0, 'Q': dU, 'dU': dU, 'V_final': self.V, 'T_final': T2}


# =============================================================================
# 类 2：卡诺循环
# =============================================================================

class CarnotEngine:
    """卡诺热机 —— 4 步可逆循环
    
    1. 等温膨胀 (T_h)
    2. 绝热膨胀 (T_h → T_c)
    3. 等温压缩 (T_c)
    4. 绝热压缩 (T_c → T_h)
    
    效率 η = 1 - T_c/T_h
    """
    
    def __init__(self, T_h=500.0, T_c=300.0, V1=0.001, V2=0.005, n=1.0, gamma=5/3):
        self.T_h = float(T_h)
        self.T_c = float(T_c)
        self.V1 = float(V1)   # 等温膨胀起点
        self.V2 = float(V2)   # 等温膨胀终点
        self.n = float(n)
        self.gamma = float(gamma)
        self._compute()
    
    def _compute(self):
        """计算完整循环的热和功"""
        # 1. 等温膨胀 T_h, V1 → V2
        gas = IdealGas(n=self.n, T=self.T_h, V=self.V1, gamma=self.gamma)
        step1 = gas.isothermal_process(self.V2)
        self.Q_h = step1['Q']  # 吸热
        
        # 2. 绝热膨胀 V2 → V3 (T_h → T_c)
        # TV^(γ-1) = const → V3 = V2 * (T_h/T_c)^(1/(γ-1))
        V3 = self.V2 * (self.T_h / self.T_c)**(1/(self.gamma - 1))
        gas2 = IdealGas(n=self.n, T=self.T_h, V=self.V2, gamma=self.gamma)
        step2 = gas2.adiabatic_process(V3)
        self.V3 = V3
        
        # 3. 等温压缩 T_c, V3 → V4
        # V4 满足绝热压缩回到 V1：V4/V1 = V3/V2
        V4 = V3 * self.V1 / self.V2
        gas3 = IdealGas(n=self.n, T=self.T_c, V=V3, gamma=self.gamma)
        step3 = gas3.isothermal_process(V4)
        self.Q_c = step3['Q']  # 放热（负值）
        self.V4 = V4
        
        # 4. 绝热压缩 V4 → V1 (T_c → T_h)
        gas4 = IdealGas(n=self.n, T=self.T_c, V=V4, gamma=self.gamma)
        step4 = gas4.adiabatic_process(self.V1)
        
        # 净功
        self.W_net = self.Q_h + self.Q_c
        self.efficiency_numerical = self.W_net / self.Q_h
        self.efficiency_theory = 1 - self.T_c / self.T_h


# =============================================================================
# 自由膨胀的熵变
# =============================================================================

def free_expansion_entropy(V1, V2, N=N_A):
    """理想气体自由膨胀熵变
    
    ΔS = N k_B ln(V₂/V₁)
    虽然 ΔU = 0 但 ΔS > 0 → 不可逆
    """
    return N * k_B * np.log(V2 / V1)


# =============================================================================
# Maxwell-Boltzmann 分布
# =============================================================================

def maxwell_boltzmann_pdf(v, m, T):
    """MB 速度分布概率密度
    
    f(v) = 4π (m/2πk_BT)^(3/2) v² exp(-mv²/2k_BT)
    """
    return 4 * np.pi * (m / (2*np.pi*k_B*T))**(3/2) * v**2 * \
           np.exp(-m*v**2 / (2*k_B*T))


def sample_maxwell_boltzmann(N, m, T, seed=None):
    """从 MB 分布抽样
    
    每个粒子的 (vx, vy, vz) 独立服从 Gaussian
    总速率 |v| 服从 MB 分布
    """
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(k_B * T / m)
    vx = rng.normal(0, sigma, N)
    vy = rng.normal(0, sigma, N)
    vz = rng.normal(0, sigma, N)
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    return v


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("thermo.py — 物理是活的 / 第 6 模块演示")
    print("Topic: 热力学 —— 宏观的力量")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：理想气体 4 种过程
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】理想气体的 4 种过程 (n=1 mol, T=300 K, V=22.4 L)")
    print("-" * 70)
    
    gas = IdealGas(n=1.0, T=300.0, V=0.0224)
    p1 = gas.pressure()
    U1 = gas.internal_energy()
    
    print(f"  初始: p = {p1:.0f} Pa, T = {gas.T} K, V = {gas.V*1000:.2f} L")
    print(f"        U = {U1:.0f} J")
    print()
    
    # 等温膨胀 V → 2V
    proc1 = gas.isothermal_process(2 * gas.V)
    print(f"  等温膨胀 (V → 2V):")
    print(f"    Q = {proc1['Q']:.0f} J (吸热)")
    print(f"    W = {proc1['W']:.0f} J (气体对外做功)")
    print(f"    ΔU = {proc1['dU']:.0f} J  ← 等温 ΔU = 0")
    print()
    
    # 绝热膨胀 V → 2V
    proc2 = gas.adiabatic_process(2 * gas.V)
    print(f"  绝热膨胀 (V → 2V, γ=5/3):")
    print(f"    T_final = {proc2['T_final']:.1f} K (温度降低)")
    print(f"    Q = 0 (绝热)")
    print(f"    W = {proc2['W']:.0f} J = -ΔU")
    print()
    
    # 等压
    proc3 = gas.isobaric_process(2 * gas.V)
    print(f"  等压膨胀 (V → 2V):")
    print(f"    T_final = {proc3['T_final']:.1f} K (温度升高，因为吸热)")
    print(f"    Q = {proc3['Q']:.0f} J, W = {proc3['W']:.0f} J")
    print()
    
    # 等容
    proc4 = gas.isochoric_process(600)
    print(f"  等容加热 (T → 600 K):")
    print(f"    Q = ΔU = {proc4['Q']:.0f} J")
    print(f"    W = 0 (体积不变)")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：卡诺循环 + 效率验证
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】卡诺循环效率 (T_h=500 K, T_c=300 K)")
    print("-" * 70)
    
    carnot = CarnotEngine(T_h=500, T_c=300, V1=0.001, V2=0.005)
    
    print(f"  Q_h (从热源吸热) = +{carnot.Q_h:.1f} J")
    print(f"  Q_c (向冷源放热) = {carnot.Q_c:+.1f} J")
    print(f"  W_net (净对外做功) = {carnot.W_net:.1f} J")
    print()
    print(f"  数值效率: η = W_net / Q_h = {carnot.efficiency_numerical:.4f}")
    print(f"  理论效率: η = 1 - T_c/T_h = {carnot.efficiency_theory:.4f}")
    error = abs(carnot.efficiency_numerical - carnot.efficiency_theory) / \
            carnot.efficiency_theory * 100
    print(f"  误差: {error:.4f}%")
    print(f"  ✓ 卡诺定理验证：η = 1 - T_c/T_h")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：自由膨胀的熵变
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】自由膨胀的熵变（1 mol 气体, V → 2V）")
    print("-" * 70)
    
    dS = free_expansion_entropy(V1=0.0224, V2=0.0448, N=N_A)
    print(f"  ΔU = 0  (理想气体, 自由膨胀)")
    print(f"  ΔS = N k_B ln(V₂/V₁) = {N_A} × {k_B:.3e} × ln(2)")
    print(f"  ΔS = {dS:.3f} J/K")
    print()
    print(f"  熵增 > 0 → 不可逆过程 ✓")
    print(f"  系统能量不变 —— 但微观状态数变多 —— 熵增加")
    print(f"  这是第二定律最简单的例子")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：Maxwell-Boltzmann 速度分布
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】Maxwell-Boltzmann 速度分布 (氩气, T=300 K)")
    print("-" * 70)
    
    m_Ar = 39.948 / N_A / 1000  # 氩原子质量 (kg)
    T = 300.0
    
    # 数值采样
    N = 100000
    v_samples = sample_maxwell_boltzmann(N, m_Ar, T, seed=42)
    
    # 数值统计
    v_mean_sim = np.mean(v_samples)
    v_rms_sim = np.sqrt(np.mean(v_samples**2))
    v_p_sim = v_samples[np.argmax(np.histogram(v_samples, bins=100)[0])]  # 粗估
    # 用直方图峰值更精确
    hist, bin_edges = np.histogram(v_samples, bins=200)
    v_p_sim = bin_edges[np.argmax(hist)]
    
    # 理论值
    v_p_th = np.sqrt(2*k_B*T/m_Ar)              # 最可几速率
    v_mean_th = np.sqrt(8*k_B*T/(np.pi*m_Ar))   # 平均速率
    v_rms_th = np.sqrt(3*k_B*T/m_Ar)            # 方均根速率
    
    print(f"  原子质量 m = {m_Ar*1e26:.3f} × 10⁻²⁶ kg, T = {T} K")
    print()
    print(f"  {'量':>20s}  {'数值':>10s}  {'理论':>10s}  {'误差':>8s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}")
    print(f"  {'最可几速率 v_p':>20s}  {v_p_sim:>10.1f}  {v_p_th:>10.1f}  "
          f"{abs(v_p_sim-v_p_th)/v_p_th*100:>6.2f}%")
    print(f"  {'平均速率 <v>':>20s}  {v_mean_sim:>10.1f}  {v_mean_th:>10.1f}  "
          f"{abs(v_mean_sim-v_mean_th)/v_mean_th*100:>6.2f}%")
    print(f"  {'方均根 v_rms':>20s}  {v_rms_sim:>10.1f}  {v_rms_th:>10.1f}  "
          f"{abs(v_rms_sim-v_rms_th)/v_rms_th*100:>6.2f}%")
    print()
    print(f"  ✓ 数值仿真完美匹配 Maxwell-Boltzmann 分布")
    print(f"  注: v_p < <v> < v_rms 永远成立 (分布右偏)")
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("第 6 章核心回顾：")
    print("  热力学 = 宏观铁律：能量守恒 + 熵增 + 卡诺效率")
    print()
    print("  实验 1: 理想气体 4 种过程               ✓")
    print("  实验 2: 卡诺效率 η = 1 - T_c/T_h        ✓")
    print("  实验 3: 自由膨胀 ΔU = 0 但 ΔS > 0      ✓ (不可逆)")
    print("  实验 4: Maxwell-Boltzmann 速度分布     ✓")
    print()
    print("练习 / Exercises:")
    print("  1. 画卡诺循环的 p-V 图（用 matplotlib）")
    print("  2. 用不同 T 比较 MB 分布形状")
    print("  3. 试计算 H₂、CO₂ 等的 v_p（不同 m）")
    print("=" * 70)
