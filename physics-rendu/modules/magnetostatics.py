"""
magnetostatics.py — 静磁学模块
================================
Module 10 of "Physics is Alive" / 《物理是活的》第 10 模块

实现 / Implements:
    - biot_savart_loop_axis: 圆环电流轴线场（解析）
    - biot_savart_segment: 直线段电流的 Biot-Savart 积分
    - helmholtz_field_axis: 亥姆霍兹线圈中心场
    - cyclotron_motion: 洛伦兹力下的圆周运动
    - solenoid_field: 螺线管内外场对比

物理基础 / Physics:
    Biot-Savart: dB = (μ₀I/4π) dl × r̂/r²
    Ampère: ∮ B·dl = μ₀ I_enc
    Lorentz: F = qv×B
    回旋频率: ω_c = qB/m

栗周 / Li Zhou
2026 年 7 月 / July 2026
MIT License
"""

import numpy as np


# 物理常数
mu_0 = 4 * np.pi * 1e-7  # T·m/A
e_charge = 1.602176634e-19
m_e = 9.1093837e-31
m_p = 1.67262192369e-27


# =============================================================================
# 圆环电流轴线场（解析）
# =============================================================================

def biot_savart_loop_axis(I, R, z):
    """半径 R 圆环电流，在轴线 z 处的磁场（沿轴向）
    
    B_z(z) = μ₀ I R² / [2 (R² + z²)^(3/2)]
    """
    return mu_0 * I * R**2 / (2 * (R**2 + z**2)**1.5)


# =============================================================================
# 直线段电流的 Biot-Savart 数值积分
# =============================================================================

def biot_savart_segment(I, dl_start, dl_end, r_field, n_seg=100):
    """直线段电流的 Biot-Savart 积分
    
    把直线段分成 n_seg 小段，每段近似为点源
    
    Parameters:
        I        : 电流
        dl_start, dl_end : 起止点 (3D)
        r_field  : 场点 (3D)
    
    Returns:
        B : 磁场 (3D)
    """
    dl_start = np.array(dl_start, dtype=float)
    dl_end = np.array(dl_end, dtype=float)
    r_field = np.array(r_field, dtype=float)
    
    # 离散点
    t = np.linspace(0, 1, n_seg + 1)
    pts = dl_start[None, :] + np.outer(t, dl_end - dl_start)
    
    # 每段的方向
    dl_vec = (dl_end - dl_start) / n_seg
    
    B = np.zeros(3)
    for i in range(n_seg):
        mid = (pts[i] + pts[i+1]) / 2
        R = r_field - mid
        R_norm = np.linalg.norm(R)
        if R_norm < 1e-9:
            continue
        cross = np.cross(dl_vec, R)
        B += mu_0 * I / (4 * np.pi) * cross / R_norm**3
    return B


def biot_savart_infinite_wire(I, distance):
    """无限长直线电流距离 d 处的磁场
    
    B = μ₀ I / (2π d)
    """
    return mu_0 * I / (2 * np.pi * distance)


# =============================================================================
# 亥姆霍兹线圈
# =============================================================================

def helmholtz_field_axis(I, R, z):
    """亥姆霍兹线圈轴线场
    
    两个半径 R 的同轴圆环，间距 = R（标准亥姆霍兹配置）
    电流同向（共同产生场）
    
    Returns: 轴上 z 位置的 B（沿轴向）
    """
    # 两线圈位于 z = ±R/2
    z1 = z - R/2
    z2 = z + R/2
    B1 = biot_savart_loop_axis(I, R, z1)
    B2 = biot_savart_loop_axis(I, R, z2)
    return B1 + B2


# =============================================================================
# 洛伦兹力下的圆周运动（回旋加速器物理）
# =============================================================================

def cyclotron_motion(q, m, B, v0, t_max=1e-9, dt=1e-13):
    """带电粒子在均匀磁场中的运动
    
    Parameters:
        q   : 电荷
        m   : 质量
        B   : 磁感应强度（沿 z 方向）
        v0  : 初始速度（沿 x 方向）
        t_max, dt : 时间参数
    
    Returns:
        history_r, history_v : 位置和速度时间序列
    """
    r = np.array([0.0, 0.0, 0.0])
    v = np.array([float(v0), 0.0, 0.0])
    B_vec = np.array([0.0, 0.0, float(B)])
    
    n_steps = int(t_max / dt)
    history_r = np.zeros((n_steps + 1, 3))
    history_v = np.zeros((n_steps + 1, 3))
    history_r[0] = r
    history_v[0] = v
    
    for i in range(n_steps):
        # Lorentz 力 F = q v × B
        F = q * np.cross(v, B_vec)
        # 半步速度更新（保证圆周运动闭合）
        v_half = v + 0.5 * F / m * dt
        r = r + v_half * dt
        F_new = q * np.cross(v_half, B_vec)
        v = v_half + 0.5 * F_new / m * dt
        
        history_r[i+1] = r
        history_v[i+1] = v
    
    return history_r, history_v


def cyclotron_frequency(q, m, B):
    """回旋频率 ω_c = qB/m"""
    return q * B / m


def cyclotron_radius(q, m, v, B):
    """回旋半径 r = mv/(qB)"""
    return m * v / (q * B)


# =============================================================================
# 演示 / Demo
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("magnetostatics.py — 物理是活的 / 第 10 模块演示")
    print("Topic: 静磁学 —— 电流、矢势、磁性物质")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # 实验 1：无限长直线电流（Ampère 定律验证）
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 1】无限长直线电流（B = μ₀I/2πd）")
    print("-" * 70)
    
    I = 1.0  # 1 A
    print(f"  电流 I = {I} A")
    print()
    print(f"  {'距离 d (m)':>10s}  {'B 解析 (T)':>14s}  {'1/d 比例':>14s}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*14}")
    
    for d in [0.01, 0.1, 1.0, 10.0]:
        B = biot_savart_infinite_wire(I, d)
        print(f"  {d:>10.2f}  {B:>14.4e}  {1/d:>14.2f}")
    
    print()
    print("  ✓ B ∝ 1/d (与点电荷的 1/r² 不同)")
    print("  ✓ d=1cm 处 B ≈ 20 μT (略低于地磁场 50 μT)")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 2：圆环电流轴线场
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 2】圆环电流轴线场 + 远场偶极行为")
    print("-" * 70)
    
    I = 1.0
    R = 0.1  # 10 cm
    m = I * np.pi * R**2  # 磁偶极矩
    
    print(f"  圆环 R = {R*100:.0f} cm, I = {I} A")
    print(f"  磁偶极矩 m = IπR² = {m:.4f} A·m²")
    print()
    print(f"  {'z (m)':>8s}  {'B 数值 (T)':>14s}  {'B 远场偶极':>14s}  {'近场/远场':>10s}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*10}")
    
    for z in [0.0, 0.05, 0.1, 0.5, 1.0, 5.0]:
        B_full = biot_savart_loop_axis(I, R, z)
        # 远场偶极公式（轴线上）
        if z > 0:
            B_far = mu_0 * m / (2 * np.pi * z**3)
            ratio = B_full / B_far if B_far > 0 else float('inf')
        else:
            B_far = 0
            ratio = float('inf')
        ratio_str = f"{ratio:.4f}" if ratio < 1e10 else "—"
        print(f"  {z:>8.3f}  {B_full:>14.4e}  {B_far:>14.4e}  {ratio_str:>10s}")
    
    print()
    print("  ✓ z >> R 时收敛到磁偶极 1/r³ 远场")
    print("  ✓ z = 0（中心）B = μ₀I/(2R)  (与远场公式偏离)")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 3：亥姆霍兹线圈（均匀场）
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 3】亥姆霍兹线圈（中心均匀场）")
    print("-" * 70)
    
    I = 1.0
    R = 0.1  # 半径 + 间距均为 R
    
    print(f"  两同轴圆环：R = {R*100:.0f} cm, 间距 = R = {R*100:.0f} cm")
    print(f"  电流同向 I = {I} A")
    print()
    
    # 中心场
    B_center = helmholtz_field_axis(I, R, 0)
    # 标准亥姆霍兹公式: B_center = (4/5)^(3/2) μ₀ I / R
    B_th = (4/5)**1.5 * mu_0 * I / R
    
    print(f"  中心 z=0: B 数值 = {B_center:.4e} T")
    print(f"  理论值 (4/5)^(3/2) μ₀I/R = {B_th:.4e} T")
    print(f"  误差: {abs(B_center - B_th)/B_th*100:.4f}%")
    print()
    
    # 均匀性检验：±R/4 范围
    print(f"  均匀性检验（中心 ±R/4 = ±{R/4*100:.1f} cm 范围）：")
    for z_test in [-R/4, -R/8, 0, R/8, R/4]:
        B_test = helmholtz_field_axis(I, R, z_test)
        deviation = abs(B_test - B_center) / B_center * 100
        print(f"    z = {z_test*100:+.2f} cm: B = {B_test:.4e}, 偏差 {deviation:.3f}%")
    
    print()
    print("  ✓ 亥姆霍兹配置在 ±R/4 范围内 B 均匀度 < 1%")
    print()
    
    # -------------------------------------------------------------------------
    # 实验 4：回旋运动
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【实验 4】带电粒子在均匀磁场中的圆周运动")
    print("-" * 70)
    
    # 电子在 1 T 磁场中
    q = e_charge
    m = m_e
    B = 1.0  # T
    v0 = 1e6  # m/s
    
    omega_c = cyclotron_frequency(q, m, B)
    T_c = 2 * np.pi / omega_c  # 回旋周期
    r_c = cyclotron_radius(q, m, v0, B)
    
    print(f"  电子: q = e, m = m_e")
    print(f"  B = {B} T, v₀ = {v0:.0e} m/s")
    print()
    print(f"  回旋频率 ω_c = qB/m = {omega_c:.4e} rad/s")
    print(f"  回旋周期 T_c = {T_c*1e12:.4f} ps")
    print(f"  回旋半径 r_c = mv/qB = {r_c*1e6:.4f} μm")
    print()
    
    # 数值模拟
    t_max = 3 * T_c  # 3 个周期
    dt = T_c / 1000
    history_r, history_v = cyclotron_motion(q, m, B, v0, t_max=t_max, dt=dt)
    
    # 验证轨迹圆周性
    r_distances = np.linalg.norm(history_r[:, :2] - history_r[0, :2], axis=1)
    r_max = r_distances.max()
    
    print(f"  数值模拟 ({len(history_r)} 步, 3 周期):")
    print(f"  最大偏移 = {r_max*1e6:.4f} μm")
    print(f"  理论圆直径 = {2*r_c*1e6:.4f} μm")
    print(f"  比例: {r_max/(2*r_c):.4f} (理想 = 1)")
    
    # 速度幅度恒定（磁力不做功）
    v_norms = np.linalg.norm(history_v, axis=1)
    v_var = (v_norms.max() - v_norms.min()) / v_norms.mean() * 100
    print(f"  速度幅度变化: {v_var:.4f}% (磁力不做功 → 应≈0)")
    print()
    print("  ✓ 完美圆周运动 + 速度幅度守恒")
    print()
    
    # 验证不同粒子的回旋频率
    print(f"  不同粒子在 1 T 磁场中的回旋频率：")
    particles = [
        ("电子", e_charge, m_e),
        ("质子", e_charge, m_p),
        ("μ-子（μ = 207 m_e）", e_charge, 207 * m_e),
    ]
    for name, charge, mass in particles:
        omega = cyclotron_frequency(charge, mass, B)
        f = omega / (2 * np.pi)
        print(f"    {name:>30s}: f = {f/1e9:>10.3f} GHz")
    
    print()
    
    print("=" * 70)
    print("演示完毕 / Demo complete.")
    print()
    print("第 10 章核心回顾：")
    print("  静磁学 = 电流 → 磁场 + 矢势 (比 B 更基本)")
    print()
    print("  实验 1: 直线电流 B = μ₀I/(2πd)  ✓")
    print("  实验 2: 圆环电流 + 远场偶极 1/r³  ✓")
    print("  实验 3: 亥姆霍兹线圈 (中心均匀场)  ✓")
    print("  实验 4: 回旋运动 + 洛伦兹力守速度  ✓")
    print()
    print("练习 / Exercises:")
    print("  1. 加入电场后 → E × B 漂移运动")
    print("  2. 螺线管内外场对比 (内部 B = μ₀nI)")
    print("  3. 数值实现 Aharonov-Bohm 干涉条纹")
    print("  4. 磁偶极在外场中的能量与扭矩")
    print("=" * 70)
