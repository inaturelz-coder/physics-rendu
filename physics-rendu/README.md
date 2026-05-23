# 打通物理任督二脉 / Physics Rendu

> **高中到研究生的物理贯通课**
> *A Fun Lecture Note from High School to Graduate Level*

作者 / Author: **Li Zhou**
邮箱 / Email: lizhou_alfred2011@hotmail.com
第一版 / First Edition: 2026 - 2027
许可证 / License: MIT

---

## 📖 关于本项目 / About

这是一本试图**贯通中国学生从高中到研究生**整个物理学习路径的中文讲义。

**特色 / Features**:

- 📚 **21 章覆盖整个物理图景**: 力学 → 热统 → 电磁 → 量子 → 凝聚态 → 光学 → AI
- 🌐 **双语写作**: 中文为主，关键概念配英文
- 📐 **每章 7 件套结构**: 本质 + 教材 + 直觉 + 数学 + 例子 + 现代 + 代码
- 💻 **20 个 Python 模块**: MIT 开源，每章 5+ 数值实验
- 🎯 **跨段位**: 高中生能看进，研究生有收获

---

## 📂 仓库结构 / Repository Structure

```
physics-rendu/
├── README.md              ← 本文件
├── LICENSE                ← MIT 许可证
├── requirements.txt       ← Python 依赖
├── modules/               ← 20 个配套 Python 模块
│   ├── particle.py        Ch1 动量
│   ├── oscillator.py      Ch2 能量
│   ├── rotation.py        Ch3 角动量
│   ├── wave.py            Ch4 振动与波
│   ├── chaos.py           Ch5 混沌
│   ├── thermo.py          Ch6 热力学
│   ├── stat.py            Ch7 统计力学
│   ├── quantum_stat.py    Ch8 量子统计
│   ├── electrostatics.py  Ch9 静电学
│   ├── magnetostatics.py  Ch10 静磁学
│   ├── phases.py          Ch11 相变
│   ├── em_wave.py         Ch12 Maxwell
│   ├── matter_em.py       Ch13 物质中电磁
│   ├── quantum.py         Ch14 量子力学初步
│   ├── quantum_core.py    Ch15 量子力学核心
│   ├── bell_chsh.py       Ch16 Bell 不等式
│   ├── qcomp.py           Ch17 量子计算
│   ├── solid_state.py     Ch18 凝聚态入门
│   ├── advanced_cm.py     Ch19 凝聚态前沿
│   └── optics.py          Ch20 光学专题
├── chapters/              ← 各章独立 PDF
│   ├── Ch1_Momentum.pdf
│   ├── Ch2_Energy.pdf
│   └── ...
└── docs/                  ← 完整书 PDF
    ├── book_v2_public.pdf  ← 公开版（删代码，链 GitHub）
    └── book_v1_full.pdf    ← 完整版（含代码段）
```

---

## 🚀 快速开始 / Quick Start

### 安装依赖 / Install Dependencies

```bash
pip install -r requirements.txt
```

只需要：

- Python ≥ 3.10
- numpy
- scipy

可选（用于绘图，但 demo 默认只打印数值）：

- matplotlib

### 运行第一个 demo

```bash
cd modules
python3 quantum.py
```

预期输出：1D Schrödinger 方程数值解，无限井、谐振子、隧穿、波包等 5 个实验。

### 运行所有 demo

```bash
cd modules
for f in *.py; do
    echo "=== $f ==="
    python3 "$f"
done
```

---

## 📚 全书结构 / Book Structure

### 第一部分：力学篇

| 章 | 主题 | 模块 | 关键实验 |
|---:|---|---|---|
| 1 | 动量 | particle.py | 守恒律 + 弹性碰撞 |
| 2 | 能量 | oscillator.py | 简谐 + 阻尼 + 共振 |
| 3 | 角动量 | rotation.py | L 守恒误差 < 1e-12 |
| 4 | 振动与波 | wave.py | 共振 < 0.01%, 波速 3.16 |
| 5 | 混沌 | chaos.py | Feigenbaum 常数 4.668 |

### 第二部分：热与统计篇

| 章 | 主题 | 模块 | 关键实验 |
|---:|---|---|---|
| 6 | 热力学 | thermo.py | Carnot 效率精确 |
| 7 | 统计力学 | stat.py | 2D Ising MC, χ 峰 78 |
| 8 | 量子统计 | quantum_stat.py | Cu ε_F=7.04eV, Rb-87 BEC 0.398μK |

### 第三部分：电磁篇

| 章 | 主题 | 模块 | 关键实验 |
|---:|---|---|---|
| 9 | 静电学 | electrostatics.py | Jacobi 收敛 3280 次 |
| 10 | 静磁学 | magnetostatics.py | 亥姆霍兹中心场 |
| 11 | 相变与对称破缺 | phases.py | 2D Ising β≈0.0851 vs Onsager 0.125 |
| 12 | Maxwell | em_wave.py | 5 实验: c, Poynting, Rayleigh, Stokes, 相对论 |
| 13 | 物质中电磁 | matter_em.py | Drude + Lorentz + Fresnel + London + KK |

### 第四部分：量子篇

| 章 | 主题 | 模块 | 关键实验 |
|---:|---|---|---|
| 14 | 量子力学初步 | quantum.py | 1D TISE 数值解 |
| 15 | 量子力学核心 | quantum_core.py | H 原子 -13.6 eV, He 变分 -77.5 eV |
| 16 | 解释与延展 | bell_chsh.py | CHSH S=2.828=2√2 |
| 17 | 量子信息 | qcomp.py | Bell + 隐形传态(F=1.0) + Grover(N=4, 100%) + QFT |

### 第五部分：凝聚态篇

| 章 | 主题 | 模块 | 关键实验 |
|---:|---|---|---|
| 18 | 凝聚态入门 | solid_state.py | 1D紧束缚 + 石墨烯 Dirac (v_F=8.7e5 m/s) |
| 19 | 凝聚态前沿 | advanced_cm.py | SSH拓扑 + Haldane + altermag (净M=0局部分裂) + BCS |

### 第六部分：光学专题

| 章 | 主题 | 模块 | 关键实验 |
|---:|---|---|---|
| 20 | 光学专题 | optics.py | 双缝 + 衍射 + 激光阈值 + SHG + 光纤色散 |

### 第七部分：未来（2027 年初）

| 章 | 主题 |
|---:|---|
| 21 | 物理与 AI（待发布） |

---

## 📖 如何阅读 / How to Read

### 第一遍（速读）
读每章 §1 一句话本质 + §5 Aha 例子
→ 看完 20 章 = **物理学全景图**

### 第二遍（细读）
补每章 §3 + §4 数学推导
→ 看完 = **大学物理 + 半个研究生物理**

### 第三遍（实战）
跑每章配套 Python 模块，改参数 + 看结果
→ 你才真正"懂"

### 推荐顺序

- **顺序读最稳**（章节有依赖关系）
- 跳读也可，每章相对独立
- **强烈建议先掌握 Ch6 (热) + Ch7 (统计) + Ch11 (相变)** —— 这是凝聚态语言入口

---

## 🤝 反馈与贡献 / Feedback

欢迎以下方式反馈：

- **GitHub Issues**: 报错、建议、问题
- **Pull Requests**: 修订、新实验、章节改进
- **邮箱**: lizhou_alfred2011@hotmail.com

---

## 📅 路线图 / Roadmap

- [x] Ch 1-20 主体内容（2026）
- [x] 配套 Python 模块（2026）
- [x] 第一版书 PDF（v1 含代码 + v2 公开版）
- [ ] Ch 21 物理与 AI（2027 年初）
- [ ] 个人网站 / 配套博客
- [ ] 第二版修订（根据读者反馈）

---

## 📝 许可证 / License

本仓库所有内容（书 PDF + 代码 + 文档）均采用 **MIT 许可证**开源：

- ✅ 自由阅读、复制、分发、修改
- ✅ 商用允许（但请保留原作者署名）
- ❌ 不承诺任何使用结果的保证

详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢 / Acknowledgments

- 感谢一位 AI 助手（Anthropic Claude）在本书写作过程中提供的对话陪伴、数学推导校对、代码调试帮助

文责自负。

---

> **物理是活的——但它需要有人来贯通**
>
>  · 2026 - 2027
> 德国 达姆施塔特 / 中国 桂林
