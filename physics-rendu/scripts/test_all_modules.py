#!/usr/bin/env python3
"""
测试所有 20 个模块能否正常运行 demo
"""

import subprocess
import sys
from pathlib import Path


MODULES = [
    'particle.py', 'oscillator.py', 'rotation.py', 'wave.py', 'chaos.py',
    'thermo.py', 'stat.py', 'quantum_stat.py',
    'electrostatics.py', 'magnetostatics.py', 'phases.py',
    'em_wave.py', 'matter_em.py',
    'quantum.py', 'quantum_core.py', 'bell_chsh.py', 'qcomp.py',
    'solid_state.py', 'advanced_cm.py', 'optics.py',
]


def test_module(module_path):
    """运行单个模块的 demo"""
    try:
        result = subprocess.run(
            [sys.executable, module_path],
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.returncode == 0, result.stderr[:200] if result.stderr else ''
    except subprocess.TimeoutExpired:
        return False, 'Timeout'
    except Exception as e:
        return False, str(e)


def main():
    modules_dir = Path(__file__).parent.parent / 'modules'
    print(f"测试模块目录: {modules_dir}\n")
    
    success = []
    failed = []
    
    for module_name in MODULES:
        module_path = modules_dir / module_name
        if not module_path.exists():
            print(f"⚠ {module_name}: 文件不存在")
            failed.append((module_name, 'missing'))
            continue
        
        ok, err = test_module(str(module_path))
        if ok:
            print(f"✓ {module_name}")
            success.append(module_name)
        else:
            print(f"✗ {module_name}: {err}")
            failed.append((module_name, err))
    
    print(f"\n{'='*50}")
    print(f"总结: 成功 {len(success)}/{len(MODULES)}")
    if failed:
        print(f"失败:")
        for name, err in failed:
            print(f"  - {name}")


if __name__ == '__main__':
    main()
