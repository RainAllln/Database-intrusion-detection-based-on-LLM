import torch
import sys
import os

def check_cuda():
    print("="*30)
    print("CUDA 环境检查脚本")
    print("="*30)
    
    # 获取当前 Python 解释器的绝对路径，方便用户直接复制命令
    python_exec = sys.executable
    
    print(f"Step 1: 检查软件版本")
    print(f" - Python path: {python_exec}")
    print(f" - PyTorch version: {torch.__version__}")
    
    print(f"\nStep 2: 检查 CUDA 可用性")
    if torch.cuda.is_available():
        print("\n✅ 恭喜！CUDA (GPU) 可用。")
        try:
            device_count = torch.cuda.device_count()
            print(f" - 检测到的 GPU 数量: {device_count}")
            
            for i in range(device_count):
                print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")
                
            print(f" - 当前 CUDA 版本: {torch.version.cuda}")
            
            # 测试简单的 Tensor 运算
            print("\nStep 3: 简单的 Tensor 运算测试")
            x = torch.tensor([1.0, 2.0]).cuda()
            print(f" - Tensor 成功移至 GPU: {x.device}")
            print(" - 测试通过！")
            
        except Exception as e:
            print(f"\n⚠️ 虽然检测到 CUDA，但在获取信息时出错: {e}")
    else:
        print("\n❌ CUDA 不可用。您的 PyTorch 版本是 CPU 专用版。")
        
        print("\n⬇️⬇️⬇️ 请在 CMD 或 PowerShell 中依次运行以下两条命令来修复 ⬇️⬇️⬇️")
        print("-" * 60)
        # 1. 卸载旧版本
        print(f'"{python_exec}" -m pip uninstall -y torch torchvision torchaudio')
        # 2. 安装 CUDA 12.1 版本 (兼容大多数现代 N 卡)
        print(f'"{python_exec}" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')
        print("-" * 60)
        print("注意: 安装完成后，请再次运行此脚本以验证 GPU 是否可用。")

if __name__ == "__main__":
    check_cuda()
