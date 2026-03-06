import subprocess
import sys

def install_packages():
    # 定义需要安装的包列表
    packages = [
        "fastapi",
        "uvicorn",
        "requests",
        "python-multipart"
    ]

    print("--- 正在开始一键配置 ComfyUI Agent 环境 ---")
    
    for package in packages:
        try:
            print(f"正在检查/下载包: {package}...")
            # 使用当前 python 解释器对应的 pip
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} 安装成功。")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} 安装失败，请检查网络或权限。错误信息: {e}")
            continue

    print("\n--- 配置完成！你可以运行 'python comfy_agent.py' 启动服务了 ---")

if __name__ == "__main__":
    install_packages()

#在终端或命令行执行  python setup_env.py   即可配置所需依赖项