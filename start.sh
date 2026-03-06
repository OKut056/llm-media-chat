#!/bin/bash

echo "========================================"
echo "      正在启动 AI Agent 服务..."
echo "========================================"

# 0. 检查并启动 Ollama
echo "[0/3] 正在检测 Ollama 服务状态..."
# 使用 curl 探测默认端口 11434 是否有响应
if curl -s http://localhost:11434/ > /dev/null; then
    echo "[*] Ollama 服务已在运行中。"
else
    echo "[*] 检测到 Ollama 未启动，正在尝试在后台启动..."
    # 以后台静默方式启动并忽略输出
    ollama serve >/dev/null 2>&1 &
    OLLAMA_PID=$!
    # 等待3秒让其初始化
    sleep 3
fi

# 1. 启动后端服务 (后台运行)
echo "[1/2] 正在启动后端服务 (端口 8000)..."
python3 comfy_agent.py &
BACKEND_PID=$!

# 2. 启动前端服务 (后台运行)
echo "[2/2] 正在启动前端服务 (端口 5500)..."
python3 -m http.server 5500 &
FRONTEND_PID=$!

echo "========================================"
echo "服务已成功启动！"
echo "手机端请确保连接同WiFi，并访问此电脑的局域网IP，例如: http://192.168.X.X:5500"
echo "按 [Ctrl+C] 退出并关闭所有服务。"
echo "========================================"

# 3. 自动打开浏览器（仅限Mac，Linux请注释掉这行）
open http://localhost:5500

# 捕捉 Ctrl+C 信号，以便连同后台服务一起关闭
trap "echo '正在关闭服务...'; kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT

wait