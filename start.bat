@echo off
chcp 65001 >nul
echo ========================================
echo       正在启动 AI Agent 服务...
echo ========================================

:: 0. 检测并重置 Ollama 进程
echo [0/3] 正在检测并清理 Ollama 进程状态...
tasklist | find /i "ollama" >nul
if %errorlevel% equ 0 (
    echo [*] 检测到 Ollama 已在后台运行，正在清理旧进程以防止冲突...
    :: 强制杀死所有名为 ollama.exe 和 ollama app.exe 的进程
    taskkill /f /im ollama.exe >nul 2>nul
    taskkill /f /im "ollama app.exe" >nul 2>nul
    :: 稍微等待1秒确保端口释放
    timeout /t 1 >nul
) else (
    echo [*] Ollama 当前未运行，准备全新启动。
)

echo [*] 正在启动 Ollama 桌面端程序接管大模型...
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" (
    start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
) else (
    start "" ollama
)
:: 给予 5 秒钟的显存加载和程序初始化时间
timeout /t 5 >nul

:: 1. 启动 Python 后端服务 (运行在新窗口)
echo [1/3] 正在启动后端服务 (端口 8000)...
start "AI Backend" cmd /k "python comfy_agent.py"

:: 2. 启动前端静态网页服务 (运行在新窗口)
echo [2/3] 正在启动前端服务 (端口 5500)...
start "AI Frontend" cmd /k "python -m http.server 5500"

:: 3. 延迟 2 秒后自动打开电脑浏览器测试
timeout /t 2 >nul
start http://localhost:5500

echo ========================================
echo 服务已全部启动！
echo 电脑端请访问: http://localhost:5500
echo 手机端请确保连接同WiFi，并访问你的局域网IP (例如: http://192.168.X.X:5500)
echo ========================================
pause