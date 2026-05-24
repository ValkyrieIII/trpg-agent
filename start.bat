@echo off
chcp 65001 >nul
echo ========================================
echo   TRPG Agent 一键启动
echo ========================================
echo.

REM 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.13+
    pause
    exit /b 1
)

REM 检查 Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Node.js，请先安装 Node.js
    pause
    exit /b 1
)

REM 检查 API Key
if "%DEEPSEEK_API_KEY%"=="" (
    echo [提示] 未设置 DEEPSEEK_API_KEY 环境变量
    set /p API_KEY=请输入你的 DeepSeek API Key: 
    if "!API_KEY!"=="" (
        echo [错误] API Key 不能为空
        pause
        exit /b 1
    )
    set DEEPSEEK_API_KEY=%API_KEY%
)

echo [1/5] 安装 Python 依赖...
pip install fastapi uvicorn >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] pip install 失败，尝试继续...
)

echo [2/5] 安装前端依赖...
cd web
call npm install >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] npm install 失败，尝试继续...
)
cd ..

echo [3/5] 构建前端...
cd web
call npm run build >nul 2>&1
cd ..

echo [4/5] 启动后端服务...
echo [提示] 按 Ctrl+C 可停止服务
echo.
echo ========================================
echo   浏览器打开: http://localhost:8000
echo ========================================
echo.

set HF_ENDPOINT=https://hf-mirror.com
python -m trpg_agent.api_server

pause
