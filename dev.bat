@echo off
chcp 65001 >nul
echo ========================================
echo   TRPG Agent 开发模式 (热更新)
echo ========================================
echo.

REM 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python
    pause
    exit /b 1
)

REM 检查 Node.js (尝试自动安装)
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] 未找到 Node.js，尝试自动安装...
    where winget >nul 2>&1
    if %errorlevel% equ 0 (
        winget install OpenJS.NodeJS.LTS --silent --accept-package-agreements --accept-source-agreements
        if %errorlevel% equ 0 (
            echo [✓] Node.js 安装成功，请重新运行此脚本
        ) else (
            echo [错误] Node.js 安装失败，请手动安装: https://nodejs.org
        )
    ) else (
        echo [错误] 未找到 Node.js 且 winget 不可用，请手动安装: https://nodejs.org
    )
    pause
    exit /b 1
)

REM 从 .env 文件加载 API Key
if "%DEEPSEEK_API_KEY%"=="" (
    if exist ".env" (
        for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
            if "%%a"=="DEEPSEEK_API_KEY" set "DEEPSEEK_API_KEY=%%b"
        )
    )
)

if "%DEEPSEEK_API_KEY%"=="" (
    set /p API_KEY=请输入你的 DeepSeek API Key:
    if "!API_KEY!"=="" (
        echo [错误] API Key 不能为空
        pause
        exit /b 1
    )
    set DEEPSEEK_API_KEY=!API_KEY!
)

echo [1/3] 安装 Python 依赖...
pip install fastapi uvicorn python-dotenv >nul 2>&1

echo [2/3] 安装前端依赖...
cd web
call npm install >nul 2>&1
cd ..

echo [3/3] 启动服务...
echo.
echo ========================================
echo   前端 (热更新): http://localhost:5173
echo   后端 API:     http://localhost:8000
echo   按 Ctrl+C 停止所有服务
echo ========================================
echo.

REM 设置环境变量
set HF_ENDPOINT=https://hf-mirror.com

REM 在同一窗口启动两个进程: Vite 后台 + Python 前台
start "Vite Dev Server" /b cmd /c "cd web && npx vite --host"
REM 等 Vite 启动
timeout /t 3 /nobreak >nul
python -m trpg_agent.api_server

REM Ctrl+C 后会到这里，关闭 Vite 进程
taskkill /f /fi "WINDOWTITLE eq Vite Dev Server" >nul 2>&1
pause
