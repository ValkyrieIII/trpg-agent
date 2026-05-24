# ========================================
#   TRPG Agent 一键启动脚本
#   用法: .\start.ps1
# ========================================

$ErrorActionPreference = "Stop"
$originalEncoding = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  TRPG Agent 启动器" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[✓] $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[✗] 未找到 Python，请先安装 Python 3.13+" -ForegroundColor Red
    Read-Host "按回车键退出"
    exit 1
}

# 检查 Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "[✓] Node.js $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "[✗] 未找到 Node.js，请先安装 Node.js" -ForegroundColor Red
    Read-Host "按回车键退出"
    exit 1
}

# 检查 API Key
if (-not $env:DEEPSEEK_API_KEY) {
    Write-Host "[提示] 未设置 DEEPSEEK_API_KEY 环境变量" -ForegroundColor Yellow
    $apiKey = Read-Host "请输入你的 DeepSeek API Key"
    if (-not $apiKey) {
        Write-Host "[错误] API Key 不能为空" -ForegroundColor Red
        Read-Host "按回车键退出"
        exit 1
    }
    $env:DEEPSEEK_API_KEY = $apiKey
    Write-Host "[✓] API Key 已设置（本次会话有效）" -ForegroundColor Green
} else {
    Write-Host "[✓] DEEPSEEK_API_KEY 已设置" -ForegroundColor Green
}

Write-Host ""

# 安装 Python 依赖
Write-Host "[1/4] 安装 Python 依赖..." -ForegroundColor Cyan
try {
    pip install fastapi uvicorn -q 2>$null
    Write-Host "[✓] Python 依赖就绪" -ForegroundColor Green
} catch {
    Write-Host "[!] Python 依赖安装失败，将继续启动..." -ForegroundColor Yellow
}

# 安装前端依赖
Write-Host "[2/4] 安装前端依赖..." -ForegroundColor Cyan
Push-Location web
try {
    npm install --silent 2>$null
    Write-Host "[✓] 前端依赖就绪" -ForegroundColor Green
} catch {
    Write-Host "[!] 前端依赖安装失败，将继续启动..." -ForegroundColor Yellow
}
Pop-Location

# 构建前端
Write-Host "[3/4] 构建前端..." -ForegroundColor Cyan
Push-Location web
try {
    npm run build 2>$null | Out-Null
    Write-Host "[✓] 前端构建完成" -ForegroundColor Green
} catch {
    Write-Host "[!] 前端构建失败，请手动运行 'cd web && npm run build'" -ForegroundColor Yellow
}
Pop-Location

# 启动服务
Write-Host ""
Write-Host "[4/4] 启动服务..." -ForegroundColor Cyan
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  浏览器打开: http://localhost:8000" -ForegroundColor Green
Write-Host "  按 Ctrl+C 可停止服务" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

$env:HF_ENDPOINT = "https://hf-mirror.com"
python -m trpg_agent.api_server
