# ========================================
#   TRPG Agent 开发模式 (热更新)
#   用法: .\dev.ps1
# ========================================

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  TRPG Agent 开发模式 (热更新)" -ForegroundColor Cyan
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

# 检查 Node.js (尝试自动安装)
try {
    $nodeVersion = node --version 2>&1
    Write-Host "[✓] Node.js $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "[提示] 未找到 Node.js，尝试自动安装..." -ForegroundColor Yellow
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        winget install OpenJS.NodeJS.LTS --silent --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[✓] Node.js 安装成功，请重新运行此脚本" -ForegroundColor Green
        } else {
            Write-Host "[✗] Node.js 安装失败，请手动安装: https://nodejs.org" -ForegroundColor Red
        }
    } else {
        Write-Host "[✗] 未找到 Node.js 且 winget 不可用，请手动安装: https://nodejs.org" -ForegroundColor Red
    }
    Read-Host "按回车键退出"
    exit 1
}

# 从 .env 文件加载 API Key
if ((-not $env:DEEPSEEK_API_KEY) -and (Test-Path ".env")) {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*DEEPSEEK_API_KEY\s*=\s*(.+)\s*$') {
            $env:DEEPSEEK_API_KEY = $matches[1].Trim()
        }
    }
}

if (-not $env:DEEPSEEK_API_KEY) {
    Write-Host "[提示] 未设置 DEEPSEEK_API_KEY，请在 .env 文件中配置或设置环境变量" -ForegroundColor Yellow
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
Write-Host "[1/3] 安装 Python 依赖..." -ForegroundColor Cyan
try {
    pip install fastapi uvicorn python-dotenv -q 2>$null
    Write-Host "[✓] Python 依赖就绪" -ForegroundColor Green
} catch {
    Write-Host "[!] Python 依赖安装失败，将继续启动..." -ForegroundColor Yellow
}

# 安装前端依赖
Write-Host "[2/3] 安装前端依赖..." -ForegroundColor Cyan
Push-Location web
try {
    npm install --silent 2>$null
    Write-Host "[✓] 前端依赖就绪" -ForegroundColor Green
} catch {
    Write-Host "[!] 前端依赖安装失败，将继续启动..." -ForegroundColor Yellow
}
Pop-Location

# 启动服务
Write-Host "[3/3] 启动服务..." -ForegroundColor Cyan
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  前端 (热更新): http://localhost:5173" -ForegroundColor Green
Write-Host "  后端 API:      http://localhost:8000" -ForegroundColor Green
Write-Host "  按 Ctrl+C 停止所有服务" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

$env:HF_ENDPOINT = "https://hf-mirror.com"

# 启动 Vite 后台进程
$viteJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD/web
    npx vite --host
}

try {
    python -m trpg_agent.api_server
} finally {
    Stop-Job -Job $viteJob
    Remove-Job -Job $viteJob
    Write-Host "开发服务已停止" -ForegroundColor Yellow
}
