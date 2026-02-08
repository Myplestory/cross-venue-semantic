# PowerShell setup script for testing environment

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Setting up Market Discovery Test Environment" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install pytest and testing dependencies
Write-Host ""
Write-Host "Installing testing dependencies..." -ForegroundColor Yellow
pip install pytest pytest-asyncio pytest-mock pytest-cov

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the virtual environment:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To run tests:" -ForegroundColor Yellow
Write-Host "  pytest discovery/tests/ -v" -ForegroundColor White
Write-Host ""
Write-Host "To deactivate:" -ForegroundColor Yellow
Write-Host "  deactivate" -ForegroundColor White
Write-Host ""

