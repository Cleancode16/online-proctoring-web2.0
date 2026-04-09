# Start Frontend for Online Proctoring System
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Online Proctoring - Frontend" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Navigate to frontend directory
Set-Location ".\client\proctoring"

# Check if node_modules exists
if (-Not (Test-Path ".\node_modules")) {
    Write-Host "⚠️  Node modules not found. Installing..." -ForegroundColor Yellow
    npm install
    Write-Host "✅ Node modules installed`n" -ForegroundColor Green
} else {
    Write-Host "✅ Node modules already installed`n" -ForegroundColor Green
}

# Start development server
Write-Host "🚀 Starting frontend development server..." -ForegroundColor Green
Write-Host "   Frontend will run on http://localhost:5173" -ForegroundColor Cyan
Write-Host "   Press Ctrl+C to stop`n" -ForegroundColor Yellow
npm run dev
