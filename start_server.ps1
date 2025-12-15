# Start Flask Server in Background
Write-Host "Starting Flask server in the background..." -ForegroundColor Green

# Stop any existing Python processes running app.py
Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -like "*python*"
} | ForEach-Object {
    try {
        $_.Kill()
        Start-Sleep -Milliseconds 500
    } catch {
        # Ignore errors
    }
}

# Start Flask app in background (hidden window)
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "python"
$psi.Arguments = "app.py"
$psi.WorkingDirectory = $PSScriptRoot
$psi.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
$psi.CreateNoWindow = $true
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true

$process = [System.Diagnostics.Process]::Start($psi)

Start-Sleep -Seconds 3

# Test if server is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/api/data/status" -UseBasicParsing -TimeoutSec 5
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✓ Server started successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Access your website at:" -ForegroundColor Cyan
    Write-Host "  http://localhost:5000" -ForegroundColor Yellow
    Write-Host "  http://127.0.0.1:5000" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Server Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host ""
    Write-Host "To stop the server, run:" -ForegroundColor Gray
    Write-Host "  Get-Process python | Stop-Process -Force" -ForegroundColor Gray
    Write-Host ""
} catch {
    Write-Host ""
    Write-Host "Server may still be starting..." -ForegroundColor Yellow
    Write-Host "Please wait a moment and open:" -ForegroundColor Yellow
    Write-Host "  http://localhost:5000" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "If it doesn't start, run 'python app.py' directly to see errors." -ForegroundColor Gray
}

