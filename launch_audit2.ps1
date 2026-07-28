$base = "C:\Users\Adaptive Network\Documents\Lung Cancer\lung-nodule-fusion-xai"
$py = Join-Path $base ".venv\Scripts\python.exe"
$script = Join-Path $base "_audit_missing.py"
$outLog = Join-Path $base "_audit_out.log"
$errLog = Join-Path $base "_audit_err.log"

Remove-Item $outLog, $errLog -ErrorAction SilentlyContinue

$p = Start-Process -FilePath $py -ArgumentList $script -WorkingDirectory $base -RedirectStandardOutput $outLog -RedirectStandardError $errLog -NoNewWindow -PassThru
Start-Sleep -Seconds 3
$still = Get-Process -Id $p.Id -ErrorAction SilentlyContinue
Write-Output "PID=$($p.Id) StillRunning=$($null -ne $still)"
