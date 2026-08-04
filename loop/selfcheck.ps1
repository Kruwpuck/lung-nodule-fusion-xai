<#
Cek kondisi berhenti loop/loop.ps1 tanpa memanggil claude sama sekali.

Kenapa ada: seluruh keamanan loop bergantung pada driver yang benar-benar
berhenti. Kalau salah satu kondisi berhenti rusak, loop akan memanggil model
semalaman tanpa menghasilkan apa pun. Nol pemanggilan API di sini, jadi murah
dijalankan setiap kali loop.ps1 disunting.

Pakai:
  powershell -ExecutionPolicy Bypass -File loop\selfcheck.ps1

STATE.json dan ledger dicadangkan ke .bak lalu dipulihkan di blok finally,
termasuk kalau salah satu assert gagal.
#>

$ErrorActionPreference = "Stop"

$Root      = Split-Path -Parent $PSScriptRoot
$RunId     = "2026-08-04-run01"
$StatePath = Join-Path $Root "STATE.json"
$Ledger    = Join-Path $Root "runs\$RunId\ledger.md"
$Driver    = Join-Path $Root "loop\loop.ps1"

$StateBak  = "$StatePath.bak"
$LedgerBak = "$Ledger.bak"

$failed = 0

function Assert([bool]$Cond, [string]$What) {
    if ($Cond) {
        Write-Host "  ok   $What" -ForegroundColor Green
    }
    else {
        Write-Host "  GAGAL $What" -ForegroundColor Red
        $script:failed = $script:failed + 1
    }
}

function Set-Synthetic([hashtable]$Fields) {
    $s = [ordered]@{
        run_id                = $RunId
        iterasi               = 0
        level                 = 1
        gagal_beruntun        = 0
        commit_hijau_terakhir = "selfcheck"
        status                = "running"
        menunggu_manusia      = $false
        diperbarui            = "2000-01-01T00:00:00Z"
    }
    foreach ($k in $Fields.Keys) { $s[$k] = $Fields[$k] }
    ($s | ConvertTo-Json) | Set-Content -Path $StatePath -Encoding utf8
}

function Get-State { Get-Content $StatePath -Raw | ConvertFrom-Json }

Copy-Item $StatePath $StateBak
Copy-Item $Ledger    $LedgerBak

try {
    Write-Host "1. level 6 harus berhenti dan menandai blocked"
    Set-Synthetic @{ level = 6 }
    & $Driver -RunId $RunId | Out-Null
    $s = Get-State
    Assert ($s.status -eq "blocked")       "status jadi blocked"
    Assert ($s.menunggu_manusia -eq $true) "menunggu_manusia jadi true"
    Assert ((Get-Content $Ledger -Raw) -match "lvl 6 \| BLOCKED") "ledger dapat baris BLOCKED"

    Write-Host "2. iterasi >= MaxIter harus berhenti dan menandai blocked"
    Copy-Item $LedgerBak $Ledger
    Set-Synthetic @{ iterasi = 99 }
    & $Driver -RunId $RunId -MaxIter 15 | Out-Null
    $s = Get-State
    Assert ($s.status -eq "blocked") "status jadi blocked pada MAX_ITER"
    Assert ((Get-Content $Ledger -Raw) -match "MAX_ITER 15 tercapai") "ledger menyebut MAX_ITER"

    Write-Host "3. status bukan running harus langsung keluar tanpa menyentuh apa pun"
    Copy-Item $LedgerBak $Ledger
    Set-Synthetic @{ status = "done" }
    $ledgerBefore = Get-Content $Ledger -Raw
    & $Driver -RunId $RunId | Out-Null
    $s = Get-State
    Assert ($s.status -eq "done")                          "status done tidak diubah"
    Assert ((Get-Content $Ledger -Raw) -eq $ledgerBefore)  "ledger tidak disentuh"

    Write-Host "4. DryRun pada level 1 tidak boleh mengubah STATE.json"
    Copy-Item $LedgerBak $Ledger
    Set-Synthetic @{ }
    $stampBefore = (Get-State).diperbarui
    & $Driver -RunId $RunId -DryRun | Out-Null
    Assert ((Get-State).diperbarui -eq $stampBefore) "DryRun tidak menulis STATE.json"
}
finally {
    Copy-Item $StateBak  $StatePath
    Copy-Item $LedgerBak $Ledger
    Remove-Item $StateBak
    Remove-Item $LedgerBak
    Write-Host ""
    Write-Host "STATE.json dan ledger dipulihkan."
}

if ($failed -gt 0) {
    Write-Host "$failed assert gagal." -ForegroundColor Red
    exit 1
}
Write-Host "semua assert lulus." -ForegroundColor Green
