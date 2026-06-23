$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ResultsDir = Join-Path $ScriptDir "sweep_results"
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

$BuildDir = Join-Path $ScriptDir "Build_msvc"
$Binary = Join-Path $BuildDir "src\Release\benchmarksuite-main.exe"

Write-Host "========================================================"
Write-Host " MSVC SWEEP -- CPB 1-16 & BPS 1-16 (Powers of 2)"
Write-Host "========================================================"

foreach ($cpb in @(1, 2, 4, 8, 16)) {
    foreach ($bps in @(1, 2, 4, 8, 16)) {
        Write-Host ""
        Write-Host "--------------------------------------------------------"
        Write-Host " MSVC | CHUNKS_PER_BLOCK=$cpb | BLOCKS_PER_STEP=$bps"
        Write-Host "--------------------------------------------------------"

        if (Test-Path $BuildDir) {
            Remove-Item -Recurse -Force $BuildDir
        }

        cmake -S . -B $BuildDir `
            -DCMAKE_BUILD_TYPE=Release `
            -DSIMDJSON_DEVELOPMENT_CHECKS=OFF `
            -DBNCH_SWT_BENCHMARKS=ON `
            -DCHUNKS_PER_BLOCK="$cpb" `
            -DBLOCKS_PER_STEP="$bps"

        if ($LASTEXITCODE -ne 0) { throw "CMake configure failed for cpb=$cpb bps=$bps" }

        $Jobs = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
        cmake --build $BuildDir --config Release --parallel $Jobs

        if ($LASTEXITCODE -ne 0) { throw "CMake build failed for cpb=$cpb bps=$bps" }

        $ResultFile = Join-Path $ResultsDir "msvc_cpb${cpb}_bps${bps}.txt"
        "# COMPILER=msvc CHUNKS_PER_BLOCK=$cpb BLOCKS_PER_STEP=$bps" | Set-Content $ResultFile

        if (Test-Path $Binary) {
            & $Binary 2>&1 | Where-Object {
                $_ -notmatch "^---$|^All correct|^Building|entries:|buf size:|\[bench\]"
            } | Add-Content $ResultFile
        } else {
            "ERROR: Binary not found at $Binary" | Add-Content $ResultFile
        }

        Write-Host "Saved: $ResultFile"
        Get-Content $ResultFile
    }
}

Write-Host ""
Write-Host "========================================================"
Write-Host " MSVC SWEEP COMPLETE -- SUMMARY"
Write-Host "========================================================"
Write-Host ""

Write-Host ("{0,-6} {1,-6} {2,-22} {3,-22} {4,-22}" -f "CPB", "BPS", "ASCII jsonifier", "MIXED jsonifier", "MULTIBYTE jsonifier")
Write-Host "--------------------------------------------------------------------------------"

foreach ($cpb in @(1, 2, 4, 8, 16)) {
    foreach ($bps in @(1, 2, 4, 8, 16)) {
        $ResultFile = Join-Path $ResultsDir "msvc_cpb${cpb}_bps${bps}.txt"
        if (-not (Test-Path $ResultFile)) { continue }

        $Lines = Get-Content $ResultFile
        $JsonLines = $Lines | Where-Object { $_ -match "jsonifier" }

        $AsciiJ = if ($JsonLines.Count -ge 1) { ($JsonLines[0] -split ",")[1] } else { "N/A" }
        $MixedJ = if ($JsonLines.Count -ge 2) { ($JsonLines[1] -split ",")[1] } else { "N/A" }
        $MultiJ = if ($JsonLines.Count -ge 3) { ($JsonLines[2] -split ",")[1] } else { "N/A" }

        Write-Host ("{0,-6} {1,-6} {2,-22} {3,-22} {4,-22}" -f $cpb, $bps, $AsciiJ, $MixedJ, $MultiJ)
    }
}

Write-Host ""
Write-Host "--- simdjson baseline ---"

$BaselineFile = Join-Path $ResultsDir "msvc_cpb1_bps1.txt"
if (Test-Path $BaselineFile) {
    $Lines = Get-Content $BaselineFile
    $SimdLines = $Lines | Where-Object { $_ -match "simdjson" }

    $AsciiS = if ($SimdLines.Count -ge 1) { ($SimdLines[0] -split ",")[1] } else { "N/A" }
    $MixedS = if ($SimdLines.Count -ge 2) { ($SimdLines[1] -split ",")[1] } else { "N/A" }
    $MultiS = if ($SimdLines.Count -ge 3) { ($SimdLines[2] -split ",")[1] } else { "N/A" }

    Write-Host ("{0,-6} {1,-6} {2,-22} {3,-22} {4,-22}" -f "smdj", "smdj", $AsciiS, $MixedS, $MultiS)
} else {
    Write-Host "Baseline file $BaselineFile not found."
}

Write-Host ""
Write-Host "Raw results in: $ResultsDir\msvc_cpb*_bps*.txt"