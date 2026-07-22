param(
    [Parameter(Mandatory = $true)]
    [string] $OffExecutable,

    [Parameter(Mandatory = $true)]
    [string] $OnExecutable,

    [string[]] $Models = @(
        (Join-Path $PSScriptRoot "../models/ggml-vocab-gpt-2.gguf"),
        (Join-Path $PSScriptRoot "../models/ggml-vocab-llama-spm.gguf")
    ),

    [int] $Iterations = 30,

    [long[]] $Sizes = @(1024, 65536, 1048576, 16777216),

    [string] $Output = (Join-Path (Get-Location) "gigatoken-benchmark.json"),

    [int] $BootstrapSamples = 10000
)

$ErrorActionPreference = "Stop"

function Get-Median([double[]] $Values) {
    $sorted = @($Values | Sort-Object)
    $middle = [int] [Math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2 -eq 0) {
        return ($sorted[$middle - 1] + $sorted[$middle]) / 2.0
    }
    return [double] $sorted[$middle]
}

function Get-BootstrapInterval([double[]] $Ratios, [int] $Count, [int] $Seed) {
    $random = [System.Random]::new($Seed)
    $statistics = [double[]]::new($Count)
    $sample = [double[]]::new($Ratios.Count)
    for ($iteration = 0; $iteration -lt $Count; ++$iteration) {
        for ($index = 0; $index -lt $Ratios.Count; ++$index) {
            $sample[$index] = $Ratios[$random.Next($Ratios.Count)]
        }
        $statistics[$iteration] = Get-Median $sample
    }
    [Array]::Sort($statistics)
    $lower = $statistics[[int] [Math]::Floor(0.025 * ($Count - 1))]
    $upper = $statistics[[int] [Math]::Ceiling(0.975 * ($Count - 1))]
    return @($lower, $upper)
}

function Invoke-Benchmark([string] $Executable, [string] $Model, [string] $Label) {
    $arguments = @($Model, $Label, $Iterations) + $Sizes
    $json = & $Executable @arguments | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark failed: $Executable"
    }
    return $json | ConvertFrom-Json
}

$modelReports = @()
foreach ($model in $Models) {
    $off = Invoke-Benchmark $OffExecutable $model "off"
    $on = Invoke-Benchmark $OnExecutable $model "on"
    if ($off.results.Count -ne $on.results.Count) {
        throw "OFF and ON result counts differ for $model"
    }

    $comparisons = @()
    for ($resultIndex = 0; $resultIndex -lt $off.results.Count; ++$resultIndex) {
        $offResult = $off.results[$resultIndex]
        $onResult = $on.results[$resultIndex]
        if ($offResult.size_bytes -ne $onResult.size_bytes) {
            throw "OFF and ON sizes differ for $model"
        }
        if ($offResult.output_hash -ne $onResult.output_hash) {
            throw "Tokenizer output hash mismatch for $model at $($offResult.size_bytes) bytes"
        }

        $ratios = [double[]]::new($Iterations)
        for ($sample = 0; $sample -lt $Iterations; ++$sample) {
            $ratios[$sample] = [double] $offResult.samples_us[$sample] / [double] $onResult.samples_us[$sample]
        }
        $interval = Get-BootstrapInterval $ratios $BootstrapSamples (0x4754 + $resultIndex)
        $comparisons += [ordered]@{
            size_bytes = [long] $offResult.size_bytes
            speedup_median = Get-Median $ratios
            speedup_bootstrap_95_lower = $interval[0]
            speedup_bootstrap_95_upper = $interval[1]
            output_hash = $offResult.output_hash
            off_median_us = [double] $offResult.median_us
            on_median_us = [double] $onResult.median_us
            off_p95_us = [double] $offResult.p95_us
            on_p95_us = [double] $onResult.p95_us
            off_mb_per_second = [double] $offResult.mb_per_second
            on_mb_per_second = [double] $onResult.mb_per_second
        }
    }

    $breakEven = $null
    foreach ($comparison in ($comparisons | Sort-Object size_bytes)) {
        if ($comparison.speedup_bootstrap_95_lower -gt 1.0) {
            $breakEven = $comparison.size_bytes
            break
        }
    }
    $oneMiB = $comparisons | Where-Object size_bytes -eq 1048576 | Select-Object -First 1
    $modelReports += [ordered]@{
        model = (Resolve-Path $model).Path
        initialization = [ordered]@{
            off_backend_us = [long] $off.backend_init_us
            on_backend_us = [long] $on.backend_init_us
            off_tokenizer_us = [long] $off.tokenizer_init_us
            on_tokenizer_us = [long] $on.tokenizer_init_us
        }
        peak_rss_bytes = [ordered]@{
            off = [long] $off.peak_rss_bytes
            on = [long] $on.peak_rss_bytes
        }
        break_even_bytes = $breakEven
        one_mib_significant = ($null -ne $oneMiB -and $oneMiB.speedup_bootstrap_95_lower -gt 1.0)
        comparisons = $comparisons
        raw = [ordered]@{ off = $off; on = $on }
    }
}

$report = [ordered]@{
    schema = 1
    generated_at_utc = [DateTime]::UtcNow.ToString("o")
    iterations = $Iterations
    bootstrap_samples = $BootstrapSamples
    poc_success = (@($modelReports | Where-Object { -not $_.one_mib_significant }).Count -eq 0)
    models = $modelReports
}

$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Output -Encoding utf8NoBOM
Write-Host "Wrote $Output"
Write-Host "PoC success: $($report.poc_success)"
