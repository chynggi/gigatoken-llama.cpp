$ErrorActionPreference = "Stop"

$ZIP  = "wikitext-2-raw-v1.zip"
$FILE = Join-Path "wikitext-2-raw" "wiki.test.raw"
$URL  = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/$ZIP"

function Die([string]$Message) {
    [Console]::Error.WriteLine($Message)
    exit 1
}

function Have-Command([string]$Command) {
    return $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

function Download-File([string]$Url, [string]$Output) {
    if (Test-Path -LiteralPath $Output -PathType Leaf) {
        return
    }

    if (Have-Command "wget") {
        & wget $Url -O $Output
        if ($LASTEXITCODE -ne 0) {
            Die "wget failed"
        }
    }
    elseif (Have-Command "curl") {
        & curl.exe -L $Url -o $Output
        if ($LASTEXITCODE -ne 0) {
            Die "curl failed"
        }
    }
    else {
        try {
            Invoke-WebRequest -Uri $Url -OutFile $Output
        }
        catch {
            Die "Please install wget or curl, or use a PowerShell version with Invoke-WebRequest support"
        }
    }
}

if (-not (Have-Command "Expand-Archive")) {
    Die "PowerShell Expand-Archive is required"
}

if (-not (Test-Path -LiteralPath $FILE -PathType Leaf)) {
    Download-File $URL $ZIP

    Expand-Archive -LiteralPath $ZIP -DestinationPath "." -Force

    Remove-Item -LiteralPath $ZIP -Force -ErrorAction SilentlyContinue
}

Write-Host @"
Usage:

  llama-perplexity -m model.gguf -f $FILE [other params]

"@
