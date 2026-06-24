[CmdletBinding()]
param(
    [switch]$Build,
    [switch]$Detach,
    [switch]$SkipOllamaCreate,
    [string]$OllamaModelName = "question-rewriter-qwen3-0.6b",
    [string]$OllamaContainerName = "question-rewriter-ollama",
    [string]$OllamaServiceName = "ollama"
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$outputDir = Join-Path $repoRoot "outputs\question-rewriter-qwen3-0.6b"
$modelfilePath = Join-Path $outputDir "Modelfile"
$composeArgs = @("compose", "up")

function Invoke-OllamaCreateViaDocker {
    param(
        [string]$ContainerName,
        [string]$ModelName,
        [string]$LocalModelfilePath,
        [string]$LocalGgufPath
    )

    $remoteDir = "/tmp/question-rewriter-ollama"
    $remoteModelfilePath = "$remoteDir/Modelfile"

    & docker exec $ContainerName sh -lc "mkdir -p $remoteDir"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to prepare Ollama container directory with exit code $LASTEXITCODE"
    }

    & docker cp $LocalModelfilePath "${ContainerName}:${remoteModelfilePath}"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to copy Modelfile into Ollama container with exit code $LASTEXITCODE"
    }

    & docker cp $LocalGgufPath "${ContainerName}:${remoteDir}/"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to copy GGUF artifact into Ollama container with exit code $LASTEXITCODE"
    }

    & docker exec $ContainerName ollama create $ModelName -f $remoteModelfilePath
    if ($LASTEXITCODE -ne 0) {
        throw "docker exec ollama create failed with exit code $LASTEXITCODE"
    }
}

function Ensure-OllamaDockerContainer {
    param(
        [string]$ContainerName,
        [string]$ServiceName
    )

    $existingImageOutput = & docker ps -a --filter "name=^/${ContainerName}$" --format "{{.Image}}"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to inspect Ollama Docker container with exit code $LASTEXITCODE"
    }
    $existingImage = if ($null -ne $existingImageOutput) { "$existingImageOutput".Trim() } else { "" }

    $isRunningOutput = & docker ps --filter "name=^/${ContainerName}$" --format "{{.Names}}"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to inspect running Ollama Docker container state with exit code $LASTEXITCODE"
    }
    $isRunning = if ($null -ne $isRunningOutput) { "$isRunningOutput".Trim() } else { "" }

    if (-not $existingImage) {
        Write-Host "Starting compose-managed Ollama service '$ServiceName'." -ForegroundColor Yellow
        & docker compose up -d $ServiceName
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to start Ollama compose service with exit code $LASTEXITCODE"
        }
    } elseif (-not $isRunning) {
        Write-Host "Starting existing compose-managed Ollama service '$ServiceName'." -ForegroundColor Yellow
        & docker compose start $ServiceName
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to start existing Ollama compose service with exit code $LASTEXITCODE"
        }
    }

    $maxAttempts = 30
    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        & docker exec $ContainerName ollama list | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return
        }

        Start-Sleep -Seconds 2
    }

    throw "Ollama Docker container '$ContainerName' did not become ready in time."
}

if ($Build) {
    $composeArgs += "--build"
} else {
    $composeArgs += "--no-build"
}

if ($Detach) {
    $composeArgs += "--detach"
} else {
    $composeArgs += "--abort-on-container-exit"
    $composeArgs += "--exit-code-from"
    $composeArgs += "fine-tuning"
}

$composeArgs += "fine-tuning"

Write-Host "Starting fine-tuning from $repoRoot" -ForegroundColor Cyan
if ($Build) {
    Write-Host "Build mode enabled; Docker will rebuild the fine-tuning image first." -ForegroundColor Yellow
} else {
    Write-Host "Reusing the existing fine-tuning image. Pass -Build when Dockerfile or dependency changes require a rebuild." -ForegroundColor Yellow
}
Write-Host ("Running: docker " + ($composeArgs -join " ")) -ForegroundColor DarkCyan

Push-Location $repoRoot
try {
    & docker @composeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose up failed with exit code $LASTEXITCODE"
    }

    if ($Detach) {
        Write-Host "Detached mode enabled; skipping Ollama model creation until training finishes." -ForegroundColor Yellow
        return
    }

    if ($SkipOllamaCreate) {
        Write-Host "Skipping Ollama model creation by request." -ForegroundColor Yellow
        return
    }

    if (-not (Test-Path $modelfilePath)) {
        throw "Training completed but no Modelfile was found at $modelfilePath"
    }

    $ggufPath = Get-ChildItem -Path $outputDir -Filter *.gguf -File | Select-Object -First 1
    if (-not $ggufPath) {
        throw "Training completed but no GGUF artifact was found in $outputDir"
    }

    Ensure-OllamaDockerContainer -ContainerName $OllamaContainerName -ServiceName $OllamaServiceName
    Write-Host "Creating Ollama model '$OllamaModelName' in Docker container '$OllamaContainerName' from $modelfilePath" -ForegroundColor Cyan
    Invoke-OllamaCreateViaDocker `
        -ContainerName $OllamaContainerName `
        -ModelName $OllamaModelName `
        -LocalModelfilePath $modelfilePath `
        -LocalGgufPath $ggufPath.FullName
}
finally {
    Pop-Location
}
