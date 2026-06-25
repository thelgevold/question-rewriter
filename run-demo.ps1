[CmdletBinding()]
param(
    [switch]$Build
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$ollamaContainerName = "question-rewriter-ollama"

function Wait-ForOllama {
    param(
        [string]$ContainerName
    )

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

Push-Location $repoRoot
try {
    if ($Build) {
        Write-Host "Building the shared runtime image for the demo." -ForegroundColor Yellow
        & docker compose build demo
        if ($LASTEXITCODE -ne 0) {
            throw "docker compose build demo failed with exit code $LASTEXITCODE"
        }
    }

    Write-Host "Starting the Docker Ollama service." -ForegroundColor Cyan
    & docker compose up -d ollama
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose up -d ollama failed with exit code $LASTEXITCODE"
    }

    Write-Host "Waiting for the Docker Ollama service to become ready." -ForegroundColor Cyan
    Wait-ForOllama -ContainerName $ollamaContainerName

    Write-Host "Running the demo container." -ForegroundColor Cyan
    & docker compose run --rm demo
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose run --rm demo failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
