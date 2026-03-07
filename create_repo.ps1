# Create private GitHub repository script
$repoName = "Paper-Archive-2026"
$username = "stalinnn"
$isPrivate = $true

# Check for GitHub token
$token = $env:GITHUB_TOKEN
if (-not $token) {
    Write-Host "Need GitHub Personal Access Token to create repository"
    Write-Host "Get token: https://github.com/settings/tokens (need 'repo' permission)"
    Write-Host ""
    $secureToken = Read-Host "Enter your GitHub Personal Access Token" -AsSecureString
    $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
    $token = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
    
    if (-not $token -or $token.Length -eq 0) {
        Write-Host "Error: No token provided"
        exit 1
    }
}

# Create repository API request
$headers = @{
    "Authorization" = "token $token"
    "Accept" = "application/vnd.github.v3+json"
}

$body = @{
    name = $repoName
    private = $isPrivate
    description = "Paper Archive 2026"
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "https://api.github.com/user/repos" -Method Post -Headers $headers -Body $body -ContentType "application/json"
    Write-Host "Successfully created private repository: $($response.html_url)"
    Write-Host ""
    Write-Host "Adding remote and pushing code..."
    
    # Remove old remote if exists
    $existingRemote = git remote get-url origin 2>$null
    if ($existingRemote) {
        git remote remove origin
    }
    
    # Add remote and push
    git remote add origin $response.clone_url
    Write-Host "Remote added, pushing to main..."
    git push -u origin main
    Write-Host "Done!"
} catch {
    Write-Host "Error: $($_.Exception.Message)"
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response: $responseBody"
    }
    exit 1
}
