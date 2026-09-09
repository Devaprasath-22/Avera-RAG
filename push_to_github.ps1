Write-Host "Pushing Avera-RAG to https://github.com/Devaprasath-22/Avera-RAG.git..." -ForegroundColor Cyan
git push -u origin main
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSuccessfully pushed to GitHub! View at: https://github.com/Devaprasath-22/Avera-RAG" -ForegroundColor Green
} else {
    Write-Host "`nPush failed because the repository does not exist on GitHub yet." -ForegroundColor Yellow
    Write-Host "Please create the empty repo at: https://github.com/new?name=Avera-RAG" -ForegroundColor Yellow
    Write-Host "Then re-run this script." -ForegroundColor Yellow
}
