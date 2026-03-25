Write-Host "正在检查 Cloudflared..."
if (-not (Test-Path "cloudflared-windows-amd64.exe")) {
    Write-Host "正在下载 Cloudflared (可能需要几分钟)..."
    try {
        # 尝试使用 Invoke-WebRequest 下载，设置 TLS 1.2 以防旧系统问题
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile "cloudflared-windows-amd64.exe"
    } catch {
        Write-Error "自动下载失败: $_"
        Write-Host "请手动下载并放入此目录: https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        exit
    }
}

Write-Host "Cloudflared 准备就绪。"
Write-Host "正在启动外网映射... Streamlit 应在后台运行中。"
Write-Host "-----------------------------------------------------"
Write-Host "请等待下方出现 'https://....trycloudflare.com' 的链接"
Write-Host "复制该链接即可在任何设备访问本应用"
Write-Host "-----------------------------------------------------"

# 启动 tunnel
.\cloudflared-windows-amd64.exe tunnel --url http://localhost:8501
