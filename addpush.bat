@echo off
cd /d "%~dp0"
git add -A
git status
git commit -m "Update: %date% %time:~0,5%"
git push origin main
pause
