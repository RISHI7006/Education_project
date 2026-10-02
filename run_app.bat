@echo off
cd /d "%~dp0"
echo Starting EduPro Academy Dashboard...
python -m streamlit run app.py
pause
