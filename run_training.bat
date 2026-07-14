@echo off
cd /d "C:\Users\Adaptive Network\Documents\Lung Cancer\lung-nodule-fusion-xai"
".venv\Scripts\python.exe" _run_all_training.py 2>> _orchestrator_err.log
