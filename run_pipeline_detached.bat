@echo off
cd /d "C:\Users\Adaptive Network\Documents\Lung Cancer\lung-nodule-fusion-xai"
"C:\Users\Adaptive Network\Documents\Lung Cancer\lung-nodule-fusion-xai\.venv\Scripts\python.exe" run_data_pipeline.py > data_pipeline_detached.log 2>&1
echo DONE_EXIT_%ERRORLEVEL% >> data_pipeline_detached.log
