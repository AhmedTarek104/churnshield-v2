@echo off
set PY=python3.13
echo === Step 1: Ingest ===
%PY% src/ingest.py
if errorlevel 1 goto error

echo === Step 2: Features ===
%PY% src/features.py
if errorlevel 1 goto error

echo === Step 3: Preprocess ===
%PY% src/preprocess.py
if errorlevel 1 goto error

echo === Step 4: Train ===
%PY% src/train.py
if errorlevel 1 goto error

echo === Step 5: Evaluate ===
%PY% src/evaluate.py
if errorlevel 1 goto error

echo === Step 6: Explain ===
%PY% src/explain.py
if errorlevel 1 goto error

echo === PIPELINE COMPLETE ===
goto end

:error
echo PIPELINE FAILED
:end
