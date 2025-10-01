# Drec-DreamRecorder

This repository was originally based on dreamento (accessed from https://github.com/dreamento/dreamento, mid 2024). 
It has developed into a standalone project for **eeg recording** of the signal captured by a ZMax Headband (by Hypnodyne) and **scoring** the eeg signal using the **yasa** library. Additionally the option to send the scoring to a separate **webhook** is implemented, to allow to control external (audio, visual, ...) impulses or applications.

## requirements
- python 3.8
- libraries:
  - see requirements.txt

 We did not specifically 'opt for' these versions, they were just the current ones when implementing. May work with future versions of these libraries.

## run from scripts
all the following steps describe the procedure for the windows operating system. On other os the commands may differ slightly.

1. clone the repository

2. create a virtual env (in cmd):
```python -m venv /path/to/wherever/you/want/it/to/live/```.
if multiple python versions are installed a specific can be used by typing
```python -3.11```

4. activate the venv:
run the activate.bat file from your cmd located at /path/to/venv/Scripts/activate.bat
  
5. make sure pip is upgraded:
```python -m pip install --upgrade pip```
6. install the requirements:
- from requirements.txt located in the repository or
- install the required ones manually 
```python -m pip install -r requirements.txt```
or
```python -m pip install <package==version>```
6. run mainconsole.py from your cmd that has the venv activated
```
python mainconsole.py
```

## create .exe from scripts
1. activate the virtualenv
2. make sure the requirements are fulfilled: ```python -m pip install -r requirements.txt```
4. navigate to /Drec-DreamRecorder/source_code/
5. run following command:
```pyinstaller --onefile --collect-all mne --collect-all lazy_loader --collect-data lspopt mainconsole.py```
6. a folder called 'dist' will be created, containing one file called 'mainconsole.exe'
7. you can move mainconsole.exe wherever you want and do not need the python environment to be running

## usage
make sure **HDServer** from hypnodyne is running. 

