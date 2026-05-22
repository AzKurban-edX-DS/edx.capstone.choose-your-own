py install 3.12
python3.12 --version
python3.12 -m venv .venv

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install tensorflow