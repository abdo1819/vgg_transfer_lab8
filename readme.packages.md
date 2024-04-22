python -m venv .venv



(venv) [...]$ mkdir pkgs
(venv) [...]$ cd pkgs
(venv) [...]$ pip freeze > requirements.txt
(venv) [...]$ pip download -r requirements.txt



(venv) [...]$ cd pkgs
# --- unarchive pip.tar.gz ---
(venv) [...]$ python setup.py install
(venv) [...]$ pip install --no-index --find-links . -r requirements.txt
