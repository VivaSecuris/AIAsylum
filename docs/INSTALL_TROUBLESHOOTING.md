# Installation troubleshooting

Common installation issues and solutions. The supported path is `./install.sh` on Python 3.10, 3.11, or 3.12.

## Python version

`setup.py` requires `>=3.10,<3.13`. Python 3.13 and newer cannot build the pinned `pydantic-core` wheels.

```bash
python3 --version
```

macOS:

```bash
brew install python@3.11
python3.11 -m venv venv
source venv/bin/activate
```

Debian/Ubuntu:

```bash
sudo apt-get install python3.11 python3.11-venv
python3.11 -m venv venv
source venv/bin/activate
```

pyenv:

```bash
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv venv
source venv/bin/activate
```

Then re-run `./install.sh`. Check with `python --version` (3.10, 3.11, or 3.12) and `venv/bin/python scripts/quick_test.py`.

## psycopg2-binary installation error

```
Error: pg_config executable not found.
pg_config is required to build psycopg2 from source.
```

SQLite is the default and needs no extra system packages. `./install.sh` does not install PostgreSQL drivers unless you pass `--postgres`.

If you do want PostgreSQL:

macOS: `brew install postgresql`, then `./install.sh --postgres`.

Ubuntu/Debian: `sudo apt-get install libpq-dev`, then `./install.sh --postgres`.

Windows: install PostgreSQL from https://www.postgresql.org/download/windows/ so `pg_config` is on `PATH`, then `pip install -r requirements-postgres.txt` inside the virtual environment.

## ModuleNotFoundError or command not found

The virtual environment has to be the interpreter you are using:

```bash
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows
which python               # should be venv/bin/python
```

Prefer `venv/bin/python` and `venv/bin/aiasylum` without activating. `./install.sh` installs the package with `pip install -e .`; installing `requirements.txt` alone does not put `vivasecuris` on the path.

## Permission errors

Do not use `sudo` with pip inside the virtual environment. Activate `venv` (or call `venv/bin/python -m pip`) and install again.

## Missing system build tools

macOS: `xcode-select --install`

Ubuntu/Debian: `sudo apt-get install build-essential python3-dev`

## Verify

```bash
venv/bin/python scripts/quick_test.py
venv/bin/python -m pytest tests/ -q
```

More detail: [../README.md](../README.md), [TESTING.md](TESTING.md).
