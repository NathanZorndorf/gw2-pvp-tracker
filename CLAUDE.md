# Project Name

## Quick Facts
*   **Stack**: Python (specify version, e.g., 3.11), relevant frameworks (e.g., Django, Flask, FastAPI, Pandas)
*   **Windows**: Program and guild wars 2 run in windows. 
*   **Virtual Environment**: `venv` (or `conda`)
*   **Dependency File**: `requirements.txt` (or `pyproject.toml`, `environment.yml`)
*   **Test Command**: `pytest`
*   **Lint Command**: `pylint` or `flake8`

## Key Directories
*   `src/`: Primary application source code
*   `tests/`: Test files and test data
*   `docs/`: Project documentation
*   `notebooks/`: Jupyter notebooks for data analysis (if applicable)

## Code Style
*   Follow [PEP 8](peps.python.org) style guidelines.
*   Use type hints for function signatures and variables.
*   Write clear, descriptive docstrings for all functions and classes (NumPy or Google style preferred).
*   Prefer f-strings for string formatting.

## Critical Rules
*   **MANDATORY VENV USAGE**: Always activate the virtual environment before running any Python commands or installing packages. Use `source venv/bin/activate` (Linux/macOS) or `.\venv\Scripts\activate` (Windows).
*   **CONCURRENT OPERATIONS**: Batch all related Python operations (venv setup, pip installs, tests) into a single message or action to ensure coordination.
*   **TESTING**: All code changes must be accompanied by new or updated tests, and all existing tests must pass before a pull request can be created.
*   **SECURITY**: Prioritize security best practices, including input validation, preventing SQL injection (if using a database), and avoiding sensitive data in the codebase.

