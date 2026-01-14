---
applyTo: "**/*.py"
---

# Python Project Coding Standards

## 1. Environment & Execution
- **Virtual Environment:** Always ensure the virtual environment at the project root is active before executing any code. 
- **Execution Rule:** If a command fails, analyze the error, resolve the dependency or logic issue, and iterate until the code runs successfully.

## 2. Project Structure
- **Folder Layout:** Maintain a strict `src/` layout. Place all application logic inside `src/<package_name>/` and never in the root directory.
- **Organization:** Use an intelligent folder structure (e.g., `src/models/`, `src/utils/`, `src/api/`) and ensure `__init__.py` files are present to maintain modularity.

## 3. Testing Requirements
- **Mandatory Tests:** For every new feature or bug fix in `src/`, you must create a corresponding test file in the `tests/` directory.
- **Test Execution:** Always run `pytest` at the root and verify all tests pass before considering a task complete.
- **Test Style:** Use `pytest` conventions. Tests should be isolated, descriptive, and focus on core functionality.

## 4. Development Workflow
- **Validation:** Ensure all code follows PEP 8 standards and includes proper type hints.
- **Documentation:** Include docstrings for all new functions and classes.
