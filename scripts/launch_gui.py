"""Launch GUI helper script."""
import sys
import os

if __name__ == "__main__":
    # Ensure src/ is on PYTHONPATH
    root = os.path.dirname(os.path.dirname(__file__))
    src = os.path.join(root, 'src')
    sys.path.insert(0, src)

    from ui.gui_app import run_gui

    run_gui()
