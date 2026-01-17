import tkinter as tk
import json
from pathlib import Path
from typing import Dict, Callable
from PIL import ImageTk
from .styles import COLORS, FONTS, PADDING_INNER

class ProfessionSelectorPopup(tk.Toplevel):
    """Popup window for selecting a profession."""
    
    def __init__(self, parent, icons: Dict[str, ImageTk.PhotoImage], callback: Callable[[str], None]):
        super().__init__(parent)
        self.title("Select Profession")
        self.configure(bg=COLORS['bg_main'])
        self.resizable(False, False)
        self.callback = callback
        self.transient(parent) # Output on top of parent
        self.grab_set() # Modal
        
        container = tk.Frame(self, bg=COLORS['bg_main'], padx=10, pady=10)
        container.pack(fill=tk.BOTH, expand=True)

        # Load professions structure
        json_path = Path(__file__).parents[2] / 'data' / 'professions.json'
        ordered_professions = []
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                ordered_professions = data.get('professions', [])
        except Exception as e:
            print(f"Error loading professions.json: {e}")

        placed_icons = set()
        current_row = 0

        def create_btn(name, icon, r, c):
            btn_frame = tk.Frame(container, bg=COLORS['bg_main'])
            btn_frame.grid(row=r, column=c, padx=2, pady=2)
            
            btn = tk.Button(
                btn_frame, 
                image=icon, 
                command=lambda n=name: self._select(n),
                bg=COLORS['bg_main'],
                activebackground=COLORS['bg_header'],
                bd=1,
                relief=tk.FLAT
            )
            btn.pack()
            
            # Tooltip logic could go here
            # Hovering logic for name display could be added

        # 1. Place organized professions from JSON
        for prof_entry in ordered_professions:
            col = 0
            
            # Core profession
            base_name = prof_entry['name']
            if base_name in icons:
                create_btn(base_name, icons[base_name], current_row, col)
                placed_icons.add(base_name)
            
            col += 1
            
            # Elite specializations
            for spec_name in prof_entry.get('elite_specializations', []):
                if spec_name in icons:
                    create_btn(spec_name, icons[spec_name], current_row, col)
                    placed_icons.add(spec_name)
                col += 1
            
            current_row += 1

        # 2. Place any remaining icons that weren't in the JSON (fallback)
        # Sort remaining alphabetically
        remaining_keys = sorted([k for k in icons.keys() if k not in placed_icons and k != 'Unknown'])
        
        if remaining_keys:
            # Add a separator or gap if we have extra icons
            if placed_icons:
                current_row += 1
                
            col = 0
            max_cols = 6
            for name in remaining_keys:
                create_btn(name, icons[name], current_row, col)
                col += 1
                if col >= max_cols:
                    col = 0
                    current_row += 1

        # Center popup on parent
        self.update_idletasks()
        try:
            x = parent.winfo_rootx() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
            y = parent.winfo_rooty() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
            self.geometry(f"+{x}+{y}")
        except:
             # Fallback if parent not fully realized
             pass

    def _select(self, profession):
        self.callback(profession)
        self.destroy()
