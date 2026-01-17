import tkinter as tk
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
        
        # Calculate grid
        # We have ~9 professions * 3 weights + new ones... lots of professions.
        # Let's do a scrollable area or just a big grid. 
        # Using a simple frame grid for now.
        
        container = tk.Frame(self, bg=COLORS['bg_main'], padx=10, pady=10)
        container.pack(fill=tk.BOTH, expand=True)

        row = 0
        col = 0
        max_cols = 6 # 6 columns
        
        # Sort professions alphabetically
        sorted_profs = sorted([k for k in icons.keys() if k != 'Unknown'])
        
        for name in sorted_profs:
            icon = icons[name]
            
            # Button frame for hover effect
            btn_frame = tk.Frame(container, bg=COLORS['bg_main'])
            btn_frame.grid(row=row, column=col, padx=2, pady=2)
            
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
            
            # Tooltip-ish label
            # tk.Label(btn_frame, text=name, font=FONTS['small'], bg=COLORS['bg_main'], fg=COLORS['text_secondary']).pack()
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
                
        # Center popup on parent
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_rooty() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

    def _select(self, profession):
        self.callback(profession)
        self.destroy()
