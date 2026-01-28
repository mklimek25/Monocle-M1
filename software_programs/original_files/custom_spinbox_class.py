import tkinter as tk
from tkinter import font

class CustomSpinbox(tk.Frame):
    def __init__(self, parent, var, options, callback_var,
                 fontsize=20, spinbox_width=25, button_width=15,  *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.var = var  # Now a plain string, not a tk.StringVar
        self.options = options
        self.callback_var = callback_var
        self.index = 0  # Keep track of the current selection index if options is a list

        # Determine whether `options` is a list or a float
        if isinstance(options, list):
            self.is_float = False
        elif isinstance(options, (int, float)):
            self.is_float = True
            self.options = float(options)  # Ensure it's stored as a float
        else:
            raise ValueError("Options must be either a list or a float.")

        # Create a custom font for the Entry
        custom_font = font.Font(family="Helvetica", size=fontsize)

        # Create an Entry to display the selected option
        self.entry = tk.Entry(self, font=custom_font, width=spinbox_width, justify='center', state='readonly')
        self.entry.grid(row=0, column=1, sticky='nsew')

        # Create the up arrow button with custom size
        self.up_button = tk.Button(self, text=">", width=button_width, command=self.increment)
        self.up_button.grid(row=0, column=2, sticky='nsew')

        # Create the down arrow button with custom size
        self.down_button = tk.Button(self, text="<", width=button_width, command=self.decrement)
        self.down_button.grid(row=0, column=0, sticky='nsew')

        # Set the initial value
        self.update_entry()

    def increment(self):
        """Move to the next option or increment the float value."""
        if self.is_float:
            self.options += 0.1  # Increment float by 0.1
        else:
            self.index = (self.index + 1) % len(self.options)
        self.update_entry()

    def decrement(self):
        """Move to the previous option or decrement the float value."""
        if self.is_float:
            self.options -= 0.1  # Decrement float by 0.01
        else:
            self.index = (self.index - 1) % len(self.options)
        self.update_entry()

    def update_entry(self):
        """Update the displayed value in the entry."""
        if self.is_float:
            # Round to two decimal places for float display
            self.var = f"{self.options:.2f}"
        else:
            self.var = self.options[self.index]

        if self.callback_var is not None:
            self.callback_var(self.var)
        # Update the entry widget
        self.entry.config(state='normal')
        self.entry.delete(0, tk.END)
        self.entry.insert(0, self.var)
        self.entry.config(state='readonly')