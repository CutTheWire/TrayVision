import time
import tkinter as tk
import tkinter.ttk as ttk

tuple_1 = tuple(range(1, 25))

def progress_bar_func(style, progress_bar, sequence):
    root.after(500, update_progress_bar, style, progress_bar, 1, len(sequence))

def update_progress_bar(style, progress_bar, num, limit):
    if num <= limit:
        percentage = round(num/limit * 100)  # Calculate percentage.
        progress_bar.config(value=num)
        style.configure('text.Horizontal.TProgressbar', text='{:g} %'.format(percentage))
        num += 1
        root.after(500, update_progress_bar, style, progress_bar, num, limit)

root = tk.Tk()
root.geometry("300x300")

style = ttk.Style(root)
style.layout('text.Horizontal.TProgressbar',
             [('Horizontal.Progressbar.trough',
               {'children': [('Horizontal.Progressbar.pbar',
                              {'side': 'left', 'sticky': 'ns'})],
                'sticky': 'nswe'}),
              ('Horizontal.Progressbar.label', {'sticky': ''})])
style.configure('text.Horizontal.TProgressbar', text='0 %')

progress_bar = ttk.Progressbar(root, style='text.Horizontal.TProgressbar', length=200,
                               maximum=len(tuple_1), value=0)
progress_bar.pack()

progress_button = tk.Button(root, text="start",
                            command=lambda: progress_bar_func(style, progress_bar, tuple_1))
progress_button.pack()

root.mainloop()