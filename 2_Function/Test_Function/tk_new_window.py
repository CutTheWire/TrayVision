import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk

def new_tk(image):
    new_root = tk.Toplevel()
    new_root.title("new_tk_window")
    label = tk.Label(new_root, image=image)
    label.pack()

root = tk.Tk()

image = Image.open("./WIN_2023.jpg")
image = ImageTk.PhotoImage(image)

tk_button = ttk.Button(root, text="new_window", command=lambda: new_tk(image))
tk_button.pack()

root.title("test_tk_window")
root.mainloop()
