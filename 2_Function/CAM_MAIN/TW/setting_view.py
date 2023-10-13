import os
import re
import tkinter as tk
from tkinter import messagebox
from tkinter import font

class SettingsWindow:
    def __init__(self, root,) -> None:
        self.root_style = {
            'bg': '#565659'}

        self.text_label_style = {
            'bg': '#343437',}

        self.tk_style = {
            'bg': '#343437',
            'fg': 'white',
            'font': font.Font(family="Helvetica", size=10)}
        
        self.root = root
        self.root.title("부품별 임계값 설정")
        self.root.geometry("600x500")
        self.root.configure(self.root_style)
        self.root.wm_attributes("-topmost", 1)

        # Fix the window size
        self.root.resizable(width=False, height=False)

        self.frame1_1 = tk.Frame(self.root, width=600, height=80)
        self.frame1_2 = tk.Frame(self.root, width=600, height=80)
        self.frame2 = tk.Frame(self.root, width=600, height=260)
        self.frame3 = tk.Frame(self.root, width=600, height=80)

        self.frame1_1.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.frame1_2.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        self.frame2.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        self.frame3.grid(row=4, column=0, padx=10, pady=10, sticky="s")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=10)
        self.root.grid_columnconfigure(0, weight=10)

        # Add text labels to frame1_1
        self.label1 = tk.Label(self.frame1_1, text="ㅤ부ㅤ품ㅤ:ㅤㅤ", font=("Helvetica", 15), fg='white')
        self.label2 = tk.Label(self.frame1_2, text="ㅤ임계값ㅤ:ㅤㅤ", font=("Helvetica", 15), fg='white')

        # Add text boxes to frame1_2
        vcmd = self.root.register(self.validate_input)
        self.entry1 = tk.Entry(self.frame1_1, width=40, font=("Helvetica", 15), fg='white')
        self.entry2 = tk.Entry(self.frame1_2, validate="key", validatecommand=(vcmd, "%P"), width=40, font=("Helvetica", 15), fg='white')
        self.output_list = tk.Listbox(self.frame2, height=50)
        self.scrollbar = tk.Scrollbar(self.output_list, orient=tk.VERTICAL)
        self.xscrollbar = tk.Scrollbar(self.output_list, orient=tk.HORIZONTAL)

        self.apply_button = tk.Button(self.frame3, text="저장", font=("Helvetica", 15), fg='white', width=10, command=self.save_settings)
        self.empty = tk.Label(self.frame3, text="ㅤㅤㅤㅤㅤㅤ")
        self.delete_button =  tk.Button(self.frame3, text="삭제",  font=("Helvetica", 15), fg='white', width=10, command=self.delete_selected_item)

        # Listbox와 Scrollbar 연결
        self.output_list.config(yscrollcommand = self.scrollbar.set)
        self.scrollbar.config(command = self.output_list.yview)
        self.output_list.config(xscrollcommand=self.xscrollbar.set)
        self.xscrollbar.config(command = self.output_list.xview)

        self.entry1.bind("<Control-v>", lambda event: self.on_entry_key(event, self.entry1))
        self.entry2.bind("<Control-v>", lambda event: self.on_entry_key(event, self.entry2))

        self.frame2.configure(self.root_style)
        self.frame1_1.configure(self.root_style)
        self.frame1_2.configure(self.root_style)
        self.frame3.configure(self.root_style)
        
        self.output_list.configure(self.tk_style)
        self.apply_button.configure(self.tk_style)
        self.empty.configure(self.root_style)
        self.delete_button.configure(self.tk_style)
        
        self.label1.configure(self.root_style)
        self.label2.configure(self.root_style)
        self.entry1.configure(self.text_label_style)
        self.entry2.configure(self.text_label_style)
        
        self.label1.pack(side=tk.LEFT, fill=tk.Y)
        self.label2.pack(side=tk.LEFT, fill=tk.Y)
        self.entry1.pack(side=tk.RIGHT, fill=tk.Y)
        self.entry2.pack(side=tk.RIGHT, fill=tk.Y)

        self.output_list.pack(fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.xscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.delete_button.pack(side=tk.RIGHT, fill=tk.Y)
        self.empty.pack(side=tk.RIGHT, fill=tk.Y)
        self.apply_button.pack(side=tk.RIGHT, fill=tk.Y)
        self.load_settings()

    def validate_input(self, P: str) -> bool:
        if P == "" or P == ".":
            return True
        try:
            fP= float(P)
            if fP <= 400:
                return True
            else:
                return False
        except ValueError:
            return False
        
    def on_entry_key(self, event, entry: str) -> str:
        # 사용자가 Ctrl+V(붙여넣기) 키를 누를 때 해당 Entry에 아무 것도 넣지 않도록 처리
        if event.keycode == 86 and (event.state & 4) != 0:
            return "break"  # 붙여넣기 이벤트 중단

    def is_valid_input(self, text: str) -> bool:
        if re.match(r'^[A-Za-z0-9!@#$%^_.-]*$', text):
            return True
        else:
            return False

    def load_settings(self):
        # Get the user's document folder path
        document_folder = os.path.expanduser("~/Documents")
        # Define the file path to the settings file
        file_path = os.path.join(document_folder, "TW", "settings.txt")
        # Check if the settings file exists
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                # Read the lines from the file
                lines = file.readlines()

                # Clear the current contents of the Listbox
                self.output_list.delete(0, tk.END)

                # Populate the Listbox with the loaded settings
                for line in lines:
                    self.output_list.insert(tk.END, line.strip())

    def save_settings(self):
        # Get the part and threshold values from the text boxes
        try:
            part = str(self.entry1.get())
            threshold = float(self.entry2.get())
            
        except ValueError:
            messagebox.showerror("Settings Error", "임계값은 숫자여야 합니다.")
            self.entry1.delete(0, tk.END)
            self.entry2.delete(0, tk.END)
            return
        if part:
            if self.is_valid_input(part):
                threshold = str(threshold)
                # Get the user's document folder path
                document_folder = os.path.expanduser("~/Documents")

                # Create the TW folder if it doesn't exist
                tw_folder = os.path.join(document_folder, "TW")
                if not os.path.exists(tw_folder):
                    os.makedirs(tw_folder)

                # Define the file path to save the settings
                file_path = os.path.join(tw_folder, "settings.txt")

                # Read existing settings from the file (if it exists)
                existing_settings = {}
                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        lines = file.readlines()

                    # Create a list to store valid lines
                    valid_lines = []

                    for line in lines:
                        if ':' in line:
                            key, value = line.strip().split(': ')
                            existing_settings[key] = value
                            valid_lines.append(line)

                    # Overwrite the file with the valid lines
                    with open(file_path, 'w') as file:
                        for line in valid_lines:
                            file.write(line)

                # Update or add the part and threshold values
                existing_settings[part] = threshold

                # Write the updated settings to the text file
                with open(file_path, 'w') as file:
                    for key, value in existing_settings.items():
                        file.write(f"{key}: {value}\n")

                # Optionally, clear the text boxes after saving
                self.entry1.delete(0, tk.END)
                self.entry2.delete(0, tk.END)

                # Provide feedback to the user that the settings are saved
                messagebox.showinfo("Settings Saved", "부품 및 임계값이 저장되었습니다.")
            else:
                messagebox.showerror("Settings Error",
'''제품 입력에 지원하지 않는 문자가 포함되어 있습니다.
영어 또는 숫자 일부 기호만 사용.
ㅤ
기호 ! @ # $ % ^ _ - . 만 입력 가능합니다.
''')
                self.entry1.delete(0, tk.END)
                self.entry2.delete(0, tk.END)
                return
        else:
            messagebox.showerror("Settings Error", "부품 또는 임계값이 입력되지 않았습니다.")
            self.entry1.delete(0, tk.END)
            self.entry2.delete(0, tk.END)
            return
        self.load_settings()

    def delete_selected_item(self):
        # Get the selected item in the Listbox
        selected_index = self.output_list.curselection()

        if not selected_index:
            messagebox.showerror("Delete Error", "삭제할 항목을 선택하세요.")
            return

        # Ask for confirmation
        confirmation = messagebox.askyesno("Delete Confirmation", "선택한 항목을 삭제하시겠습니까?")

        if confirmation:
            # Remove the selected item from the Listbox
            self.output_list.delete(selected_index)

            document_folder = os.path.expanduser("~/Documents")

            # Create the TW folder if it doesn't exist
            tw_folder = os.path.join(document_folder, "TW")
            if not os.path.exists(tw_folder):
                os.makedirs(tw_folder)

            # Define the file path to save the settings
            file_path = os.path.join(tw_folder, "settings.txt")

            # Delete the existing "settings.txt" file if it exists
            if os.path.exists(file_path):
                os.remove(file_path)

            # Get the items from the Listbox
            items = self.output_list.get(0, tk.END)

            # Write the settings to the text file (using 'w' mode to create a new file)
            with open(file_path, 'w') as file:
                for item in items:
                    file.write(f"{item}\n")

            # Provide feedback to the user that the settings are saved
            messagebox.showinfo("Delete Item", "선택한 항목이 삭제되었습니다.")
            self.load_settings()

class main:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Main Window")
        
        # 버튼을 생성하고 버튼 누를 때 설정 창을 열도록 합니다.
        self.open_settings_button = tk.Button(self.root, text="Open Settings", command=self.open_settings_window)
        self.open_settings_button.pack()

        self.root.mainloop()

    def open_settings_window(self):
        S_root = tk.Tk()
        SettingsWindow(S_root)

if __name__ == "__main__":
    main()
    
    # 빈 딕셔너리 생성
    data_dict = {}

    # 파일 경로
    file_path = "C:\\Users\\sjmbe\\Documents\\TW\\settings.txt"

    # 파일 열기
    try:
        with open(file_path, "r") as file:
            for line in file:
                # 각 줄을 ":"를 기준으로 분리
                parts = line.strip().split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = float(parts[1].strip())  # 숫자로 변환
                    data_dict[key] = value

        # 딕셔너리 출력
        print(data_dict)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없음: {file_path}")