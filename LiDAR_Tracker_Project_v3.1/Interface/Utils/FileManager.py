import os
from tkinter import filedialog

class FileManager:
    @staticmethod
    def select_file(file_types, title="Select File"):
        print("Selecting file")
        return filedialog.askopenfilename(filetypes=file_types, title=title)

    @staticmethod
    def select_folder(title="Select Folder"):
        print("Selecting folder")
        return filedialog.askdirectory(title=title)