from tkinter import filedialog
from tkinter.messagebox import showinfo
def update_flag(flag):
    pass

def select_file(tab, param, string_var, config, file_types):
    """Handles file selection."""
    file_path = filedialog.askopenfilename(filetypes=file_types)
    if file_path:
        string_var.set(file_path)
        config.set_param(tab, param, file_path)
        # showinfo("File Selected", f"Selected file: {file_path}")


def select_folder(tab, param, string_var, config):
    """Handles folder selection."""
    folder_path = filedialog.askdirectory()
    if folder_path:
        string_var.set(folder_path)
        config.set_param(tab, param, folder_path)
        # showinfo("Folder Selected", f"Selected folder: {folder_path}")