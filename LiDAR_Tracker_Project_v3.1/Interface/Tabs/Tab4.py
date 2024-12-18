import pandas as pd
from tkinter import filedialog
from tkinter import StringVar, ttk

def build_tab4(tab, config, processor):
    """Builds the PCAP Clipping tab."""
    ttk.Label(tab, text="PCAP Clipping").grid(column=0, row=0, padx=10, pady=10)

    # Variables
    csv_file = StringVar(value=config.get_param("tab4")["time_reference_file_path"])
    
    table_data = []  # Store table rows for time_ref

    # Treeview for displaying time_ref
    tree = ttk.Treeview(tab, columns=("pcap_name", "frame_index"), show="headings")
    tree.heading("pcap_name", text="PCAP Name")
    tree.heading("frame_index", text="Frame Index")
    tree.grid(column=0, row=2, columnspan=4, sticky="nsew")

    # Add scrollbar to the treeview
    scrollbar = ttk.Scrollbar(tab, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.grid(column=4, row=2, sticky="ns")

    # File selection for time_ref
    ttk.Label(tab, text="Time Reference CSV").grid(column=0, row=1, padx=10, pady=10)
    ttk.Entry(tab, textvariable=csv_file, width=50).grid(column=1, row=1)
    # Add "Create Empty CSV" button
    ttk.Button(
        tab,
        text="Create Empty CSV",
        command=lambda: create_empty_csv(csv_file, tree, table_data, config)
    ).grid(column=3, row=1)

    ttk.Button(
        tab,
        text="Select Time Reference (CSV) File",
        command=lambda: select_csv_file(csv_file, tree, table_data, config)
    ).grid(column=2, row=1)

    # Manual entry fields
    ttk.Label(tab, text="PCAP Name").grid(column=0, row=3, padx=10, pady=10)
    pcap_name = StringVar()
    ttk.Entry(tab, textvariable=pcap_name, width=20).grid(column=1, row=3, padx=10, pady=10)

    ttk.Label(tab, text="Frame Index").grid(column=2, row=3, padx=10, pady=10)
    frame_index = StringVar()
    ttk.Entry(tab, textvariable=frame_index, width=10).grid(column=3, row=3, padx=10, pady=10)

    # Buttons for table manipulation
    ttk.Button(
        tab,
        text="Add Row",
        command=lambda: add_row(tree, table_data, pcap_name, frame_index)
    ).grid(column=0, row=4, padx=10, pady=10)

    ttk.Button(
        tab,
        text="Edit Row",
        command=lambda: edit_row(tree, table_data, pcap_name, frame_index)
    ).grid(column=1, row=4, padx=10, pady=10)

    ttk.Button(
        tab,
        text="Delete Row",
        command=lambda: delete_row(tree, table_data)
    ).grid(column=2, row=4, padx=10, pady=10)

    # Export updated CSV
    ttk.Button(
        tab,
        text="Export to CSV",
        command=lambda: export_csv(table_data)
    ).grid(column=3, row=4, padx=10, pady=10)

    # Generate PCAP Clips
    ttk.Button(
        tab,
        text="Generate Clips",
        command=lambda: processor.start_pcap_clipping(table_data)
    ).grid(column=0, row=5, columnspan=4, padx=10, pady=10)


# Helper Functions
def select_csv_file(csv_var, tree, table_data, config):
    """Select a CSV file and populate the table."""
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        csv_var.set(file_path)
        config.set_param("tab4", "time_reference_file", file_path)
        populate_treeview(file_path, tree, table_data)


def populate_treeview(file_path, tree, table_data):
    """Populate the treeview with data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        table_data.clear()
        table_data.extend(df.to_dict(orient="records"))
        update_treeview(tree, table_data)
    except Exception as e:
        print(f"Error reading CSV: {e}")


def update_treeview(tree, data):
    """Update the treeview with the provided data."""
    for row in tree.get_children():
        tree.delete(row)
    for item in data:
        tree.insert("", "end", values=(item["pcap_name"], item["frame_index"]))


def add_row(tree, table_data, pcap_name_var, frame_index_var):
    """Add a new row to the treeview."""
    pcap_name = pcap_name_var.get()
    frame_index = frame_index_var.get()
    if pcap_name and frame_index:
        new_row = {"pcap_name": pcap_name, "frame_index": frame_index}
        table_data.append(new_row)
        update_treeview(tree, table_data)


def edit_row(tree, table_data, pcap_name_var, frame_index_var):
    """Edit the selected row in the treeview."""
    selected_item = tree.selection()
    if selected_item:
        selected_index = tree.index(selected_item[0])
        table_data[selected_index] = {
            "pcap_name": pcap_name_var.get(),
            "frame_index": frame_index_var.get(),
        }
        update_treeview(tree, table_data)


def delete_row(tree, table_data):
    """Delete the selected row from the treeview."""
    selected_item = tree.selection()
    if selected_item:
        selected_index = tree.index(selected_item[0])
        del table_data[selected_index]
        update_treeview(tree, table_data)


def export_csv(data):
    """Export the updated table data to a CSV file."""
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        print(f"Exported data to {file_path}")

# Helper Functions
def create_empty_csv(csv_var, tree, table_data, config):
    """Create an empty CSV file and prepare it for editing."""
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        # Create an empty DataFrame with the required columns
        df = pd.DataFrame(columns=["pcap_name", "frame_index"])
        df.to_csv(file_path, index=False)
        print(f"Created empty CSV at {file_path}")

        # Set the file path in the entry field and config
        csv_var.set(file_path)
        config.set_param("tab4", "time_reference_file", file_path)

        # Clear and refresh the table for editing
        table_data.clear()
        update_treeview(tree, table_data)