from Interface_builder import *
def build_tab3(tab, config, processor):
    """Builds the Geometry Referencing tab."""
    ttk.Label(tab, text="Geometry Referencing").grid(column=0, row=0, padx=10, pady=10)

    traj_folder = StringVar(value=config.get_param("tab3")["traj_folder"])
    output_folder = StringVar(value=config.get_param("tab3")["output_folder"])
    ref_xyz_file = StringVar(value=config.get_param("tab3")["ref_xyz_file_path"])
    ref_llh_file = StringVar(value=config.get_param("tab3")["ref_llh_file_path"])

    ttk.Label(tab, text="Trajectory Folder").grid(column=0, row=1)
    ttk.Entry(tab, textvariable=traj_folder, width=50).grid(column=1, row=1)
    ttk.Button(
        tab,
        text="Select Folder",
        command=lambda: select_folder("tab3", "traj_folder", traj_folder, config)
    ).grid(column=2, row=1)

    ttk.Label(tab, text="Output Folder").grid(column=0, row=2)
    ttk.Entry(tab, textvariable=output_folder, width=50).grid(column=1, row=2)
    ttk.Button(
        tab,
        text="Select Folder",
        command=lambda: select_folder("tab3", "output_folder", output_folder, config)
    ).grid(column=2, row=2)

    ttk.Label(tab, text="Reference XYZ File").grid(column=0, row=3)
    ttk.Entry(tab, textvariable=ref_xyz_file, width=50).grid(column=1, row=3)
    ttk.Button(
        tab,
        text="Select File",
        command=lambda: select_file("tab3", "ref_xyz_file_path", ref_xyz_file, config, [("CSV files", "*.csv")])
    ).grid(column=2, row=3)

    ttk.Label(tab, text="Reference LLH File").grid(column=0, row=4)
    ttk.Entry(tab, textvariable=ref_llh_file, width=50).grid(column=1, row=4)
    ttk.Button(
        tab,
        text="Select File",
        command=lambda: select_file("tab3", "ref_llh_file_path", ref_llh_file, config, [("CSV files", "*.csv")])
    ).grid(column=2, row=4)

    ttk.Button(
        tab,
        text="Start Geometry Referencing",
        command=lambda: processor.start_geometry_referencing(
            traj_folder.get(), output_folder.get(), ref_xyz_file.get(), ref_llh_file.get()
        )
    ).grid(column=0, row=5, padx=10, pady=10)