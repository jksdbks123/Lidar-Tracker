
from Tabs.Tab1 import build_tab1
from Tabs.Tab2 import build_tab2
from Tabs.Tab3 import build_tab3
from Tabs.Tab4 import build_tab4
from Tabs.Tab5 import build_tab5
from Tabs.Tab6 import build_tab6


    
def build_interface(tabs, config, processor, visualizer, dummy_processor):
    """
    Build the GUI interface for the tabs.

    Args:
        tabs (dict): Dictionary containing tab frames.
        config (Config): Configuration manager for persistent parameters.
        processor (Processor): Processor instance for operations.
        visualizer (Visualizer): Visualizer instance for visualization.
    """
    build_tab1(tabs["tab1"], config, visualizer,dummy_processor)
    build_tab2(tabs["tab2"], config)
    build_tab3(tabs["tab3"], config)
    build_tab4(tabs["tab4"], config)
    build_tab5(tabs["tab5"], config)
    build_tab6(tabs["tab6"], config)

