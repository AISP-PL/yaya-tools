import argparse
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtWidgets

# Use Qt5Agg backend for Matplotlib
matplotlib.use("Qt5Agg")


# --------------------------
# Log Parsing Functions
# --------------------------
def parse_log_text(log_text):
    """
    Parses a Darknet log text and extracts per-class metrics.
    Expected log lines (example):
      class_id = 0, name = a1.rowery, ap = 98.44%   	 (TP = 814, FP = 65)
    Returns a dictionary with key 'classes' (a sorted list by class_id).
    """
    data = {"classes": []}
    pattern = (
        r"class_id\s*=\s*(\d+),\s*name\s*=\s*([\w\.\-]+),\s*ap\s*=\s*([\d\.]+)%\s*"
        r"\(TP\s*=\s*(\d+),\s*FP\s*=\s*(\d+)\)"
    )
    for line in log_text.splitlines():
        m = re.search(pattern, line)
        if m:
            class_id = int(m.group(1))
            name = m.group(2)
            ap = float(m.group(3))
            tp = int(m.group(4))
            fp = int(m.group(5))
            data["classes"].append({"class_id": class_id, "name": name, "ap": ap, "TP": tp, "FP": fp})
    data["classes"].sort(key=lambda x: x["class_id"])
    return data


def load_log_file(file_path):
    """
    Loads a log file from the given path and parses it.
    """
    with open(file_path, "r") as f:
        text = f.read()
    return parse_log_text(text)


# --------------------------
# Figure Creation Functions
# --------------------------
def create_confusion_heatmap_figure(data, title):
    """
    Creates a heatmap (using seaborn) of per-class metrics (AP, TP, FP).
    The DataFrame rows are labeled "ID: name".
    """
    # Build a dictionary where each key is "ID: name" and values are metrics.
    rows = {}
    for item in data["classes"]:
        key = f"{item['class_id']}: {item['name']}"
        rows[key] = {"AP (%)": item["ap"], "TP": item["TP"], "FP": item["FP"]}
    df = pd.DataFrame.from_dict(rows, orient="index")

    # Determine figure height based on the number of rows.
    num_rows = len(df)
    fig, ax = plt.subplots(figsize=(8, num_rows * 0.8 + 2))

    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title(title)
    return fig


def create_comparison_matrix_heatmap_figure(data1, data2, title):
    """
    Creates a comparison heatmap for the AP differences.
    The DataFrame index is "ID: name" and the single column "Diff (%)" contains the difference.
    Custom annotations show:
      AP Log1
      AP Log2
      Diff (with +/- sign)
    """
    rows = {}
    for d1, d2 in zip(data1["classes"], data2["classes"]):
        key = f"{d1['class_id']}: {d1['name']}"
        diff = d1["ap"] - d2["ap"]
        rows[key] = {"AP Log1 (%)": d1["ap"], "AP Log2 (%)": d2["ap"], "Diff (%)": diff}
    df = pd.DataFrame.from_dict(rows, orient="index")

    # Create a DataFrame with only the difference column for the heatmap.
    diff_df = df[["Diff (%)"]]

    # Create custom annotations: "AP1\nAP2\nDiff"
    annot = df.apply(
        lambda row: f"{row['AP Log1 (%)']:.2f}\n{row['AP Log2 (%)']:.2f}\n"
        f"{'+' if row['Diff (%)']>=0 else ''}{row['Diff (%)']:.2f}",
        axis=1,
    )
    # Turn into a DataFrame (one column) matching diff_df
    annot = pd.DataFrame(annot, index=diff_df.index, columns=diff_df.columns)

    num_rows = len(diff_df)
    fig, ax = plt.subplots(figsize=(4, num_rows * 0.8 + 2))
    sns.heatmap(diff_df, annot=annot, fmt="", cmap="coolwarm", center=0, cbar=True, ax=ax)
    ax.set_title(title)
    return fig


# --------------------------
# Main Window with Menu & Figures
# --------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, log1_path="", log2_path=""):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Darknet Log Comparison")
        self.resize(900, 800)

        # Data holders for the two logs (None if not loaded)
        self.data1 = None
        self.data2 = None

        # Try loading logs from provided paths (if any)
        if log1_path and os.path.exists(log1_path):
            try:
                self.data1 = load_log_file(log1_path)
            except Exception as e:
                print(f"Error loading log1: {e}")
        if log2_path and os.path.exists(log2_path):
            try:
                self.data2 = load_log_file(log2_path)
            except Exception as e:
                print(f"Error loading log2: {e}")

        # Create the menu
        self.create_menu()

        # Create a scroll area and a container widget
        scroll_area = QtWidgets.QScrollArea()
        self.container = QtWidgets.QWidget()
        scroll_area.setWidget(self.container)
        scroll_area.setWidgetResizable(True)
        self.setCentralWidget(scroll_area)

        # Create a vertical layout in the container for figures
        self.figure_layout = QtWidgets.QVBoxLayout(self.container)

        # Initial update of figures
        self.update_figures()

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_log1_action = QtWidgets.QAction("Open Log 1", self)
        open_log1_action.triggered.connect(self.open_log1)
        file_menu.addAction(open_log1_action)

        open_log2_action = QtWidgets.QAction("Open Log 2", self)
        open_log2_action.triggered.connect(self.open_log2)
        file_menu.addAction(open_log2_action)

    def open_log1(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Log 1", "", "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            try:
                self.data1 = load_log_file(file_path)
                self.update_figures()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Could not load Log 1:\n{e}")

    def open_log2(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Log 2", "", "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            try:
                self.data2 = load_log_file(file_path)
                self.update_figures()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Could not load Log 2:\n{e}")

    def clear_layout(self, layout):
        """Remove all widgets from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def update_figures(self):
        """
        Rebuilds the figure canvases and adds them to the scrollable layout.
        Shows:
         - Confusion matrix for Log 1 (if available)
         - Confusion matrix for Log 2 (if available)
         - Comparison matrix (if both logs are loaded)
        """
        self.clear_layout(self.figure_layout)

        if self.data1 is not None:
            fig1 = create_confusion_heatmap_figure(self.data1, "Confusion Matrix - Log 1")
            canvas1 = FigureCanvas(fig1)
            self.figure_layout.addWidget(canvas1)
        if self.data2 is not None:
            fig2 = create_confusion_heatmap_figure(self.data2, "Confusion Matrix - Log 2")
            canvas2 = FigureCanvas(fig2)
            self.figure_layout.addWidget(canvas2)
        if self.data1 is not None and self.data2 is not None:
            fig3 = create_comparison_matrix_heatmap_figure(self.data1, self.data2, "Comparison Matrix (AP Difference)")
            canvas3 = FigureCanvas(fig3)
            self.figure_layout.addWidget(canvas3)
        # Add a spacer at the end so figures are top-aligned.
        self.figure_layout.addStretch()


# --------------------------
# Main Routine with Argparse
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare two Darknet logs and show previews in a Qt5 window.")
    parser.add_argument("--log1", type=str, default="", help="Path to log 1 file")
    parser.add_argument("--log2", type=str, default="", help="Path to log 2 file")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(log1_path=args.log1, log2_path=args.log2)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
