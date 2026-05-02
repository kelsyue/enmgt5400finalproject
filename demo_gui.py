"""
Interactive Dashboard for CERT Insider Threat Detection.

This module provides a GUI-based visualization tool to explore the results of the
LSTM Autoencoder model. It allows users to scan the test population, adjust
detection sensitivity, and investigate individual users' behavioral anomalies.

Features:
- Population Grid: Visual representation of all users in the test set.
- Scanning Animation: Simulates a system-wide threat scan.
- Sensitivity Control: Real-time threshold adjustment and color-coded results.
- Detailed Investigation: Per-user breakdown of feature-level reconstruction errors.
- Model Metrics: Quick access to ROC, training curves, and score distributions.
"""

import os
import time
import pickle
import threading
import numpy as np
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle
from PIL import Image

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# --- Theme & Colors ---
plt.style.use('dark_background')
COLORS = {
    "bg": "#0A0A0A",
    "unscanned": "#222222",
    "safe": "#00C853",    # Vibrant Green
    "danger": "#FF1744",  # Vibrant Red
    "accent": "#AA00FF",  # Deep Purple
    "text": "#FFFFFF",
    "highlight": "#FFEA00" # Yellow for finding malicious
}

class Dashboard:
    """
    Main Dashboard application class for visualizing threat detection results.
    
    This class manages the Matplotlib-based UI, handles user interactions,
    and performs on-the-fly inference for detailed user investigation.
    """
    def __init__(self):
        """Initializes the dashboard, loads models/data, and sets up the UI layout."""
        print("Loading model and data... Please wait.")
        # Load Data
        with open('processed/feature_cols.pkl', 'rb') as f: self.feature_cols = pickle.load(f)
        with open('processed/malicious_users.pkl', 'rb') as f: self.malicious_users = pickle.load(f)
        self.u_test = np.load('processed/u_test.npy', allow_pickle=True)
        self.X_test = np.load('processed/X_test.npy')
        self.model = tf.keras.models.load_model('models/best_model.keras')
        self.default_threshold = float(np.load('models/threshold.npy'))
        self.threshold = self.default_threshold
        
        # Pre-calculate scores
        print("Running inference...")
        preds = self.model.predict(self.X_test, batch_size=512, verbose=0)
        all_errors = np.mean(np.power(self.X_test - preds, 2), axis=(1, 2))
        
        self.user_data = {}
        unique_users = sorted(np.unique(self.u_test))
        for user in unique_users:
            idx = np.where(self.u_test == user)[0]
            scores = all_errors[idx]
            max_idx = np.argmax(scores)
            self.user_data[user] = {
                "max_score": float(scores[max_idx]),
                "is_malicious": user in self.malicious_users,
                "max_idx": int(idx[max_idx]),
                "status": "unscanned"
            }
        
        self.users = unique_users
        self.selected_user = None
        self.is_scanning = False
        self.active_tab = "Details"

        # --- Setup Figure (1270x780 @ 100dpi) ---
        self.fig = plt.figure(figsize=(12.7, 7.8), dpi=100)
        self.fig.canvas.manager.set_window_title('CERT Insider Threat Detection Dashboard')
        self.fig.patch.set_facecolor(COLORS["bg"])
        
        # Layout Definition
        self.gs = self.fig.add_gridspec(3, 2, height_ratios=[0.12, 0.78, 0.1], width_ratios=[1, 1],
                                        left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3)
        
        self.ax_header = self.fig.add_subplot(self.gs[0, :])
        self.ax_grid = self.fig.add_subplot(self.gs[1, 0])
        self.ax_content = self.fig.add_subplot(self.gs[1, 1])
        self.ax_footer = self.fig.add_subplot(self.gs[2, :])
        
        self.setup_grid()
        self.setup_ui()
        self.update_stats()
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def setup_grid(self):
        """Creates the grid of rectangles representing the user population."""
        self.ax_grid.set_axis_off()
        self.rects = {}
        
        cols = 20
        rows = int(np.ceil(len(self.users) / cols))
        self.ax_grid.set_xlim(-0.5, cols - 0.5)
        self.ax_grid.set_ylim(rows - 0.5, -0.5)
        
        for i, user in enumerate(self.users):
            r, c = i // cols, i % cols
            rect = Rectangle((c-0.4, r-0.4), 0.8, 0.8, color=COLORS["unscanned"], picker=True)
            self.ax_grid.add_patch(rect)
            self.rects[user] = rect

    def setup_ui(self):
        """Configures the control widgets: sensitivity slider, scan button, and tabs."""
        self.ax_header.set_axis_off()
        self.ax_footer.set_axis_off()
        
        # --- Threshold Slider (Bottom Left) ---
        # INVERTED LOGIC: High Slider Value = High Sensitivity = Low Threshold
        ax_slider = self.fig.add_axes([0.15, 0.05, 0.25, 0.02])
        init_sens = 1.5 - self.default_threshold
        self.slider = Slider(ax_slider, 'SENSITIVITY ', 0.0, 1.5, valinit=init_sens, color=COLORS["accent"])
        self.slider.on_changed(self.on_sensitivity_change)
        
        # --- Action Buttons (Bottom Right) ---
        ax_btn_scan = self.fig.add_axes([0.72, 0.04, 0.08, 0.04])
        self.btn_scan = Button(ax_btn_scan, 'SCAN', color="#2C2C2C", hovercolor="#444444")
        self.btn_scan.on_clicked(self.start_scan)

        ax_btn_find = self.fig.add_axes([0.81, 0.04, 0.12, 0.04])
        self.btn_find = Button(ax_btn_find, 'FIND THREATS', color="#2C2C2C", hovercolor="#444444")
        self.btn_find.on_clicked(self.find_threats)

        # --- Tab Buttons (Top of Content Area) ---
        self.tab_buttons = {}
        tabs = ["Details", "ROC", "Curves", "Dist"]
        for i, name in enumerate(tabs):
            ax_tab = self.fig.add_axes([0.55 + (i*0.09), 0.84, 0.08, 0.03])
            btn = Button(ax_tab, name, color=COLORS["accent"] if i==0 else "#2C2C2C", hovercolor="#444444")
            btn.on_clicked(lambda e, n=name: self.switch_tab(n))
            self.tab_buttons[name] = btn

    def update_stats(self):
        """Updates the header text with current population stats and detection results."""
        self.ax_header.clear()
        self.ax_header.set_axis_off()
        
        scanned = sum(1 for u in self.user_data.values() if u["status"] == "scanned")
        flagged = [u for u, d in self.user_data.items() if d["status"] == "scanned" and d["max_score"] > self.threshold]
        tp = sum(1 for u in flagged if self.user_data[u]["is_malicious"])
        fp = len(flagged) - tp
        
        self.ax_header.text(0.0, 0.7, "CERT INSIDER THREAT DETECTION SYSTEM", color=COLORS["accent"], fontsize=16, fontweight='bold')
        stat_text = (f"POPULATION: {len(self.users)} | SCANNED: {scanned} | THRESHOLD: {self.threshold:.3f} | "
                     f"TRUE POSITIVES: {tp}/5 | FALSE POSITIVES: {fp}")
        self.ax_header.text(0.0, 0.2, stat_text, color=COLORS["text"], fontsize=11)
        self.fig.canvas.draw_idle()

    def switch_tab(self, name):
        """Handles switching between different visualization views (Details, ROC, etc.)."""
        self.active_tab = name
        for n, btn in self.tab_buttons.items():
            btn.ax.set_facecolor(COLORS["accent"] if n == name else "#2C2C2C")
        
        if name == "Details":
            if self.selected_user: self.show_details(self.selected_user)
            else: self.clear_content("SELECT A USER FROM THE GRID TO BEGIN INVESTIGATION")
        elif name == "ROC": self.show_image("evaluation.png")
        elif name == "Curves": self.show_image("training_curves.png")
        elif name == "Dist": self.show_image("score_distribution.png")
        self.fig.canvas.draw_idle()

    def show_image(self, path):
        """Displays a static image (e.g., a metric plot) in the content area."""
        self.ax_content.clear()
        self.ax_content.set_axis_off()
        if os.path.exists(path):
            img = Image.open(path)
            self.ax_content.imshow(img)
            self.ax_content.set_title(f"Model Metric: {path}", color=COLORS["accent"], pad=10)
        else:
            self.ax_content.text(0.5, 0.5, f"Missing: {path}", color="red", ha='center')

    def clear_content(self, msg):
        """Clears the content area and displays a message."""
        self.ax_content.clear()
        self.ax_content.set_axis_off()
        self.ax_content.text(0.5, 0.5, msg, color=COLORS["text"], ha='center', fontsize=12)

    def on_sensitivity_change(self, val):
        """Callback for the sensitivity slider to update the detection threshold."""
        # Higher Sensitivity = Lower Threshold
        self.threshold = max(0.001, 1.5 - val)
        self.refresh_colors()
        self.update_stats()
        if self.selected_user and self.active_tab == "Details": self.show_details(self.selected_user)

    def refresh_colors(self):
        """Updates the grid colors based on the current threshold."""
        for user in self.users:
            if self.user_data[user]["status"] == "scanned":
                score = self.user_data[user]["max_score"]
                color = COLORS["danger"] if score > self.threshold else COLORS["safe"]
                self.rects[user].set_color(color)
        self.fig.canvas.draw_idle()

    def start_scan(self, event):
        """Initiates the threaded scanning process."""
        if self.is_scanning: return
        self.is_scanning = True
        threading.Thread(target=self.run_scan, daemon=True).start()

    def run_scan(self):
        """Performs a sequential scan of all users and updates the grid colors."""
        for user in self.users:
            self.user_data[user]["status"] = "scanned"
            score = self.user_data[user]["max_score"]
            color = COLORS["danger"] if score > self.threshold else COLORS["safe"]
            self.rects[user].set_color(color)
            if self.users.index(user) % 20 == 0:
                self.update_stats()
                self.fig.canvas.draw_idle()
            time.sleep(0.005)
        self.is_scanning = False
        self.update_stats()

    def find_threats(self, event):
        """Highlights confirmed malicious users with a bright yellow border."""
        for user, data in self.user_data.items():
            if data["is_malicious"]:
                self.rects[user].set_edgecolor(COLORS["highlight"])
                self.rects[user].set_linewidth(3)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        """Handles mouse clicks on the user grid to select a user for investigation."""
        if event.inaxes != self.ax_grid: return
        c, r = int(round(event.xdata)), int(round(event.ydata))
        cols = 20
        idx = r * cols + c
        if 0 <= idx < len(self.users):
            user = self.users[idx]
            self.selected_user = user
            if self.active_tab != "Details": self.switch_tab("Details")
            else: self.show_details(user)

    def show_details(self, user):
        """Displays feature-level reconstruction error breakdown for the selected user."""
        data = self.user_data[user]
        self.ax_content.clear()
        self.ax_content.set_axis_on()
        
        status = "FLAGGED" if data["max_score"] > self.threshold else "CLEARED"
        title_color = COLORS["danger"] if data["max_score"] > self.threshold else COLORS["safe"]
        
        seq_idx = data["max_idx"]
        actual = self.X_test[seq_idx]
        pred = self.model.predict(np.expand_dims(actual, 0), verbose=0)[0]
        feat_errors = np.mean(np.square(actual - pred), axis=0)
        
        y_pos = np.arange(len(self.feature_cols))
        self.ax_content.barh(y_pos, feat_errors, color=title_color)
        self.ax_content.set_yticks(y_pos)
        self.ax_content.set_yticklabels([f.replace('_', ' ') for f in self.feature_cols], fontsize=9)
        self.ax_content.invert_yaxis()
        
        truth = "CONFIRMED THREAT" if data["is_malicious"] else "BENIGN"
        title = f"USER: {user} | SCORE: {data['max_score']:.4f}\nSTATUS: {status} | TRUTH: {truth}"
        self.ax_content.set_title(title, color=title_color, fontsize=10, fontweight='bold', pad=15)
        self.ax_content.set_xlabel("Reconstruction Error (Behavioral Contribution)", fontsize=9, color=COLORS["text"])
        
        self.ax_content.tick_params(colors=COLORS["text"], labelsize=8)
        for spine in self.ax_content.spines.values(): spine.set_color('#444444')
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    Dashboard()


if __name__ == "__main__":
    Dashboard()
