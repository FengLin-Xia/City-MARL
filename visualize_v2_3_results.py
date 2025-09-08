import os
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Polygon
import numpy as np
from matplotlib.widgets import Button, Slider
import tkinter as tk
from tkinter import filedialog
import glob

class V2_3Visualizer:
    def __init__(self, output_dir="enhanced_simulation_v2_3_output"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.current_frame = 0
        self.playing = False
        self.fps = 2  # Frames per second
        
        # Load configuration
        self.load_config()
        
        # Get all image files
        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, "month_*.png")))
        self.total_frames = len(self.image_files)
        
        print(f"Found {self.total_frames} frames in {self.images_dir}")
        
        # Setup matplotlib
        self.setup_plot()
        
    def load_config(self):
        """Load colors configuration"""
        try:
            with open("configs/colors.json", "r", encoding="utf-8") as f:
                self.colors = json.load(f)
        except:
            self.colors = {
                "gov": {"hub": "#0B5ED7", "school": "#22A6B3", "clinic": "#3CA6FF", "park": "#2ECC71"},
                "firm": {"residential": "#F6C344", "retail": "#FD7E14", "office": "#E67E22"},
                "layers": {"trunk": "#9AA4B2", "heat": "#FF00FF", "agent": "#FFFFFF"}
            }
    
    def setup_plot(self):
        """Setup the matplotlib figure and controls"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.suptitle('Enhanced City Simulation V2.3 - Frame-by-Frame Viewer', fontsize=16)
        
        # Create control buttons
        self.ax_play = plt.axes([0.1, 0.02, 0.1, 0.04])
        self.ax_prev = plt.axes([0.25, 0.02, 0.1, 0.04])
        self.ax_next = plt.axes([0.4, 0.02, 0.1, 0.04])
        self.ax_slider = plt.axes([0.6, 0.02, 0.3, 0.04])
        
        # Create buttons
        self.btn_play = Button(self.ax_play, 'Play/Pause')
        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_next = Button(self.ax_next, 'Next')
        
        # Create slider
        self.slider = Slider(self.ax_slider, 'Frame', 0, self.total_frames-1, 
                           valinit=0, valfmt='%d')
        
        # Connect events
        self.btn_play.on_clicked(self.toggle_play)
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)
        self.slider.on_changed(self.on_slider_change)
        
        # Load first frame
        self.load_frame(0)
        
    def load_frame(self, frame_idx):
        """Load and display a specific frame"""
        if 0 <= frame_idx < self.total_frames:
            self.current_frame = frame_idx
            self.slider.set_val(frame_idx)
            
            # Clear current plot
            self.ax.clear()
            
            # Load image
            img_path = self.image_files[frame_idx]
            img = plt.imread(img_path)
            self.ax.imshow(img)
            
            # Set title with frame info
            month = frame_idx
            self.ax.set_title(f'Month {month:02d}', fontsize=14)
            self.ax.axis('off')
            
            # Update display
            self.fig.canvas.draw_idle()
            
    def toggle_play(self, event):
        """Toggle play/pause animation"""
        self.playing = not self.playing
        if self.playing:
            self.animate()
    
    def animate(self):
        """Animate frames"""
        if self.playing and self.current_frame < self.total_frames - 1:
            self.next_frame(None)
            self.fig.canvas.get_tk_widget().after(int(1000/self.fps), self.animate)
        else:
            self.playing = False
    
    def prev_frame(self, event):
        """Go to previous frame"""
        if self.current_frame > 0:
            self.load_frame(self.current_frame - 1)
    
    def next_frame(self, event):
        """Go to next frame"""
        if self.current_frame < self.total_frames - 1:
            self.load_frame(self.current_frame + 1)
    
    def on_slider_change(self, val):
        """Handle slider change"""
        frame_idx = int(val)
        if frame_idx != self.current_frame:
            self.load_frame(frame_idx)
    
    def show(self):
        """Show the visualization"""
        plt.tight_layout()
        plt.show()

def create_simple_viewer():
    """Create a simple frame-by-frame viewer"""
    output_dir = "enhanced_simulation_v2_3_output"
    images_dir = os.path.join(output_dir, "images")
    
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return
    
    image_files = sorted(glob.glob(os.path.join(images_dir, "month_*.png")))
    
    if not image_files:
        print("No image files found!")
        return
    
    print(f"Found {len(image_files)} frames")
    
    # Create simple viewer
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('Enhanced City Simulation V2.3 - Simple Viewer', fontsize=16)
    
    current_frame = 0
    
    def update_frame(frame_idx):
        ax.clear()
        img = plt.imread(image_files[frame_idx])
        ax.imshow(img)
        month = frame_idx
        ax.set_title(f'Month {month:02d}', fontsize=14)
        ax.axis('off')
        return ax,
    
    # Create animation
    anim = animation.FuncAnimation(fig, update_frame, frames=len(image_files), 
                                 interval=500, repeat=True, blit=False)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function"""
    print("Enhanced City Simulation V2.3 Visualizer")
    print("=" * 50)
    
    # Check if output directory exists
    output_dir = "enhanced_simulation_v2_3_output"
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        print("Please run the simulation first.")
        return
    
    # Check if images exist
    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return
    
    image_files = glob.glob(os.path.join(images_dir, "month_*.png"))
    if not image_files:
        print("No image files found!")
        return
    
    print(f"Found {len(image_files)} visualization frames")
    print("Starting visualization...")
    
    # Create and show visualizer
    try:
        visualizer = V2_3Visualizer()
        visualizer.show()
    except Exception as e:
        print(f"Error with interactive viewer: {e}")
        print("Falling back to simple viewer...")
        create_simple_viewer()

if __name__ == "__main__":
    main()
