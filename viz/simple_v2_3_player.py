#!/usr/bin/env python3
"""
Simple v2.3 Simulation Results Player
Supports isocontour display and building position analysis
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import numpy as np
import json
import os
import glob

class SimpleV2_3Player:
    """Simple v2.3 simulation results player"""
    
    def __init__(self):
        self.output_dir = 'enhanced_simulation_v2_3_output'
        self.images_dir = f'{self.output_dir}/images'
        self.current_frame = 0
        self.total_frames = 0
        self.image_files = []
        self.building_data = {}
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.setup_ui()
        self.load_data()
        
    def setup_ui(self):
        """Setup user interface"""
        # Main title
        self.fig.suptitle('Enhanced City Simulation v2.3 - Isocontour Building Generation', 
                         fontsize=16, fontweight='bold')
        
        # Control buttons
        ax_prev = plt.axes([0.1, 0.05, 0.1, 0.04])
        ax_play = plt.axes([0.25, 0.05, 0.1, 0.04])
        ax_next = plt.axes([0.4, 0.05, 0.1, 0.04])
        ax_analyze = plt.axes([0.55, 0.05, 0.15, 0.04])
        
        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_play = Button(ax_play, 'Play/Pause')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_analyze = Button(ax_analyze, 'Analyze Buildings')
        
        # Bind events
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_play.on_clicked(self.toggle_play)
        self.btn_next.on_clicked(self.next_frame)
        self.btn_analyze.on_clicked(self.analyze_buildings)
        
        # Frame slider
        ax_slider = plt.axes([0.1, 0.12, 0.6, 0.02])
        self.slider = Slider(ax_slider, 'Frame', 0, 1, valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)
        
        # Status display
        self.status_text = self.fig.text(0.75, 0.12, '', fontsize=10)
        
        # Play state
        self.is_playing = False
        self.animation = None
        
    def load_data(self):
        """Load data"""
        # Load image files
        if os.path.exists(self.images_dir):
            self.image_files = sorted(glob.glob(f'{self.images_dir}/month_*.png'))
            self.total_frames = len(self.image_files)
            print(f"Found {self.total_frames} image files")
            
            # Update slider range
            if self.total_frames > 0:
                self.slider.valmax = self.total_frames - 1
                self.slider.valstep = 1
        else:
            print(f"Image directory not found: {self.images_dir}")
            return
        
        # Load building position data
        self.load_building_positions()
        
        # Show first frame
        if self.total_frames > 0:
            self.show_frame(0)
    
    def load_building_positions(self):
        """Load building position data"""
        building_files = glob.glob(f'{self.output_dir}/building_positions_month_*.json')
        
        for file_path in building_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    month = data['timestamp']
                    self.building_data[month] = data
            except Exception as e:
                print(f"Failed to load building data {file_path}: {e}")
        
        print(f"Loaded building data for {len(self.building_data)} months")
    
    def show_frame(self, frame_index: int):
        """Show specified frame"""
        if frame_index < 0 or frame_index >= self.total_frames:
            return
        
        self.current_frame = frame_index
        
        # Clear current image
        self.ax.clear()
        
        # Load and display image
        if os.path.exists(self.image_files[frame_index]):
            img = plt.imread(self.image_files[frame_index])
            self.ax.imshow(img)
            
            # Set title
            month = frame_index
            self.ax.set_title(f'Month {month:02d} - Isocontour Building Generation', fontsize=14)
            
            # Show building stats
            self.show_building_stats(month)
            
            # Show isocontour info
            self.show_isocontour_info(month)
        
        # Update status
        self.update_status()
        
        # Update slider
        self.slider.set_val(frame_index)
        
        # Refresh display
        self.fig.canvas.draw_idle()
    
    def show_building_stats(self, month: int):
        """Show building statistics"""
        month_key = f'month_{month:02d}'
        if month_key in self.building_data:
            data = self.building_data[month_key]
            buildings = data['buildings']
            
            # Count building types
            residential_count = len([b for b in buildings if b['type'] == 'residential'])
            commercial_count = len([b for b in buildings if b['type'] == 'commercial'])
            public_count = len([b for b in buildings if b['type'] == 'public'])
            
            # Display statistics
            stats_text = f'Buildings:\nResidential: {residential_count}\nCommercial: {commercial_count}\nPublic: {public_count}'
            self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def show_isocontour_info(self, month: int):
        """Show isocontour information"""
        # Show isocontour legend
        info_text = ('Isocontours:\n'
                    'Red dashed: Commercial\n'
                    'Blue dashed: Residential\n'
                    'Orange dots: Commercial buildings\n'
                    'Yellow squares: Residential buildings\n'
                    'Cyan markers: Public facilities')
        
        self.ax.text(0.98, 0.98, info_text, transform=self.ax.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def update_status(self):
        """Update status display"""
        if self.total_frames > 0:
            month = self.current_frame
            status = f'Current: Month {month:02d} / {self.total_frames-1}'
            self.status_text.set_text(status)
    
    def prev_frame(self, event):
        """Previous frame"""
        if self.total_frames > 0:
            new_frame = max(0, self.current_frame - 1)
            self.show_frame(new_frame)
    
    def next_frame(self, event):
        """Next frame"""
        if self.total_frames > 0:
            new_frame = min(self.total_frames - 1, self.current_frame + 1)
            self.show_frame(new_frame)
    
    def toggle_play(self, event):
        """Toggle play/pause"""
        if self.is_playing:
            self.stop_animation()
        else:
            self.start_animation()
    
    def start_animation(self):
        """Start animation"""
        if self.total_frames <= 1:
            return
        
        self.is_playing = True
        self.btn_play.label.set_text('Pause')
        
        # Create animation
        def animate(frame):
            self.show_frame(frame)
            return []
        
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=self.total_frames,
            interval=1000, repeat=True, blit=False
        )
    
    def stop_animation(self):
        """Stop animation"""
        self.is_playing = False
        self.btn_play.label.set_text('Play')
        
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
    
    def on_slider_change(self, val):
        """Slider value changed"""
        frame_index = int(val)
        if frame_index != self.current_frame:
            self.show_frame(frame_index)
    
    def analyze_buildings(self, event):
        """Analyze building distribution"""
        if not self.building_data:
            print("No building data available for analysis")
            return
        
        # Create analysis window
        self.create_analysis_window()
    
    def create_analysis_window(self):
        """Create analysis window"""
        analysis_fig, analysis_axes = plt.subplots(2, 2, figsize=(16, 12))
        analysis_fig.suptitle('Building Distribution Analysis - v2.3', fontsize=16, fontweight='bold')
        
        # 1. Building type distribution
        ax1 = analysis_axes[0, 0]
        self.plot_building_types(ax1)
        
        # 2. Building position distribution
        ax2 = analysis_axes[0, 1]
        self.plot_building_positions(ax2)
        
        # 3. SDF value distribution
        ax3 = analysis_axes[1, 0]
        self.plot_sdf_distribution(ax3)
        
        # 4. Monthly growth trend
        ax4 = analysis_axes[1, 1]
        self.plot_growth_trend(ax4)
        
        plt.tight_layout()
        plt.show()
    
    def plot_building_types(self, ax):
        """Plot building type distribution"""
        # Count all buildings
        total_residential = 0
        total_commercial = 0
        total_public = 0
        
        for month_data in self.building_data.values():
            for building in month_data['buildings']:
                if building['type'] == 'residential':
                    total_residential += 1
                elif building['type'] == 'commercial':
                    total_commercial += 1
                elif building['type'] == 'public':
                    total_public += 1
        
        # Plot pie chart
        labels = ['Residential', 'Commercial', 'Public']
        sizes = [total_residential, total_commercial, total_public]
        colors = ['#F6C344', '#FD7E14', '#22A6B3']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Building Type Distribution')
    
    def plot_building_positions(self, ax):
        """Plot building position distribution"""
        # Get latest month data
        latest_month = max(self.building_data.keys())
        data = self.building_data[latest_month]
        
        # Plot building positions
        for building in data['buildings']:
            pos = building['position']
            if building['type'] == 'residential':
                ax.scatter(pos[0], pos[1], c='yellow', s=50, marker='s', label='Residential', alpha=0.7)
            elif building['type'] == 'commercial':
                ax.scatter(pos[0], pos[1], c='orange', s=40, marker='o', label='Commercial', alpha=0.7)
            elif building['type'] == 'public':
                ax.scatter(pos[0], pos[1], c='cyan', s=60, marker='^', label='Public', alpha=0.7)
        
        # Plot transport hubs
        ax.scatter([40, 216], [128, 128], c='blue', s=200, marker='s', label='Transport Hubs', alpha=0.8)
        
        # Plot main road
        ax.plot([40, 216], [128, 128], 'gray', linewidth=3, alpha=0.5, label='Main Road')
        
        ax.set_xlim(0, 256)
        ax.set_ylim(0, 256)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'Building Position Distribution ({latest_month})')
        
        # Show legend only once
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    
    def plot_sdf_distribution(self, ax):
        """Plot SDF value distribution"""
        # Collect SDF values from all buildings
        sdf_values = []
        for month_data in self.building_data.values():
            for building in month_data['buildings']:
                if 'sdf_value' in building and building['sdf_value'] > 0:
                    sdf_values.append(building['sdf_value'])
        
        if sdf_values:
            ax.hist(sdf_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('SDF Value')
            ax.set_ylabel('Building Count')
            ax.set_title('Building SDF Value Distribution')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No SDF Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Building SDF Value Distribution')
    
    def plot_growth_trend(self, ax):
        """Plot monthly growth trend"""
        months = []
        residential_counts = []
        commercial_counts = []
        public_counts = []
        
        for month_key in sorted(self.building_data.keys()):
            month_num = int(month_key.split('_')[1])
            months.append(month_num)
            
            data = self.building_data[month_key]
            buildings = data['buildings']
            
            residential_counts.append(len([b for b in buildings if b['type'] == 'residential']))
            commercial_counts.append(len([b for b in buildings if b['type'] == 'commercial']))
            public_counts.append(len([b for b in buildings if b['type'] == 'public']))
        
        if months:
            ax.plot(months, residential_counts, 'o-', label='Residential', color='#F6C344', linewidth=2)
            ax.plot(months, commercial_counts, 's-', label='Commercial', color='#FD7E14', linewidth=2)
            ax.plot(months, public_counts, '^-', label='Public', color='#22A6B3', linewidth=2)
            
            ax.set_xlabel('Month')
            ax.set_ylabel('Building Count')
            ax.set_title('Building Growth Trend')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Growth Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Building Growth Trend')
    
    def run(self):
        """Run the visualizer"""
        print("v2.3 Simple Player Started")
        print("Supports isocontour display and building analysis")
        print("Use buttons to control playback, slider for quick navigation")
        
        plt.show()

def main():
    """Main function"""
    player = SimpleV2_3Player()
    player.run()

if __name__ == "__main__":
    main()
