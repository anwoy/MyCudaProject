import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import glob

# UI cleanup
plt.rcParams['toolbar'] = 'None'

def draw_sphere(ax, center, radius, color='black', alpha=1.0):
    """Adds a solid black sphere at the origin."""
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    
    # shade=True is vital on a light background so the black sphere 
    # looks 3D rather than just a flat black circle.
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, antialiased=True, shade=True)

def animate_light_style():
    # 1. Data Loading
    file_list = sorted(glob.glob('data/trajectory_*.csv'))
    if not file_list:
        print("No trajectory files found.")
        return

    dataframes = [pd.read_csv(f).iloc[::10].reset_index(drop=True) for f in file_list]
    max_frames = max(len(df) for df in dataframes)

    # 2. Appearance Setup (Light Mode)
    bg_color = '#E5E5E5'  # Light Platinum Gray
    fig = plt.figure(facecolor=bg_color)
    ax = fig.add_subplot(projection='3d', facecolor=bg_color)
    
    manager = plt.get_current_fig_manager()
    try: manager.full_screen_toggle()
    except: pass

    LIMIT = 5
    ax.set_xlim(-LIMIT, LIMIT)
    ax.set_ylim(-LIMIT, LIMIT)
    ax.set_zlim(-LIMIT, LIMIT)
    ax.set_box_aspect([1, 1, 1])
    
    # Static Viewpoint from (0, 10, 0)
    ax.view_init(elev=0, azim=-90)
    
    # Draw the black sphere
    draw_sphere(ax, (0, 0, 0), radius=2, color='black')

    scatters = []
    lines = []
    
    # Using 'winter' colormap (Deep Blue to Green) - looks great on light gray
    colors = plt.cm.winter(np.linspace(0, 1, len(dataframes)))
    
    for i in range(len(dataframes)):
        # Added a dark edge to the scatter points to help them pop on light background
        scat = ax.scatter([], [], [], color=colors[i], s=60, edgecolors='black', linewidth=0.8)
        line, = ax.plot([], [], [], color=colors[i], alpha=0.8, linewidth=2.5)
        scatters.append(scat)
        lines.append(line)

    def update(frame):
        artists = []
        for i, df in enumerate(dataframes):
            if frame < len(df):
                x_data = df['x'][:frame + 1]
                y_data = df['y'][:frame + 1]
                z_data = df['z'][:frame + 1]
                
                lines[i].set_data(x_data, y_data)
                lines[i].set_3d_properties(z_data)
                
                scatters[i]._offsets3d = (df['x'][frame:frame+1], 
                                          df['y'][frame:frame+1], 
                                          df['z'][frame:frame+1])
                
                artists.extend([lines[i], scatters[i]])
        return artists

    # 3. Execution
    ani = FuncAnimation(fig, update, frames=max_frames, interval=1, blit=False, repeat=True)

    # Completely clean UI
    ax.set_axis_off() 
    plt.show()

if __name__ == "__main__":
    animate_light_style()
