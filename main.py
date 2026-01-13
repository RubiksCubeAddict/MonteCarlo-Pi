import matplotlib
matplotlib.use('TkAgg')  # Ensure GUI backend with interactivity

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def monte_carlo_pi_animation(num_samples: int, batch_size: int = None):
    # Batch size logic
    if batch_size is None:
        if num_samples <= 1_000_000:
            batch_size = 10_000
        elif num_samples <= 10_000_000:
            batch_size = 100_000
        elif num_samples <= 100_000_000:
            batch_size = 1_000_000
        else:
            batch_size = 5_000_000

    inside_total = 0
    samples_done = 0
    running = [False]  # Start paused until start button clicked

    disp_x, disp_y, disp_colors = [], [], []

    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    tk_window = plt.get_current_fig_manager().window
    tk_window.attributes('-topmost', False)


    # Initially hide axes spines, ticks, labels and grid:
    ax.set_visible(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    circle = plt.Circle((0, 0), 1, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(circle)
    circle.set_visible(False)

    scatter = ax.scatter([], [], s=3, alpha=0.6)
    pi_text = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha='center', va='bottom', fontsize=12)
    samples_text = fig.text(0.99, 0.01, "", ha='right', va='bottom', fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # STOP button top-right (initially hidden)
    stop_ax = plt.axes([0.75, 0.92, 0.1, 0.05])
    stop_button = Button(stop_ax, 'STOP', color='lightcoral', hovercolor='red')
    stop_ax.set_visible(False)

    # RESUME button top-right (initially hidden)
    resume_ax = plt.axes([0.86, 0.92, 0.1, 0.05])
    resume_button = Button(resume_ax, 'RESUME', color='lightgreen', hovercolor='green')
    resume_ax.set_visible(False)

    # RESTART button top-left (initially hidden)
    restart_ax = plt.axes([0.01, 0.92, 0.1, 0.05])
    restart_button = Button(restart_ax, 'RESTART', color='lightblue', hovercolor='deepskyblue')
    restart_ax.set_visible(False)

    # START overlay button in center (big button)
    start_ax = plt.axes([0.3, 0.3, 0.4, 0.4])  # 40% width/height in center
    start_button = Button(start_ax, 'START', color='white')
    start_button.label.set_color('black')

    def reset_data():
        return 0, 0, [], [], [], []

    def stop(event):
        running[0] = False

    def resume(event):
        running[0] = True

    def restart(event):
        nonlocal inside_total, samples_done, disp_x, disp_y, disp_colors, point_queue
        inside_total, samples_done, disp_x, disp_y, disp_colors, point_queue = reset_data()
        running[0] = False
        scatter.set_offsets(np.empty((0, 2)))  # Properly clear points
        scatter.set_color([])
        pi_text.set_text("")
        samples_text.set_text("")
        # Hide main buttons, show start button again
        start_ax.set_visible(True)
        stop_ax.set_visible(False)
        resume_ax.set_visible(False)
        restart_ax.set_visible(False)
        # Hide circle and axes again
        circle.set_visible(False)
        ax.set_visible(False)
        fig.canvas.draw_idle()

    def start(event):
        running[0] = True
        start_ax.set_visible(False)
        stop_ax.set_visible(True)
        resume_ax.set_visible(True)
        restart_ax.set_visible(True)
        # Show circle and axes now
        circle.set_visible(True)
        ax.set_visible(True)
        fig.canvas.draw_idle()

    stop_button.on_clicked(stop)
    resume_button.on_clicked(resume)
    restart_button.on_clicked(restart)
    start_button.on_clicked(start)

    point_queue = []

    while True:
        if not running[0]:
            plt.pause(0.1)
            if not plt.fignum_exists(fig.number):
                break
            continue

        if samples_done >= num_samples:
            running[0] = False
            fig.canvas.draw_idle()
            break

        if len(point_queue) == 0:
            current_batch = min(batch_size, num_samples - samples_done)
            x = np.random.uniform(-1, 1, current_batch)
            y = np.random.uniform(-1, 1, current_batch)
            dist_sq = x**2 + y**2
            inside = dist_sq <= 1
            inside_total += np.sum(inside)
            samples_done += current_batch

            step = max(1, current_batch // 200)
            for i in range(0, current_batch, step):
                color = 'blue' if inside[i] else 'red'
                point_queue.append((x[i], y[i], color))

        for _ in range(min(100, len(point_queue))):
            px, py, color = point_queue.pop(0)
            disp_x.append(px)
            disp_y.append(py)
            disp_colors.append(color)

        scatter.set_offsets(np.column_stack((disp_x, disp_y)))
        scatter.set_color(disp_colors)

        pi_estimate = (inside_total / samples_done) * 4
        pi_text.set_text(f"Pi Estimate: {format(pi_estimate, '.15g')}")
        samples_text.set_text(f"Samples: {samples_done:,} / {num_samples:,}")

        fig.canvas.draw_idle()
        plt.pause(0.001)

        if not plt.fignum_exists(fig.number):
            break

    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Estimate π using the Monte Carlo method.")
    parser.add_argument('--samples', type=str, required=True,
                        help="Total number of samples (e.g. 1_000_000 or 1,000,000)")
    parser.add_argument('--batch', type=int, default=None, help="Optional batch size")
    args = parser.parse_args()

    try:
        samples = int(args.samples.replace(',', '').replace('_', ''))
        if samples <= 0:
            raise ValueError
    except ValueError:
        print("Invalid number of samples provided.")
        return

    monte_carlo_pi_animation(samples, batch_size=args.batch)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")