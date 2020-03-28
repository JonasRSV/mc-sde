from sdepy.core import PDF
from sdepy.pdf.histogram import Simple1DHistogram
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def make_time_1dx_distplot_video(pdfs: List[PDF],
                                 fps: int,
                                 steps_per_frame: int,
                                 dt,
                                 save=True):

    seconds_per_step = dt * steps_per_frame
    fig, ax = plt.subplots(figsize=(15, 8))
    frames = []

    min_max = 10000
    for i, pdf in enumerate(pdfs, 1):
        line = np.linspace(pdf.lower_bound, pdf.upper_bound)
        values = pdf(line)
        lp = ax.plot(line, values, animated=True, c="black", alpha=0.5)
        fill = ax.fill_between(line, np.zeros_like(values), values, alpha=0.3, color="blue")
        low = ax.vlines([pdf.lower_bound], ymin=-10, ymax=10, colors="green")
        mean = ax.vlines([pdf.mean], ymin=-10, ymax=10, colors="yellow")
        high = ax.vlines([pdf.upper_bound], ymin=-10, ymax=10, colors="red")

        legend = fig.legend((lp[0], high, low, mean), ("time: %.2f" % (seconds_per_step * i),
                                 "worst: %.2f" % (pdf.upper_bound),
                                 "best: %.2f" % (pdf.lower_bound),
                                 "average: %.2f" % (pdf.mean)), fontsize=14)

        ax.add_artist(legend)
        frames.append((lp[0], fill, legend, low, mean, high))

        min_max = np.minimum(min_max, values.max())

    ax.set_ylim([0, np.maximum(min_max, 1e-8)])
    ax.tick_params(axis="x", labelsize=14)
    ax.set_yticks([])
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=False, repeat=True)
    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("animation.mp4", writer=writer)

    return ani


if __name__ == "__main__":
    pass
