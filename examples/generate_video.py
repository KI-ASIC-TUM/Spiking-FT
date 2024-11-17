#!/usr/bin/env python3
"""
Module one-line definition
"""
# Standard libraries
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
# Local libraries
import dhandler.oth_handler


class Animator():
    def __init__(self, sequence, images):
        self.fig, self.ax = plt.subplots(2)
        self.sequence = sequence
        self.images = images
        self.line = None

    def init_func(self):
        self.im = self.ax[0].imshow(self.images[0])
        self.ax[0].set_title("Camera input", fontsize=16)
        self.ax[0].set_xticks([])
        self.ax[0].set_yticks([])
        self.line, = self.ax[1].plot(self.sequence[0][2:], linewidth=2)
        self.ax[1].set_yticks([])
        self.ax[1].set_title("Spiking FT", fontsize=16)
        self.ax[1].set_xlabel("Bin Nº", fontsize=13)
        self.ax[1].spines['left'].set_visible(False)
        self.ax[1].spines['top'].set_visible(False)
        self.ax[1].spines['right'].set_visible(False)

    def step(self, i):
        self.im.set_data(self.images[i])
        self.line.set_ydata(self.sequence[i][2:])

def gen_frame(in_data, in_image):
    fig, ax = plt.subplots(2)
    im = ax[0].imshow(in_image)
    ax[0].set_title("Camera input", fontsize=16)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    line, = ax[1].plot(in_data[2:], linewidth=2)
    ax[1].set_yticks([])
    ax[1].set_title("Spiking FT", fontsize=16)
    ax[1].set_xlabel("Bin Nº", fontsize=13)
    ax[1].spines['left'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    plt.savefig("spinn_out_42.pdf", dpi=142)
    plt.show()

def get_images():
    handler = dhandler.oth_handler.OTHHandler("OTH/20230127_114455")
    images = handler.load_images(image_types=['image_left_stereo'])
    return images

def collect_data(n_frames):
    sequence = []
    for frame_n in range(n_frames):
        frame = np.load("./results/bins128_timesteps100/spinn_out_{}.npy".format(frame_n))
        sequence.append(frame)
    return sequence

def gen_video(sequence, images, n_frames):
    ani = Animator(sequence, images)
    ani = FuncAnimation(ani.fig, ani.step, frames=n_frames, init_func=ani.init_func(), interval=100, repeat=False)
    ani.save('fft_movie.mp4')
    plt.show()
    pass

def main(n_frames=111):
    """
    Main routine
    """
    out_sequence = collect_data(n_frames)
    images = get_images()
    # gen_video(out_sequence, images, n_frames)
    gen_frame(out_sequence[42], images[42])


if __name__ == "__main__":
    main()
