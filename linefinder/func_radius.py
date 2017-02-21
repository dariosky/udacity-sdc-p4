import matplotlib.pyplot as plt
import numpy as np


def get_line_function(linespacey, linex):
    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(linespacey, linex, 2)
    left_fitx = left_fit[0] * linespacey ** 2 + left_fit[1] * linespacey + left_fit[2]
    plt.plot(left_fitx, linespacey, color='green', linewidth=3)
    plt.gca().invert_yaxis()  # to visualize as we do the images
    plt.show()


if __name__ == '__main__':
    linespacey = np.linspace(0, 719, num=720)  # to cover same y-range as image
