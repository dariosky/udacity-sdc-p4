import cv2
import numpy as np


def binary(img,
           sobelx_thresh=(20, 250),
           luminosity_thresh=(230, 255),
           saturation_thresh=(200, 255),
           ):
    """ Give a color image, get a binary image with edges combining sobelx gradient
        and tresholded saturation channel and luminosity
    """
    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.uint8)
    l_channel = hls[:, :, 1]
    saturation_channel = hls[:, :, 2]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    # Absolute x derivative to accentuate lines away from horizontal
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    scaled_luminosity = np.uint8(255 * l_channel / np.max(l_channel))

    # Binary composition of the 3 functions with their thresold
    composition = np.zeros_like(scaled_sobelx)
    composition[
        (scaled_sobelx >= sobelx_thresh[0]) & (scaled_sobelx < sobelx_thresh[1])
        |
        (saturation_channel >= saturation_thresh[0]) & (saturation_channel < saturation_thresh[1])
        |
        (scaled_luminosity >= luminosity_thresh[0]) & (scaled_luminosity < luminosity_thresh[1])
        ] = 255
    return composition
