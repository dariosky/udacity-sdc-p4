import cv2
import numpy as np


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def get_window_size(img):
    """ A default windows size from the image """
    # window settings
    window_width = img.shape[1] // 25
    # Break image into 9 vertical layers since image height is 720
    window_height = img.shape[0] // 7
    margin = img.shape[1] // 30  # How much to slide left and right for searching
    return window_width, window_height, margin


def find_window_centroids(img, window_width, window_height, margin):
    # Store the (left,right) window centroid positions per level
    # window_centroids = np.zeros([img.shape[0] / window_height, 2], np.float32)
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum
    #   to get the vertical image slice
    #   and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(img[int(3 * img.shape[0] / 4):, :int(img.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(img[int(3 * img.shape[0] / 4):, int(img.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(img.shape[1] / 2)

    # Add what we found for the first layer
    # window_centroids[0, :] = [l_center, r_center]
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(img.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(img[int(img.shape[0] - (level + 1) * window_height):int(
            img.shape[0] - level * window_height), :], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference
        #   is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, img.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, img.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        # window_centroids[level] = (l_center, r_center)
        window_centroids.append((l_center, r_center))
    return window_centroids


def get_convoluted_lines(img):
    window_width, window_height, margin = get_window_size(img)

    window_centroids = find_window_centroids(img, window_width, window_height, margin)
    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(img, dtype=np.uint8)
        r_points = np.zeros_like(img, dtype=np.uint8)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, img, window_centroids[level][0],
                                 level)
            r_mask = window_mask(window_width, window_height, img, window_centroids[level][1],
                                 level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | (l_mask == 1)] = 255
            r_points[(r_points == 255) | (r_mask == 1)] = 255

        return l_points, r_points

    # If no window centers found, return None
    return None, None


if __name__ == '__main__':
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    warped = mpimg.imread('warped_example.png')
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    l, r = get_convoluted_lines(warped)
    # l = centers[:, 0]
    # r = centers[:, 1]
    plt.imshow(l | r)
    plt.title('window fitting results')
    plt.show()
