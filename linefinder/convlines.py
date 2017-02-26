import cv2
import numpy as np
from blinker import signal
import matplotlib.pyplot as plt

lane_message = signal('lane_message')


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
    window_height = img.shape[0] // 9
    margin = img.shape[1] // 30  # How much to slide left and right for searching
    return window_width, window_height, margin


def find_window_centroids(img, window_width, window_height, margin,
                          suggested_centers=None):
    # Store the (left,right) window centroid positions per level
    window_centroids = np.zeros([img.shape[0] // window_height, 2], np.float32)
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum
    #   to get the vertical image slice
    #   and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    if not suggested_centers:
        # blind scan for left lane
        lane_message.send(message="Blind scan for lanes")
        l_sum = np.sum(img[int(3 * img.shape[0] / 4):, :int(img.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_sum = np.sum(img[int(3 * img.shape[0] / 4):, int(img.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(img.shape[1] / 2)

        # Add what we found for the first layer
        window_centroids[0, :] = [l_center, r_center]
        starting_level = 1
    else:
        l_center, r_center = suggested_centers
        starting_level = 0

    # Go through each layer looking for max pixel locations
    for level in range(starting_level, int(img.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(img[
                             int(img.shape[0] - (level + 1) * window_height):
                             int(img.shape[0] - level * window_height),
                             :], axis=0)
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
        window_centroids[level] = [l_center, r_center]
    return window_centroids


def get_convoluted_lines(binary):
    window_width, window_height, margin = get_window_size(binary)

    window_centroids = find_window_centroids(binary, window_width, window_height, margin)
    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(binary, dtype=np.uint8)
        r_points = np.zeros_like(binary, dtype=np.uint8)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, binary, window_centroids[level][0],
                                 level)
            r_mask = window_mask(window_width, window_height, binary, window_centroids[level][1],
                                 level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | (l_mask == 1)] = 255
            r_points[(r_points == 255) | (r_mask == 1)] = 255

        return l_points, r_points

    # If no window centers found, return None
    return None, None


def get_left_right_lanes(img, suggested_centers=None):
    window_width, window_height, margin = get_window_size(img)
    c = find_window_centroids(img, window_width, window_height, margin,
                              suggested_centers=suggested_centers)

    leftx = c[:, 0]
    rightx = c[:, 1]
    img_size = img.shape
    ploty = np.linspace(0, img_size[0] - 1,
                        num=img_size[0] // window_height)  # the Y of the detected windows

    left_fit = np.polyfit(ploty, leftx, 2)
    right_fit = np.polyfit(ploty, rightx, 2)

    y = np.linspace(0, 8, num=9)
    left_fitx = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
    right_fitx = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
    mark_size = 3

    if False:
        plt.imshow(img)
        plt.plot(leftx, 720 - y * 60, 'o', color='red', markersize=mark_size)
        plt.plot(rightx, 720 - y * 60, 'o', color='blue', markersize=mark_size)
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.gca().invert_yaxis()  # to visualize as we do the images
        plt.show()

    return left_fit, right_fit


def radius_in_meters(y, leftx, rightx,
                     YM_PER_PIX=30 / 720,  # meters per pixel in y dimension
                     XM_PER_PIX=3.7 / 700,  # meters per pixel in x dimension
                     ):
    y_eval = np.max(y)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(y * YM_PER_PIX, leftx * XM_PER_PIX, 2)
    right_fit_cr = np.polyfit(y * YM_PER_PIX, rightx * XM_PER_PIX, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (
        2 * left_fit_cr[0] * y_eval * YM_PER_PIX + left_fit_cr[1]) ** 2) ** 1.5) \
                    / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (
        2 * right_fit_cr[0] * y_eval * YM_PER_PIX + right_fit_cr[1]) ** 2) ** 1.5) \
                     / np.absolute(2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad


def get_line_points(img, suggested_centers=None):
    img_size = img.shape
    fully = np.linspace(0, img_size[0] - 1,
                        num=img_size[0])  # to cover same y-range as image

    left_fit, right_fit = get_left_right_lanes(img,
                                               suggested_centers=suggested_centers)

    left_fitx = left_fit[0] * fully ** 2 + left_fit[1] * fully + left_fit[2]
    right_fitx = right_fit[0] * fully ** 2 + right_fit[1] * fully + right_fit[2]

    # let's get where are the lane at the camera position - bottom of screen
    y = 0
    left_pos = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
    right_pos = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
    extra = dict(
        left_pos=left_pos,
        right_pos=right_pos,
    )
    return fully, left_fitx, right_fitx, extra


def line_on_the_road(binary, undist, Minv, image, y, leftx, rightx, unwarp=True):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary).astype(np.uint8)
    color_warp = warp_zero

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([leftx, y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), 255)

    color_warp = np.flipud(color_warp)

    if unwarp is False:
        points_1l, points_1r = get_convoluted_lines(binary)

        result = cv2.addWeighted(binary, 1, color_warp, 0.3, 0)  # binary + color
        result = cv2.addWeighted(result, 1, points_1l | points_1r, .5, 0)
        return result

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(np.dstack((color_warp * 0.2, color_warp, color_warp * 0.2)), Minv,
                                  (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0, dtype=24)
    return result
