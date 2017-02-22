import glob

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from linefinder.birdeye import get_dstbox, asphalt_box
from linefinder.camera_calibration import get_calibration_points
from linefinder.convlines import get_line_points, radius_in_meters, line_on_the_road
from linefinder.edges import binary


class ProcessPipeline:
    def __init__(self) -> None:
        self.mtx = None  # the Camera undistorion matrix
        self.dist = None  # dist coefficient for undistortion

    def camera_calibration(self, img_size):
        images = glob.glob('camera_cal/calibration*.jpg')  # Make a list of calibration images
        objpoints, imgpoints = get_calibration_points(images,
                                                      chessboard_size=(9, 6),
                                                      display=False,
                                                      pickle_filename='calibration_points.pickle')
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                                     img_size,
                                                                     None, None)

    def ensure_calibrated(self):
        if self.mtx is None:
            self.camera_calibration(self.img_size)

    def load(self, img):
        self.img = img  # keep the original image with the process
        self.img_size = self.img.shape[:2]

    def process(self, img,
                processes={'undistort',
                           'binary',
                           'birdeye',
                           'lines',
                           },
                plot_steps=True,
                ):
        self.load(img)
        out = img

        step = 0
        if plot_steps:
            f, axes = plt.subplots(len(processes) + 2)
            f.subplots_adjust(hspace=0)

            axes[step].imshow(img)

        if 'undistort' in processes:
            self.ensure_calibrated()
            self.undist = out = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
            if plot_steps:
                step += 1
                axes[step].imshow(out)

        if 'binary' in processes:
            self.binary = out = binary(out)
            if plot_steps:
                step += 1
                axes[step].imshow(out)

        if 'birdeye' in processes:
            dstbox = get_dstbox(asphalt_box, shape=self.img_size)
            srcbox_xy = np.array([(x, y) for y, x in asphalt_box])
            dstbox_xy = np.array([(x, y) for y, x in dstbox])

            # let's get the 2 matrix to get the birdeye and back
            self.M = cv2.getPerspectiveTransform(srcbox_xy, dstbox_xy)
            self.Minv = cv2.getPerspectiveTransform(dstbox_xy,
                                                    srcbox_xy)

            for x, y in asphalt_box:
                axes[step].plot(y, x, 'o', color='red', markersize=6)

            self.warped = out = cv2.warpPerspective(out, self.M,
                                                    self.img_size[::-1],
                                                    flags=cv2.INTER_LINEAR)
            self.undistwarped = cv2.warpPerspective(self.undist, self.M,
                                                    self.img_size[::-1],
                                                    flags=cv2.INTER_LINEAR)
            if plot_steps:
                step += 1
                axes[step].imshow(self.undistwarped)
                step += 1
                axes[step].imshow(out)

        if 'lines' in processes:
            # get the lines points
            lines = get_line_points(out)
            # get the curvature radius (in meters)
            self.radiuses = radius_in_meters(*lines)
            print(self.radiuses)
            out = line_on_the_road(self.warped, self.undist,
                                   self.Minv,
                                   self.img,
                                   *lines)

        return out


def run():
    img = mpimg.imread('test_images/straight_lines1.jpg')
    img = mpimg.imread('test_images/test5.jpg')
    pipeline = ProcessPipeline()

    out = pipeline.process(img)

    plt.imshow(out)
    plt.show()


if __name__ == '__main__':
    run()
