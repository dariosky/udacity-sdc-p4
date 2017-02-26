import glob
import os
import statistics

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

from linefinder.birdeye import get_dstbox, asphalt_box
from linefinder.camera_calibration import get_calibration_points
from linefinder.convlines import get_line_points, radius_in_meters, line_on_the_road
from linefinder.edges import binary
from blinker import signal


class Steps:
    original = -1
    undistort = 0
    warp = 1
    binary = 2
    detect_lines = 3
    lines_on_road = 4


ALL_STEPS = (
    Steps.original,
    Steps.undistort,
    Steps.warp,
    Steps.binary,
    Steps.detect_lines,
    Steps.lines_on_road,
)


class ProcessPipeline:
    def __init__(self) -> None:
        self.camera = None  # the Camera description
        self.output_prefix = ""
        self.save_steps = False
        self._save_step = 1

        self._plot_step = 0  # counter for the plot
        self.plot_axes = None

        self.YM_PER_PIX = 30 / 720  # meters per pixel in y dimension
        self.XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension
        self.message = ""
        self.message_subscribe()

    def message_subscribe(self):
        """ Subscribe to lane_message events, they can arrive from the pipeline """
        signal('lane_message').connect(self.on_lane_message)

    def on_lane_message(self, sender, message=""):
        # print("Got a message", message)
        self.message = message

    def camera_calibration(self, img_size):
        images = glob.glob('camera_cal/calibration*.jpg')  # Make a list of calibration images
        objpoints, imgpoints = get_calibration_points(images,
                                                      chessboard_size=(9, 6),
                                                      display=False,
                                                      pickle_filename='calibration_points.pickle')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                           img_size,
                                                           None, None)
        self.camera = dict(
            mtx=mtx,
            dist=dist
        )

    def save_camera_calibration_examples(self, example_path='camera_cal/calibration1.jpg'):
        img = mpimg.imread(example_path)
        self.img_size = img.shape[:2]
        self.ensure_calibrated()
        print("Saving camera undistortion example")
        mpimg.imsave('output_images/camera_1_original.jpg', img)
        mtx, dist = self.camera['mtx'], self.camera['dist']
        out = cv2.undistort(img, mtx, dist, None, mtx)
        mpimg.imsave('output_images/camera_2_undistorted.jpg', out)

    def ensure_calibrated(self):
        if self.camera is None:
            self.camera_calibration(self.img_size)

    def load(self, img):
        self.img = img  # keep the original image with the process
        self.img_size = self.img.shape[:2]

    def save(self, img, name):
        step = self._save_step
        out_file_template = "output_images/{prefix}_step{step}_{name}.jpg"
        filename = out_file_template.format(step=step,
                                            name=name,
                                            prefix=self.output_prefix
                                            )
        print("Saving step: %s" % filename)
        mpimg.imsave(filename, img)
        self._save_step += 1

    def plot(self, img):
        step = self._plot_step
        self.plot_axes[step].imshow(img)
        self._plot_step += 1

    def undistort(self, img):
        mtx, dist = self.camera['mtx'], self.camera['dist']
        return cv2.undistort(img, mtx, dist, None, mtx)

    def process(self, img,
                plot_steps=(),
                ):
        self.load(img)

        if plot_steps:
            f, self.plot_axes = plt.subplots(len(plot_steps))
            f.subplots_adjust(hspace=0)

        if self.save_steps:
            self.save(img, 'original')
        if Steps.original in plot_steps:
            self.plot(img)

        self.ensure_calibrated()
        self.undist = self.undistort(img)

        if self.save_steps:
            self.save(self.undist, 'undistorted')
        if Steps.undistort in plot_steps:
            self.plot(self.undist)

        dstbox = get_dstbox(asphalt_box, shape=self.img_size)
        srcbox_xy = np.array([(x, y) for y, x in asphalt_box])
        dstbox_xy = np.array([(x, y) for y, x in dstbox])

        # let's get the 2 matrix to get the birdeye and back
        self.M = cv2.getPerspectiveTransform(srcbox_xy, dstbox_xy)
        self.Minv = cv2.getPerspectiveTransform(dstbox_xy,
                                                srcbox_xy)

        # if plot_steps:
        #     for x, y in asphalt_box:
        #         axes[step].plot(y, x, 'o', color='red', markersize=6)

        self.warped = cv2.warpPerspective(self.undist, self.M,
                                          self.img_size[::-1],
                                          flags=cv2.INTER_LINEAR)
        if self.save_steps:
            self.save(self.warped, 'birdeye')
        if Steps.warp in plot_steps:
            self.plot(self.warped)

        self.binary = binary(self.warped)

        if self.save_steps:
            self.save(self.binary, 'binary')
        if Steps.binary in plot_steps:
            self.plot(self.binary)

        # get the lines points
        y, leftx, rightx, extra = get_line_points(
            # cv2.cvtColor(self.binary, cv2.COLOR_RGB2GRAY)
            self.binary
        )
        l_center, r_center = leftx[0], rightx[0]  # save the initial lane center

        lane_center = statistics.mean([extra['left_pos'], extra['right_pos']])
        self.distance_from_center = (self.img_size[1] // 2 - lane_center) * self.XM_PER_PIX

        gray = cv2.cvtColor(self.warped, cv2.COLOR_RGB2GRAY)

        if self.save_steps or Steps.detect_lines in plot_steps:
            warped_lines = line_on_the_road(gray,
                                            self.undist,
                                            self.Minv,
                                            self.img,
                                            y, leftx, rightx,
                                            unwarp=False)
            if self.save_steps:
                self.save(warped_lines, 'warped_detection')

            if Steps.detect_lines in plot_steps:
                self.plot(warped_lines)

        out = line_on_the_road(gray,
                               self.undist,
                               self.Minv,
                               self.img,
                               y, leftx, rightx,
                               )
        font = cv2.FONT_HERSHEY_SIMPLEX
        # get the curvature radius (in meters)
        self.radii = radius_in_meters(y, leftx, rightx)
        cv2.putText(out, "radii {:.0f}m {:.0f}m".format(*self.radii), (50, 50),
                    font, 1, (255, 100, 100), 2, cv2.LINE_AA)
        cv2.putText(out, "center shift {:.2f}m".format(self.distance_from_center), (50, 85),
                    font, 1, (255, 100, 100), 2, cv2.LINE_AA)
        cv2.putText(out, self.message, (self.img_size[1] // 2, 65),
                    font, 1.2, (255, 100, 100), 2, cv2.LINE_AA)

        if self.save_steps:
            self.save(out, 'detection')
        if Steps.lines_on_road in plot_steps:
            self.plot(out)
        if plot_steps:
            plt.show()
        return out


def run_single(filename):
    img = mpimg.imread(filename)
    pipeline = ProcessPipeline()

    if False:  # set to True to enable extra output_images saves
        pipeline.save_steps = True
        pipeline.save_camera_calibration_examples()

    pipeline.output_prefix = os.path.splitext(os.path.basename(filename))[0]

    pipeline.process(
        img,
        plot_steps=(
            # Steps.original,
            Steps.undistort,
            Steps.warp,
            # Steps.binary,
            # Steps.detect_lines,
            # Steps.lines_on_road,
        )
    )


def run_video(filename='video/project_video.mp4'):
    pipeline = ProcessPipeline()
    pipeline.output_prefix = os.path.splitext(os.path.basename(filename))[0]

    def process_image(image):
        out = pipeline.process(image, plot_steps=[])
        return out

    clip1 = VideoFileClip(filename).subclip(t_end=1)
    white_clip = clip1.fl_image(process_image)
    output_video_filename = 'video/{prefix}_output.mp4'.format(prefix=pipeline.output_prefix)
    white_clip.write_videofile(output_video_filename, audio=False)


def run_sequence():
    image_sequence = sorted(glob.glob("test_images/sequence/*.png"))[1:2]
    # pipeline = ProcessPipeline()
    # pipeline.output_prefix = "sequence"
    for filename in image_sequence:
        run_single(filename)
        # img = mpimg.imread(filename)
        # print(filename)
        # out = pipeline.process(img)
        # plt.imshow(out)
        # plt.show()


if __name__ == '__main__':
    run_single('test_images/test1.jpg')
    # run_video()
    # run_sequence()
