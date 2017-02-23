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


class ProcessPipeline:
    def __init__(self) -> None:
        self.mtx = None  # the Camera undistorion matrix
        self.dist = None  # dist coefficient for undistortion
        self.output_prefix = ""
        self.save_steps = False

        self.YM_PER_PIX = 30 / 720  # meters per pixel in y dimension
        self.XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension

    def camera_calibration(self, img_size):
        images = glob.glob('camera_cal/calibration*.jpg')  # Make a list of calibration images
        objpoints, imgpoints = get_calibration_points(images,
                                                      chessboard_size=(9, 6),
                                                      display=False,
                                                      pickle_filename='calibration_points.pickle')
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                                     img_size,
                                                                     None, None)

    def save_camera_calibration_examples(self, example_path='camera_cal/calibration1.jpg'):
        img = mpimg.imread(example_path)
        self.img_size = img.shape[:2]
        self.ensure_calibrated()
        print("Saving camera undistortion example")
        mpimg.imsave('output_images/camera_1_original.jpg', img)
        out = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        mpimg.imsave('output_images/camera_2_undistorted.jpg', out)

    def ensure_calibrated(self):
        if self.mtx is None:
            self.camera_calibration(self.img_size)

    def load(self, img):
        self.img = img  # keep the original image with the process
        self.img_size = self.img.shape[:2]

    def save(self, img, step, name):
        out_file_template = "output_images/{prefix}_step{step}_{name}.jpg"
        filename = out_file_template.format(step=step,
                                            name=name,
                                            prefix=self.output_prefix
                                            )
        print("Saving step: %s" % filename)
        mpimg.imsave(filename, img)

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
        if self.save_steps: self.save(img, step, 'original')

        if plot_steps:
            f, axes = plt.subplots(len(processes) + 2)
            f.subplots_adjust(hspace=0)
            axes[step].imshow(img)

        if 'undistort' in processes:
            self.ensure_calibrated()
            self.undist = out = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
            step += 1
            if self.save_steps: self.save(out, step, 'undistorted')
            if plot_steps:
                axes[step].imshow(out)

        if 'birdeye' in processes:
            dstbox = get_dstbox(asphalt_box, shape=self.img_size)
            srcbox_xy = np.array([(x, y) for y, x in asphalt_box])
            dstbox_xy = np.array([(x, y) for y, x in dstbox])

            # let's get the 2 matrix to get the birdeye and back
            self.M = cv2.getPerspectiveTransform(srcbox_xy, dstbox_xy)
            self.Minv = cv2.getPerspectiveTransform(dstbox_xy,
                                                    srcbox_xy)

            if plot_steps:
                for x, y in asphalt_box:
                    axes[step].plot(y, x, 'o', color='red', markersize=6)

            self.warped = out = cv2.warpPerspective(out, self.M,
                                                    self.img_size[::-1],
                                                    flags=cv2.INTER_LINEAR)
            step += 1
            if self.save_steps: self.save(out, step, 'birdeye')
            if plot_steps:
                axes[step].imshow(self.warped)
                step += 1
                axes[step].imshow(out)

        if 'binary' in processes:
            self.binary = out = binary(out)
            step += 1
            if self.save_steps: self.save(out, step, 'binary')
            if plot_steps:
                axes[step].imshow(out)

        if 'lines' in processes:
            # get the lines points
            y, leftx, rightx, extra = get_line_points(out)
            # get the curvature radius (in meters)
            self.radiuses = radius_in_meters(y, leftx, rightx)

            lane_center = statistics.mean([extra['left_pos'], extra['right_pos']])
            self.distance_from_center = (self.img_size[1] // 2 - lane_center) * self.XM_PER_PIX

            gray = cv2.cvtColor(self.warped, cv2.COLOR_RGB2GRAY)
            if self.save_steps:
                step += 1
                warped_lines = line_on_the_road(gray,
                                                self.undist,
                                                self.Minv,
                                                self.img,
                                                y, leftx, rightx,
                                                unwarp=False)
                self.save(warped_lines, step, 'warped_detection')
            step += 1
            out = line_on_the_road(gray,
                                   self.undist,
                                   self.Minv,
                                   self.img,
                                   y, leftx, rightx,
                                   )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(out, "radii {:.0f}m {:.0f}m".format(*self.radiuses), (50, 50),
                        font, 1, (255, 100, 100), 2, cv2.LINE_AA)
            cv2.putText(out, "center shift {:.2f}m".format(self.distance_from_center), (50, 85),
                        font, 1, (255, 100, 100), 2, cv2.LINE_AA)
            if self.save_steps: self.save(out, step, 'detection')
        return out


def run_single(filename):
    img = mpimg.imread(filename)
    pipeline = ProcessPipeline()

    extra_save = False
    if extra_save:  # set to True to enable extra output_images saves
        pipeline.save_steps = True
        pipeline.save_camera_calibration_examples()

    pipeline.output_prefix = os.path.splitext(os.path.basename(filename))[0]

    out = pipeline.process(img)

    plt.imshow(out)
    plt.show()


def run_video(filename='video/project_video.mp4'):
    pipeline = ProcessPipeline()
    pipeline.output_prefix = os.path.splitext(os.path.basename(filename))[0]

    def process_image(image):
        out = pipeline.process(image)
        return out

    clip1 = VideoFileClip(filename)
    white_clip = clip1.fl_image(process_image)
    output_video_filename = 'video/{prefix}_output.mp4'.format(prefix=pipeline.output_prefix)
    white_clip.write_videofile(output_video_filename, audio=False)


if __name__ == '__main__':
    run_single('test_images/test5.jpg')
    # run_video()
