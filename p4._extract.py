
# coding: utf-8

# ## Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# 
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# In[21]:

# Let's import what we need, so we won't bother later
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle


# ## First, I'll compute the camera calibration using chessboard images
# 
# We define a function, that get the chessboard size, and keep the result in a pickle file

# In[22]:


def get_calibration_points(images, chessboard_size=(9,6), display=False, pickle_filename=None):
    try:
        # get the camera calibration from pickled file if available
        # otherwise compute them
        with open(pickle_filename, 'rb') as f:
            objpoints, imgpoints = pickle.load(f)
            return objpoints, imgpoints
    except FileNotFoundError:
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboard_size[1]*chessboard_size[0],3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size,None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                if display:
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
                    cv2.imshow('img',img)
                    cv2.waitKey(500)
        if display:
            cv2.destroyAllWindows()
        if pickle_filename:
            pickle.dump( (objpoints, imgpoints), open(pickle_filename, "wb" ) )
        return objpoints, imgpoints


# *** Let's start ***

images = glob.glob('camera_cal/calibration*.jpg')  # Make a list of calibration images
objpoints, imgpoints = get_calibration_points(images, chessboard_size=(9,6), display=True,
                                             pickle_filename = 'calibration_points.pickle')


# ## And so on and so forth...

# In[ ]:

