import numpy as np

# on this image I get the coordinate of 4 points that form a rectangle on the asphalt:
# one close point for each lane
# one far point for each lane

asphalt_box = np.array(
    [
        (690, 256),  # left lane, close
        (460, 580),  # left lane, far

        (690, 1058),  # right lane, close
        (460, 705),  # right lane, far

    ], dtype=np.float32
)


def get_dstbox(bbox, shape):
    """ Build the destination box so that the srcbox x stays on bottom of image 
        while the top stays on top over them
    """
    imgmaxy, imgmaxx = shape
    minx, maxx = min(x for y, x in bbox), max(x for y, x in bbox)
    miny, maxy = min(y for y, x in bbox), max(y for y, x in bbox)
    return np.float32([
        (imgmaxy, minx),  # bottom left
        (0, minx),  # top left
        (imgmaxy, maxx),  # bottom right
        (0, maxx),  # top right
    ])



