import numpy as np
import cv2

def disparity_to_depth(baseline, f, img):
    """This is used to compute the depth values from the disparity map"""

    # Assumption image intensities are disparity values (x-x') 
    depth_map = np.zeros((img.shape[0], img.shape[1]))
    depth_array = np.zeros((img.shape[0], img.shape[1]))

    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            depth_map[i][j] = 1/img[i][j]
            depth_array[i][j] = baseline*f/img[i][j]
            # if math.isinf(depth_map[i][j]):
            #     depth_map[i][j] = 1

    return depth_map, depth_array
