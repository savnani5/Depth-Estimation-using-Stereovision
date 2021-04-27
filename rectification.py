import numpy as np
import cv2

def rectification(img1, img2, pts1, pts2, F):
    """This function is used to rectify the images to make camera pose's parallel and thus make epiplines as horizontal.
        Since camera distortion parameters are not given we will use cv2.stereoRectifyUncalibrated(), instead of stereoRectify().
    """

    # Stereo rectification
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1))
    print("H1",H1)
    print("H2",H2)

    rectified_pts1 = np.zeros((pts1.shape), dtype=int)
    rectified_pts2 = np.zeros((pts2.shape), dtype=int)

    # Rectify the feature points
    for i in range(pts1.shape[0]):
        source1 = np.array([pts1[i][0], pts1[i][1], 1])
        new_point1 = np.dot(H1, source1)
        new_point1[0] = int(new_point1[0]/new_point1[2])
        new_point1[1] = int(new_point1[1]/new_point1[2])
        new_point1 = np.delete(new_point1, 2)
        rectified_pts1[i] = new_point1

        source2 = np.array([pts2[i][0], pts2[i][1], 1])
        new_point2 = np.dot(H2, source2)
        new_point2[0] = int(new_point2[0]/new_point2[2])
        new_point2[1] = int(new_point2[1]/new_point2[2])
        new_point2 = np.delete(new_point2, 2)
        rectified_pts2[i] = new_point2

    # Rectify the images and save them
    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
    
    cv2.imwrite("rectified_1.png", img1_rectified)
    cv2.imwrite("rectified_2.png", img2_rectified)
    
    return rectified_pts1, rectified_pts2, img1_rectified, img2_rectified


   
