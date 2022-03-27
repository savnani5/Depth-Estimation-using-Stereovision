import math
import cv2
import random as rd
import numpy as np  


def draw_keypoints_and_match(img1, img2):
    """This function is used for finding keypoints and dercriptors in the image and 
        find best matches using brute force/FLANN based matcher."""

    # Note : Can use sift too to improve feature extraction, but it can be patented again so it could brake the code in future!
    # Note: ORB is not scale independent so number of keypoints depend on scale
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=10000)
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    #________________________Brute Force Matcher_____________________________
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # print(len(matches))
    
    # Select first 30 matches.
    final_matches = matches[:30]

    #___________________________________________________________________________

    #________________________FLANN based Matcher________________________________
    # FLANN_INDEX_LSH = 6
    # index_params = dict(
    #     algorithm=FLANN_INDEX_LSH,
    #     table_number=6,  # 12
    #     key_size=12,  # 20
    #     multi_probe_level=1,
    # )  # 2
    # search_params = dict(checks=50)  # or pass empty dictionary
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # flann_match_pairs = flann.knnMatch(des1, des2, k=2)
    
    # # Filter matches using the Lowe's ratio test
    # ratio_threshold = 0.3
    # filtered_matches = []
    # for m, n in flann_match_pairs:
    #     if m.distance < ratio_threshold * n.distance:
    #         filtered_matches.append(m)
    
    # print("FMatches", len(filtered_matches))
    # final_matches =  filtered_matches[:100]
    #___________________________________________________________________________


    # Draw keypoints
    img_with_keypoints = cv2.drawMatches(img1,kp1,img2,kp2,final_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("images_with_matching_keypoints.png", img_with_keypoints)

    # Getting x,y coordinates of the matches
    list_kp1 = [list(kp1[mat.queryIdx].pt) for mat in final_matches] 
    list_kp2 = [list(kp2[mat.trainIdx].pt) for mat in final_matches]

    return list_kp1, list_kp2




def calculate_F_matrix(list_kp1, list_kp2):
    """This function is used to calculate the F matrix from a set of 8 points using SVD.
        Furthermore, the rank of F matrix is reduced from 3 to 2 to make the epilines converge."""

    A = np.zeros(shape=(len(list_kp1), 9))

    for i in range(len(list_kp1)):
        x1, y1 = list_kp1[i][0], list_kp1[i][1]
        x2, y2 = list_kp2[i][0], list_kp2[i][1]
        A[i] = np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

    U, s, Vt = np.linalg.svd(A)
    F = Vt[-1,:]
    F = F.reshape(3,3)
   
    # Downgrading the rank of F matrix from 3 to 2
    Uf, Df, Vft = np.linalg.svd(F)
    Df[2] = 0
    s = np.zeros((3,3))
    for i in range(3):
        s[i][i] = Df[i]

    F = np.dot(Uf, np.dot(s, Vft))
    return F

def RANSAC_F_matrix(list_of_cood_list):
    """This method is used to shortlist the best F matrix using RANSAC based on the number of inliers."""
    
    list_kp1 = list_of_cood_list[0]
    list_kp2 = list_of_cood_list[1]
    pairs = list(zip(list_kp1, list_kp2))  
    max_inliers = 20
    threshold = 0.05  # Tune this value
  
    for i in range(1000):
        pairs = rd.sample(pairs, 8)  
        rd_list_kp1, rd_list_kp2 = zip(*pairs) 
        F = calculate_F_matrix(rd_list_kp1, rd_list_kp2)
        
        tmp_inliers_img1 = []
        tmp_inliers_img2 = []

        for i in range(len(list_kp1)):
            img1_x = np.array([list_kp1[i][0], list_kp1[i][1], 1])
            img2_x = np.array([list_kp2[i][0], list_kp2[i][1], 1])
            distance = abs(np.dot(img2_x.T, np.dot(F,img1_x)))
            # print(distance)

            if distance < threshold:
                tmp_inliers_img1.append(list_kp1[i])
                tmp_inliers_img2.append(list_kp2[i])

        num_of_inliers = len(tmp_inliers_img1)
        
        # if num_of_inliers > inlier_count:
        #     inlier_count = num_of_inliers
        #     Best_F = F

        if num_of_inliers > max_inliers:
            print("Number of inliers", num_of_inliers)
            max_inliers = num_of_inliers
            Best_F = F
            inliers_img1 = tmp_inliers_img1
            inliers_img2 = tmp_inliers_img2
            # print("Best F matrix", Best_F)

    return Best_F


def calculate_E_matrix(F, K1, K2):
    """Calculation of Essential matrix"""
    
    E = np.dot(K2.T, np.dot(F,K1))
    return E


def extract_camerapose(E):
    """This function extracts all the camera pose solutions from the E matrix"""

    U, s, Vt = np.linalg.svd(E)
    W = np.array([[0,-1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    C1, C2 = U[:, 2], -U[:, 2]
    R1, R2 = np.dot(U, np.dot(W,Vt)), np.dot(U, np.dot(W.T, Vt))
    # print("C1", C1, "\n", "C2", C2, "\n", "R1", R1, "\n", "R2", R2, "\n")
    
    camera_poses = [[R1, C1], [R1, C2], [R2, C1], [R2, C2]]
    return camera_poses
    

def disambiguate_camerapose(camera_poses, list_kp1):
    """This fucntion is used to find the correct camera pose based on the chirelity condition from all 4 solutions."""

    max_len = 0
    # Calculating 3D points 
    for pose in camera_poses:

        front_points = []        
        for point in list_kp1:
            # Chirelity check
            X = np.array([point[0], point[1], 1])
            V = X - pose[1]
            
            condition = np.dot(pose[0][2], V)
            if condition > 0:
                front_points.append(point)    

        if len(front_points) > max_len:
            max_len = len(front_points)
            best_camera_pose =  pose
    
    return best_camera_pose
    

def drawlines(img1src, img2src, lines, pts1src, pts2src):
    """This fucntion is used to visualize the epilines on the images
        img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines """
    r, c = img1src.shape
    img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
    img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
    
    return img1color, img2color

