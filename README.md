# Depth-estimation-using-stereovision
In this project, we are going to implement the concept of Stereo Vision. We
will be given 3 different datasets, each of them contains 2 images of the same
scenario but taken from two different camera angles. By comparing the
information about a scene from 2 vantage points, we can obtain the 3D
information by examining the relative positions of objects.

## Input
[Input Data](https://drive.google.com/drive/folders/1D70GsEdZhFj3jked1Vh-x61ILViSFlOp?usp=sharing)

## Output
[Output Data](https://drive.google.com/drive/folders/1D70GsEdZhFj3jked1Vh-x61ILViSFlOp?usp=sharing)

## How to Run the Code
1) Change the dataset directory in *main.py* file:

  ```PY
   img1 = cv2.imread(f"Dataset{number}/im0.png", 0)
   ```

   ```PY
   img2 = cv2.imread(f"Dataset{number}/im1.png", 0)
   ```
2) Run the following command:
  
   ```
   python main.py
   ```

## References
1) https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
2) https://cmsc733.github.io/2019/proj/p3/#estE
3) https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html
4) https://stackoverflow.com/questions/30716610/how-to-get-pixel-coordinates-from-feature-matching-in-opencv-python
5) https://stackoverflow.com/questions/27856965/stereo-disparity-map-generation
6) https://stackoverflow.com/questions/36172913/opencv-depth-map-from-uncalibrated-stereo-system
7) https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html
8) https://pramod-atre.medium.com/disparity-map-computation-in-python-and-c-c8113c63d701
9) https://stackoverflow.com/questions/46689428/convert-np-array-of-type-float64-to-type-uint8-scaling-values/46689933
10) https://stackoverflow.com/questions/59478962/how-to-convert-a-grayscale-image-to-heatmap-image-with-python-opencv
11) https://www.youtube.com/watch?v=KOSS24P3_fY
