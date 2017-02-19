# Finding Lane Lines on the Road

#### Pipeline

1. __open image__

  ![img](imgs/ll1.png)

2. __convert to grayscale__

  ![img](imgs/ll2.png)

3. __Gaussian blur__

  ![img](imgs/ll3.png)

4. __Canny filter to find edges__

  ![img](imgs/ll4.png)

5. __Mask area of interest__

  ![img](imgs/ll5.png)

6. __Hough transform__

  ![img](imgs/ll6.png)

7. __Increase lines thickness__

  ![img](imgs/ll7.png)

8. __Detect blobs and fit lines to contours__

  ![img](imgs/ll8.png)

#### Test on videos
* [Easy](https://youtu.be/Gi0zUj4NUMM): white lines on straight road.
* [Tricky](https://youtu.be/CcpD8CPBqBs): one continuous yellow line and a dashed white line in a slightly bending road.
* [Challenging](https://youtu.be/I_LWQKc0uWc): dashed and continuous lines on a bend with shadows and color changing pavement.
