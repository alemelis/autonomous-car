# Behavioral Cloning

This document discusses the dataset preparation and the model training for the project on behavioral cloning. The test runs are discussed toward the end of the document.

The proposed convolutional neural network (CNN) is a very small one as it has _41_ parameters. This idea comes from [this post](https://medium.com/@xslittlegrass/self-driving-car-in-a-simulator-with-a-tiny-neural-network-13d33b871234#.p50e94pcl) and it is discussed in the Architecture section. The CNN, made of a single 1x1 convolutional layer with two filters, performs well-enough to keep the car in track.

The major benefit of having such a small number of parameters is the short training time. Indeed, the CNN can be trained on CPU in few seconds, without the need of using GPUs or AWS cloud.

The robustness of the CNN is directly linked to the nature of the test track: the virtual track is not as complex as the real world, therefore an actual autonomous-driving CNN architecture developed to work on real images would easily result oversized to work on virtual images. Nevertheless, the small-CNN approach presents some limitations, and these are discussed at the end of the report.

## Dataset

#### Acquisition

The dataset consists of _47687_ images. These were acquired on the the simulator by using a joypad. The simulator was always used with the __Simple__ graphics quality option (see Limitations section at the end).

About two thirds of the dataset consist of images captured while driving in between the lines. The remaining images were taken starting from a side and moving towards the center. This was done to make the network learn how to recover the position if out of track.

#### Load
`model.py` requires the following folder structure
```
project_folder/
  |_ IMG/
  |_ driving_log.csv
  |_ model.py
  ...
```
The training images are stored in `IMG/` folder and the steering data is in `driving_log.csv` file.

The log file is loaded into memory and read line by line. For each line three images are loaded, one for each camera.

#### Preprocessing
 The images are preprocessed with `loadAndPreprocessImage` function:
- open and convert to numpy array;
- convert to HSV space and extract the saturation channel;
- crop sky;
- resize to `(new_cols, new_rows)` shape.

![img](http://i.imgur.com/Q7Vzod2.png)

The sky cropping is done by means of `clean_sky` parameter. A value of `50` gives good results. The cropped image is `(320,110)` in size. This is further reduced by resizing the image to `(32,11)`.

The steering angle for each image is taken from the `.csv` file. For the side cameras, the angle should be modified to compensate for the cameras' location. The parameter taking into account this offset is `stering_theta=0.3`. This value was selected after a phase of trial and error.

#### Augmentation
The dataset is doubled by flipping horizontally all the images, for a total of _95374_ samples [Ref](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.i29wbsjf3). The flipped images are associated to the original steering angle changed of sign.

## Architecture

The architecture was inspired by Mengxi Wu [post](https://medium.com/@xslittlegrass/self-driving-car-in-a-simulator-with-a-tiny-neural-network-13d33b871234#.p50e94pcl) on Medium. In the post, the proposed CNN is said _tiny_ as it contains only _63_ parameters. The CNNs analyzed during the course contain several thousands of parameters. This results in long training times when using only CPUs. Conversely, a tiny CNN can be trained in few seconds, allowing to explore faster the response of the CNN w.r.t. the various hyper-parameters.

I developed a tiny CNN consisting of one 1x1 convolutional layer with two filters activated by `relu` functions. This is followed by a max pooling layer with kernel 4x4 and stride 4x4. I used aggressive dropout (30% drop ratio) to avoid overfitting and one output neuron to predict the steering angle (as in the [nvidia paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)). The normalization is done into the model as soon as an image is passed as input.
```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
Normalization (Lambda)           (None, 11, 32, 1)     0           lambda_input_47[0][0]            
____________________________________________________________________________________________________
convolution2d_45 (Convolution2D) (None, 2, 32, 1)      24          Normalization[0][0]              
____________________________________________________________________________________________________
maxpooling2d_43 (MaxPooling2D)   (None, 2, 8, 1)       0           convolution2d_45[0][0]           
____________________________________________________________________________________________________
dropout_43 (Dropout)             (None, 2, 8, 1)       0           maxpooling2d_43[0][0]            
____________________________________________________________________________________________________
flatten_43 (Flatten)             (None, 16)            0           dropout_43[0][0]                 
____________________________________________________________________________________________________
dense_38 (Dense)                 (None, 1)             17          flatten_43[0][0]                 
====================================================================================================
Total params: 41
____________________________________________________________________________________________________
```
The resulting CNN has _41_ parameters. This is smaller than Wu's one because of the different postprocessing that brings the images size down to (32,11) pixels. Moreover, the change of kernel from 3x3 to 1x1 decreases drastically the number of parameters. I used `border_mode=same` as it results more intuitive, but I reckon that a `valid` padding would decrease even more the parameters.

#### Training

The dataset is first shuffled and then split in `train` and `test` sub-dataset. The test dataset consists of `test_samples=20` input data. The training test is further split to obtain the validation dataset. This is taken as the 10% of the training dataset.

The model is fit by using an Adam optimizer and the `mean_squared_error` as loss function. A `batch_size=128` is taken to train the model over _30_ epochs. This number of epochs results to minimize loss without overfitting.

![img](http://i.imgur.com/YuqoRgP.png)

#### Test

The trained model is tested first on the `X_test` dataset. The model predictions can be plotted against the actual values `y_test`

![img](http://i.imgur.com/LiihLAH.png)

Points lying on the _line of equality_ indicate a good agreement between predictions and real values. The model performs well as the points are scattered around the line.

### Simulator

The model weights are saved into `model.h5` and the model structure is saved into `model.json`. These two files can be used along with `drive.py` to test the model on the simulator.

`drive.py` has been modified to include the image preprocessing pipeline

```python
# Preprocess image
hsv_img = cv2.cvtColor(transformed_image_array[0], cv2.COLOR_RGB2HSV)
sky_img = hsv_img[50:,:,1]
rsz_img = cv2.resize(sky_img,(32,11)).reshape(1,11,32,1)
```
The throttle is kept constant to `0.2`. However, in the view of driving in the mountain track, the throttle should be increased to `0.3`.

#### Autonomous driving

[Video](https://youtu.be/VNUzwjJ_vEg) of the car autonomously driving on the track.

The model successfully drives the virtual car along the track. The car never leaves the track (and, most importantly, it does not wander all over the bottom of the lake!) and the ledges are not touched by the tires.

![img](gifs/bc1.gif)![img](gifs/bc2.gif)

However, the model seems to enjoy frightening the passengers as it tends to steer right at the end of the bends.

![img](gifs/bc3.gif)![img](gifs/bc4.gif)

By increasing the throttle to `0.3`, the model is able to finish also the __unseen__(!!!) mountain track ([video](https://youtu.be/Ec5RDMsQr5k)). This track is easier to process because of the continuous yellow lanes on the track sides. The car goes at full speed without touching the rocky sides.

![img](gifs/bc5.gif)
<!-- ![img](http://i.giphy.com/l0Ex993RZekjTPLjO.gif)
![img](http://i.giphy.com/26gs9aUTn2CtvnGCc.gif)
![img](http://i.giphy.com/l3q2ZHeNU0k3g1Q2s.gif) -->

#### Known limitations

* The model has been trained on images taken with a certain frame rate (related to the __Simple__ graphics options). Therefore, the tests should be done at the same frame rate. __Higher or lower fps streaming result in the model not being able to drive the car on the track.__

![img](gifs/bc6.gif)

* The model returns only the steering angle. By adding a second output neuron, we could train the model on the throttle values from the `.csv` file. However, this strategy did not succeeded as the car would need also to be trained on the current velocity. I did not find a way to combine the images from the cameras with the speed value, yet. In the forum, George [suggested](https://carnd-forums.udacity.com/questions/26220933/need-help-connecting-the-dots-between-images-steering-angles-and-throttle) to parameterize the throttle value in function of the steering angle: `throttle = (DESIRED_SPEED-float(speed))*Kp`. The car accelerates when on a straight road and decelerates while in a bend. It can be defined a constant value for the _desired speed_ and `Kp` is the proportionality constant also known as [loop gain](https://en.wikipedia.org/wiki/PID_controller). `Kp=0.5` should return good results.

### Few more ideas

* Use [ELU](https://arxiv.org/pdf/1511.07289v1.pdf) over ReLU activation function.
* [NVIDIA](https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.j1yr0so45) model;
* [Comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py) model;
* [more](http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/) and [more](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) on dropout;
* on [batch size](http://stats.stackexchange.com/questions/140811/how-large-should-the-batch-size-be-for-stochastic-gradient-descent) for stochastic gradient descent;
* models from fellow students [1](https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.5dr15may7), [2](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.ohws2l5xh) (augmentation), [3](https://medium.com/@vivek.yadav/cloning-a-car-to-mimic-human-driving-using-pretrained-vgg-networks-ac5c1f0e5076#.q6wvh54iz) (pretrained VGG net).
