# PID controller

---

[//]: # (Image References)

[image0]: ./imgs/p.png "proportional"
[image1]: ./imgs/gif1.gif "largeP"
[image2]: ./imgs/gif2.gif "smallP"
[image3]: ./imgs/pi.png "pintegral"
[image4]: ./imgs/gif3.gif "pi"
[image5]: ./imgs/pid.png "piderivative"
[image6]: ./imgs/gif4.gif "pid"
[image7]: ./imgs/gif5.gif "pidbend"

## Installation

Instructions can be found in the [original repo](https://github.com/udacity/CarND-PID-Control-Project).

## Reflection

This project aim was to code (in C++) a proportional integral derivative (PID) controller for driving along a simulated track. The controller is based on the continuous calculation of the error between the actual car position and the desired position, the cross-track error (CTE). The CTE is used to correct the car position based on three terms:

**P**: steer the car in proportion to CTE as

![alt][image0]

where `tau_P` is a constant value that indicates how much the steering angle `alpha` should be increased/decreased any time the `CTE` is computed. A controller with only the P parameter easily overshoots the desired trajectory. A large P parameter value (`tau_P=1.`) makes the car heavily oscillate even on straight road parts

![alt][image1]

A smaller value (`tau_P=.1`) makes the car unstable later on when a bend occurs

![alt][image2]

**I**: the integral term is computed as the sum of all CTEs previously observed

![alt][image3]

Hence, the I term takes into account the magnitude and the history of the track error. When the error starts accumulating, the I term becomes predominating and increases the steering angle. Therefore the I term magnitude should be smaller than P term one. Even with `tau_I=.001`, the car is almost immediately out of control

![alt][image4]

**D**: the derivative term takes into consideration two consecutive CTEs and, by considering a unit time step, computes the CTE time derivative as a the difference

![alt][image5]

The effect of the derivative term is to make the car gracefully approach the target trajectory. The magnitude of `tau_D` should be higher than the other two terms ones in order to damp the excessive oscillations and the accumulation of CTE. By setting `tau_D=3`, the car is more stable on the straight track and manages to approach the first bend without crossing the lane lines

![alt][image6]

The car even follows the bends in the most difficult tract of the circuit

![alt][image7]

### Parameters selection

The final values for each PID term (`tau_P=0.25, tau_I=0.0025, tau_D=10`) were found by trial and error by following my understanding of the algorithm. The approach I followed is similar to the one described in the above section (the iterations are show as comments in `src/main.cpp`). I then tried to fine tune the parameters to achieve a smoother driving in bends. A video of the final run can be found [here](https://youtu.be/KujZrGhdTf4)
