#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
  double px = x_[0];
  double py = x_[1];
  double vx = x_[2];
  double vy = x_[3];

	double pxpy=sqrt(px*px+py*py);
  	if (pxpy<=0.001)
  	{
  	  std::cout << "ERROR" << std::endl;
  	  pxpy=0.00001;
  	}
  VectorXd hxprime = VectorXd(3);
	hxprime(0) = pxpy;
	hxprime(1) = atan2(py,px);
	hxprime(2) = (px*vx+py*vy)/pxpy;

	VectorXd y = z - hxprime;
	// need to check to see if we blow out the -pi/pi
	if (y(1) > M_PI || y(1) < -M_PI){
		std::cout << "ERROR outside pi/-pi range " << std::endl;
		std::cout << y(1) << std::endl;
		int res = floor(y(1)/M_PI);
		y(1)=y(1)/res;
		std::cout << y(1) << std::endl;
	}

  MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd K =  P_ * Ht * Si;
	MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());

  //new state
	x_ = x_ + (K * y);
	P_ = (I - K * H_) * P_;
}
