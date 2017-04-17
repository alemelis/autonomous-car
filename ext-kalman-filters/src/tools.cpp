#include <iostream>
#include "tools.h"
using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  if(estimations.size()==0){
      cout << "ERROR: zero size estimation vector";
      return rmse;
  }
  //  * the estimation vector size should equal ground truth vector size
  // ... your code here
  if(estimations.size() != ground_truth.size()){
      cout << "ERROR: estimation size different than ground truth size";
      return rmse;
  }

  //accumulate squared residuals
  for(int i=0; i < estimations.size(); ++i){
        // ... your code here
        VectorXd residual = estimations[i] - ground_truth[i];
        residual = residual.array()*residual.array();
  	rmse += residual;
  }

  //calculate the mean
  // ... your code here
  rmse = rmse.array()/estimations.size();

  //calculate the squared root
  // ... your code here
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  MatrixXd Hj(3,4);

	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//check division by zero
	float px2py2 = px*px + py*py;
	if(px2py2 < 1e-5){
	    cout << "ERROR: division by zero!";
	    return Hj;
	}

	//compute the Jacobian matrix
  float c1 = px*px+py*py;
	float c2 = sqrt(c1);
	float c3 = (c1*c2);

	Hj(0,0) = px/c2;
	Hj(0,1) = py/c2;
	Hj(0,2) = 0.;
	Hj(0,3) = 0.;

	Hj(1,0) = -py/c1;
	Hj(1,1) =  px/c1;
	Hj(1,2) =  0.;
	Hj(1,3) =  0.;

	Hj(2,0) = py*(vx*py-vy*px)/c3;
	Hj(2,1) = px*(-vx*py+vy*px)/c3;
	Hj(2,2) = px/c2;
	Hj(2,3) = py/c2;

	return Hj;
}
