#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */

  MatrixXd Hj(3,4);
  
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//check division by zero
	float px2py2 = px*px + py*py;
	if(px2py2 == 0){
	    cout << "ERROR: division by zero!";
	    return Hj;
	}

	//compute the Jacobian matrix
	float sqrtPx2py = pow(px2py2, 0.5);
	float sqrt3px2py = pow(px2py2, 1.5);

	Hj(0,0) = px/sqrtPx2py;
	Hj(0,1) = py/sqrtPx2py;
	Hj(0,2) = 0.;
	Hj(0,3) = 0.;

	Hj(1,0) = -py/sqrtPx2py;
	Hj(1,1) =  px/sqrtPx2py;
	Hj(1,2) =  0.;
	Hj(1,3) =  0.;

	Hj(2,0) =  px*(vx*py-vy*px)/sqrt3px2py;
	Hj(2,1) = -py*(vx*py-vy*px)/sqrt3px2py;
	Hj(2,2) =  px/sqrtPx2py;
	Hj(2,3) =  py/sqrtPx2py;

	return Hj;
}
