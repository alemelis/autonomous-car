#include <iostream>
#include "tools.h"

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
