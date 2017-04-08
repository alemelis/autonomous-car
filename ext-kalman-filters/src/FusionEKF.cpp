#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_      = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0,      0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0,      0,
              0,    0.0009, 0,
              0,    0,      0.09;

  //measurement matrix - laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  //state covariance matrix P
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0,    0,
             0, 1, 0,    0,
             0, 0, 1000, 0,
             0, 0, 0,    1000;

  //the initial transition matrix F_
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::InitialiseState(const MeasurementPackage &measurement_pack){
  /**
  TODO:
    * Initialize the state ekf_.x_ with the first measurement.
    * Create the covariance matrix.
    * Remember: you'll need to convert radar from polar to cartesian coordinates.
  */

  ekf_.x_ = VectorXd(4);
  ekf_.Q_ = MatrixXd(4, 4);

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    /**
    Convert radar from polar to cartesian coordinates and initialize state.
    */
    double rho = measurement_pack.raw_measurements_[0];
    double phi = measurement_pack.raw_measurements_[1];
    ekf_.x_ << rho*cos(phi), rho*sin(phi),0,0;
  }
  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    ekf_.x_ << measurement_pack.raw_measurements_[0],
               measurement_pack.raw_measurements_[1], 0, 0;
  }

  // initialise timestamp_
  previous_timestamp_ = measurement_pack.timestamp_;

  is_initialized_ = true;
  return;
}

double FusionEKF::ComputeDeltaT(long new_timestamp){
  double dt = (new_timestamp - previous_timestamp_)*1e-6;
  return dt;
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    InitialiseState(measurement_pack);
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  // compute delta t
  double dt = ComputeDeltaT(measurement_pack.timestamp_);
 	previous_timestamp_ = measurement_pack.timestamp_;

 	// update F
 	ekf_.F_(0, 2) = dt;
 	ekf_.F_(1, 3) = dt;

 	// update Q
  float dt2 = dt * dt;
  float dt3 = dt2 * dt * 0.5;
  float dt4 = dt3 * dt * 0.5;
  float noise_ax = 9;
  float noise_ay = 9;

 	ekf_.Q_ <<  dt4*noise_ax, 0,            dt3*noise_ax, 0,
 			        0,            dt4*noise_ay, 0,            dt3*noise_ay,
 			        dt3*noise_ax, 0,            dt2*noise_ax, 0,
 			        0,            dt3*noise_ay, 0,            dt2*noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    Hj_= tools.CalculateJacobian(ekf_.x_);
    if (!Hj_.isZero()){
	    ekf_.Init(ekf_.x_ , ekf_.P_ , ekf_.F_ , Hj_, R_radar_, ekf_.Q_ );
    	ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    }
  } else {
    // Laser updates
    ekf_.Init(ekf_.x_ , ekf_.P_ , ekf_.F_ , H_laser_, R_laser_, ekf_.Q_ );
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  // cout << "x_ = " << ekf_.x_ << endl;
  // cout << "P_ = " << ekf_.P_ << endl;
}
