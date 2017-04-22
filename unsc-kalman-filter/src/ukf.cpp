#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
using std::vector;

#pragma clean diagnostic push
#pragma ide diagnostic ignored "IncompatibleTypes"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  time_us_ = 0.0;

  n_x_ = 5;
  n_aug_ = 7;
  n_sig_ = 2 * n_aug_ + 1;
  n_z_ = 3;

  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);
  x_aug_ = VectorXd(n_aug_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  Xsig_pred_ = MatrixXd(n_x_, n_sig_);
  Xsig_aug_ = MatrixXd(n_aug_, n_sig_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  weights_ = VectorXd(n_sig_);

  n_z_laser_ = 2;
  n_z_radar_ = 3;

  R_lidar_ = MatrixXd::Zero(n_z_laser_, n_z_laser_);
  R_lidar_ << std_laspx_*std_laspx_, 0,
              0, std_laspy_*std_laspy_;

  R_radar_ = MatrixXd::Zero(n_z_radar_, n_z_radar_);
  R_radar_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0, std_radrd_*std_radrd_;
}

UKF::~UKF() {}

void UKF::Initialise(MeasurementPackage measurement_pack) {

  float px, py, vx, vy, v, yaw, yaw_rate;

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    float range = measurement_pack.raw_measurements_[0];
    float bearing = measurement_pack.raw_measurements_[1];
    float range_rate = measurement_pack.raw_measurements_[2];

    px = range * cos(bearing);
    py = range * sin(bearing);
    vx = range_rate * cos(bearing);
    vy = range_rate * sin(bearing);
    v = sqrt(pow(vx, 2) + pow(vy, 2));
    yaw = vx == 0 ? 0 : vy / vx;
    yaw_rate = 0;
  }
  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    px = measurement_pack.raw_measurements_[0];
    py = measurement_pack.raw_measurements_[1];
    v = 0;
    yaw = 0;
    yaw_rate = 0;
  }

  x_ << px, py, v, yaw, yaw_rate;

  float wi = 0.5/(lambda_ + n_aug_);
  weights_(0) = lambda_*2*wi;
  for (int i=1; i<2*n_aug_+1; i++) {
      weights_(i) = wi;
  }

  time_us_ = measurement_pack.timestamp_;
  is_initialized_ = true;
  return;
}

float UKF::CalculateDeltaT(long new_timestamp) {
  float delta_t = (new_timestamp - time_us_) / 1000000.0;
  return delta_t;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  // Initialization
  if (!is_initialized_) {
    Initialise(meas_package);
  }

  // find delta_t
  float delta_t = CalculateDeltaT(meas_package.timestamp_);
  time_us_ = meas_package.timestamp_;

  // Prediction step.
  Prediction(delta_t);

  // Update
  if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
}

void UKF::GenerateAugmentedSigmaPoints() {

  MatrixXd Q = MatrixXd(n_aug_ - n_x_, n_aug_ - n_x_);
  Q << std_a_*std_a_, 0,
       0, std_yawdd_*std_yawdd_;

  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(n_aug_ - n_x_, n_aug_ - n_x_) = Q;

  VectorXd x_aug = VectorXd(n_aug_);
  x_aug << x_, 0, 0;

  P_aug = (lambda_+n_aug_)*P_aug;
  P_aug = P_aug.llt().matrixL();

  Xsig_aug_.col(0) << x_aug;
  for (int i=1; i<=7; i++) {
      Xsig_aug_.col(i) << x_aug+P_aug.col(i-1);
  }
  for (int i=1; i<=7; i++) {
      Xsig_aug_.col(i+7) << x_aug-P_aug.col(i-1);
  }
}

// this used to work for the online assignment but it is not giving
// the same result of the site one. I am using the function from the lesson...
// void UKF::PredictSigmaPoints(double delta_t) {
//
//   for (int j=0; j<2*n_aug_+1; j++) {
//     float px = Xsig_aug_(0,j);
//     float py = Xsig_aug_(1,j);
//     float v = Xsig_aug_(2,j);
//     float psi = Xsig_aug_(3,j);
//     float psi_dot = Xsig_aug_(4,j);
//     float nu_a = Xsig_aug_(5,j);
//     float nu_psi_ddot = Xsig_aug_(6,j);
//
//     float cos_psi = cos(psi);
//     float sin_psi = sin(psi);
//     float half_dt_sq = 0.5*delta_t*delta_t;
//
//     if (psi_dot < 1e-3) {
//         Xsig_pred_.col(j) << px + v*cos_psi*delta_t + half_dt_sq*cos_psi*nu_a,
//                             py + v*sin_psi*delta_t + half_dt_sq*sin_psi*nu_a,
//                             v  + delta_t*nu_a,
//                             psi+ half_dt_sq*nu_psi_ddot,
//                             delta_t*nu_psi_ddot;
//     }
//     else {
//         Xsig_pred_.col(j) << px + v/psi_dot*( sin(psi+delta_t*psi_dot)-sin_psi) + half_dt_sq*cos_psi*nu_a,
//                             py + v/psi_dot*(-cos(psi+delta_t*psi_dot)+cos_psi) + half_dt_sq*sin_psi*nu_a,
//                             v  + delta_t*nu_a,
//                             psi+ psi_dot*delta_t + half_dt_sq*nu_psi_ddot,
//                             psi_dot + delta_t*nu_psi_ddot;
//     }
//   }
// }

void UKF::PredictSigmaPoints(double dt) {
  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v/yawd * ( sin (yaw + yawd*dt) - sin(yaw));
      py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*dt) );
    }
    else {
      px_p = p_x + v*dt*cos(yaw);
      py_p = p_y + v*dt*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*dt;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*dt*dt * cos(yaw);
    py_p = py_p + 0.5*nu_a*dt*dt * sin(yaw);
    v_p = v_p + nu_a*dt;

    yaw_p = yaw_p + 0.5*nu_yawdd*dt*dt;
    yawd_p = yawd_p + nu_yawdd*dt;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

void UKF::PredictMeanAndCovariance() {

  // Predict state mean.
  x_ << (Xsig_pred_ * weights_);

  P_.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // normalise angle
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  GenerateAugmentedSigmaPoints();
  PredictSigmaPoints(delta_t);
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z_laser_, n_sig_);
  VectorXd z_pred = Zsig * weights_;

  MatrixXd S = MatrixXd(n_z_laser_, n_z_laser_);
  S.fill(0.0);
  VectorXd Zz = VectorXd(n_x_);
  Zz.fill(0.0);
  MatrixXd Tc = MatrixXd(n_x_, n_z_laser_);
  Tc.fill(0.0);
  VectorXd Xx = VectorXd(n_x_);
  Xx.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++) {
      Xx = Xsig_pred_.col(i) - x_;
      Zz = Zsig.col(i)-z_pred;
      S = S + weights_(i)*Zz*Zz.transpose();
      Tc = Tc + weights_(i)*Xx*Zz.transpose();
  }
  S = S + R_lidar_;
  MatrixXd K = Tc*S.inverse();

  VectorXd z = VectorXd(n_z_laser_);
  z <<  meas_package.raw_measurements_[0],
        meas_package.raw_measurements_[1];
  VectorXd z_innov = z - z_pred;

  x_ = x_ + K * z_innov;
  P_ = P_ - K*S*K.transpose();

  NIS_laser_ = z_innov.transpose() * S.inverse() * z_innov;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  MatrixXd Zsig = MatrixXd(n_z_radar_, n_sig_);
  VectorXd z_pred = VectorXd(n_z_radar_);
  MatrixXd R = MatrixXd(n_z_radar_, n_z_radar_);

  //go to measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;
    double zero = 1e-6;

    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);
    Zsig(1,i) = atan2(p_y, p_x < zero ? zero : p_x);
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);
  }

  z_pred = Zsig * weights_;

  MatrixXd S = MatrixXd(n_z_radar_, n_z_radar_);
  S.fill(0.0);
  VectorXd Zz = VectorXd(n_x_);
  Zz.fill(0.0);
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);
  Tc.fill(0.0);
  VectorXd Xx = VectorXd(n_x_);
  Xx.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++) {
      Xx = Xsig_pred_.col(i) - x_;
      Zz = Zsig.col(i)-z_pred;
      S = S + weights_(i)*Zz*Zz.transpose();
      Tc = Tc + weights_(i)*Xx*Zz.transpose();
  }
  S = S + R_radar_;
  MatrixXd K = Tc*S.inverse();

  VectorXd z = VectorXd(n_z_radar_);
  z <<  meas_package.raw_measurements_[0],
        meas_package.raw_measurements_[1],
        meas_package.raw_measurements_[2];
  VectorXd z_innov = z - z_pred;
  //
  // z_innov(1) = normalize_angle(z_innov(1));

  x_ = x_ + K * z_innov;
  P_ = P_ - K*S*K.transpose();

  NIS_radar_ = z_innov.transpose() * S.inverse() * z_innov;
}
