#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  bool is_initialized_;
  bool use_laser_;
  bool use_radar_;
  long long time_us_;

  int n_x_;
  int n_aug_;
  int n_sig_;
  int n_z_;

  double lambda_;

  VectorXd x_;
  VectorXd x_aug_;

  MatrixXd P_;

  MatrixXd Xsig_pred_;
  MatrixXd Xsig_aug_;

  double std_a_;
  double std_yawdd_;
  double std_laspx_;
  double std_laspy_;
  double std_radr_;
  double std_radphi_;
  double std_radrd_ ;

  VectorXd weights_;

  double NIS_radar_;
  double NIS_laser_;

  VectorXd weights;

  int n_z_laser_;
  int n_z_radar_;

  MatrixXd R_lidar_;
  MatrixXd R_radar_;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);

  void GenerateAugmentedSigmaPoints();
  void SigmaPointPrediction();
  void PredictMeanAndCovariance();
  void PredictRadarMeasurement();
  void PredictSigmaPoints(double dt);

  /**
   * Updates the state given the measruement vector z, z sigma points, predicted measurement S.
   * @param z
   * @param Zsig
   * @param z_pred
   * @param S
   */
  void UpdateState(VectorXd& z, MatrixXd& Zsig, VectorXd& z_pred, MatrixXd& S);




  /**
   * Initializes variables for the prediction - update steps to follow.
   */
  void Initialise(MeasurementPackage meas_package);

  float CalculateDeltaT(long new_timestamp);

};

#endif /* UKF_H */
