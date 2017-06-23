// In this quiz you'll implement the global kinematic model.
#include <math.h>
#include <iostream>
#include "Eigen-3.3/Eigen/Core"

//
// Helper functions
//
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

const double Lf = 2;

// TODO: Implement the global kinematic model.
// Return the next state.
//
// NOTE: state is [x, y, psi, v]
// NOTE: actuators is [delta, a]
Eigen::VectorXd globalKinematic(Eigen::VectorXd state,
                                Eigen::VectorXd actuators, double dt) {

  double x, y, psi, v, delta, a;

  x = state(0);
  y = state(1);
  psi = state(2);
  v = state(3);

  delta = actuators(0);
  a = actuators(1);

  double x_t1, y_t1, psi_t1, v_t1;

  x_t1 = x + v*cos(psi)*dt;
  y_t1 = y + v*sin(psi)*dt;
  psi_t1 = psi + v*delta*dt/Lf;
  v_t1 = v + a*dt;

  Eigen::VectorXd next_state(state.size());
  next_state << x_t1, y_t1, psi_t1, v_t1;
  return next_state;
}

int main() {
  // [x, y, psi, v]
  Eigen::VectorXd state(4);
  // [delta, v]
  Eigen::VectorXd actuators(2);

  state << 0, 0, deg2rad(45), 1;
  actuators << deg2rad(5), 1;

  // should be [0.212132, 0.212132, 0.798488, 1.3]
  auto next_state = globalKinematic(state, actuators, 0.3);

  std::cout << next_state << std::endl;
}