#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double kp, double ki, double kd,
	double p_e, double i_e, double d_e) {
	Kp = kp;
	Ki = ki;
	Kd = kd;
	p_error = p_e;
	i_error = i_e;
	d_error = d_e;
}

void PID::UpdateError(double cte) {
	d_error = cte - p_error;
	p_error = cte;
	i_error += cte;
}

double PID::TotalError() {
	return -(Kp*p_error + Kd*d_error + Ki*i_error);
}

