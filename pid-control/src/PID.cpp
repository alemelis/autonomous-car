#include "PID.h"
#include <iostream>
using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double kp, double ki, double kd) {
	Kp = kp; // proportional
	Ki = ki; // integral
	Kd = kd; // derivative

	p_error = 1.;
	i_error = 1.;
	d_error = 1.;
}

void PID::UpdateError(double cte) {
	d_error = cte - p_error;
	p_error = cte;
	i_error += cte;
}

double PID::TotalError() {
	return -(Kp*p_error + Kd*d_error + Ki*i_error);
}

// Twiddle algorithm ported straight from lecture notes.
// Not used for parameters optimisation though.
void PID::Twiddle() {
	double tol = 0.2;
	double best_err = TotalError();
	int n = 0;

	while (p_error+i_error+d_error > tol) {
		
		// for i in range(len(p))
		for (int i=1; i<=3; i++){
			
			// p[i] += dp[i]
			switch (i) {
				case 1:
					Kp += p_error;
				case 2:
					Ki += i_error;
				case 3:
					Kd += d_error;
			}

			double err = TotalError();

			if (err < best_err) {

				best_err = err;

				// dp[i] *= 1.1
				switch (i) {
					case 1:
						p_error *= 1.1;
					case 2:
						i_error *= 1.1;
					case 3:
						d_error *= 1.1;
				}
			}
			else {
				// p[i] -= 2*dp[i]
				switch (i) {
					case 1:
						Kp -= 2*p_error;
					case 2:
						Ki -= 2*i_error;
					case 3:
						Kd -= 2*d_error;
				}
				err = TotalError();

				if (err < best_err) {
					best_err = err;

					// dp[i] *= 1.1
					switch (i) {
						case 1:
							p_error *= 1.1;
						case 2:
							i_error *= 1.1;
						case 3:
							d_error *= 1.1;
					}
				}
				else {
					// p[i] += dp[i]
					// dp[i] *= 0.9
					switch (i) {
						case 1:
							Kp += p_error;
							p_error *= 0.9;
						case 2:
							Ki += i_error;
							i_error *= 0.9;
						case 3:
							Kd += d_error;
							d_error *= 0.9;
					}
				}
			}
		}
		n += 1;

		// debug output
		std::cout << n << " " << p_error << " " << i_error << " " << d_error << std::endl;
		std::cout << n << " " << Kp << " " << Ki << " " << Kd << std::endl;
	}
}
