/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 * Completed by Alessandro Melis
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;

	std::default_random_engine generator;
	std::normal_distribution<double> DistributionX(x, std[0]);
	std::normal_distribution<double> DistributionY(y, std[1]);
	std::normal_distribution<double> DistributionTheta(theta, std[2]);

	for (int i=0; i<num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = DistributionX(generator);
		particle.y = DistributionY(generator);
		particle.theta = DistributionTheta(generator);
		particle.weight = 1.;

		particles.push_back(particle);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine generator;
	std::normal_distribution<double> DistributionX(0., std_pos[0]);
	std::normal_distribution<double> DistributionY(0., std_pos[1]);
	std::normal_distribution<double> DistributionTheta(0., std_pos[2]);

	double theta_0;
	double theta_dot_dt;
	for (int i=0; i<num_particles; i++) {
		theta_0 = particles[i].theta;

		// check yaw rate != 0
		if (fabs(yaw_rate) > 1E-6){
			theta_dot_dt = yaw_rate*delta_t+theta_0;
			particles[i].x += velocity*(sin(theta_dot_dt) - sin(theta_0))/yaw_rate;
			particles[i].y += velocity*(-cos(theta_dot_dt) + cos(theta_0))/yaw_rate;
			particles[i].theta = theta_dot_dt;
		}
		else {
			particles[i].x += velocity*cos(theta_0)*delta_t;
			particles[i].y += velocity*sin(theta_0)*delta_t;
		}

		// Gaussian noise
		particles[i].x += DistributionX(generator);
		particles[i].y += DistributionY(generator);
		particles[i].theta += DistributionTheta(generator);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	double obs_pred_dist, shortest_dist;

	for (int i=0; i<observations.size(); i++) {
		shortest_dist = 1E15;
		observations[i].id = 1;
		for (int j=0; j<predicted.size(); j++) {
				obs_pred_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
				if (obs_pred_dist < shortest_dist){
					shortest_dist = obs_pred_dist;
					observations[i].id = j;
				}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	std::vector<LandmarkObs> predictions;
	for (int i = 0; i < map_landmarks.landmark_list.size(); i++) {
		Map::single_landmark_s mark = map_landmarks.landmark_list[i];

		LandmarkObs obs;
		obs.id = 1;
		obs.x = mark.x_f;
		obs.y = mark.y_f;

		predictions.push_back(obs);
	}

	for (int i=0; i<num_particles; i++) {
		std::vector<LandmarkObs> t_obs; // transformed observations
		LandmarkObs t_ob;

		for (int m=0; m<observations.size(); m++) {
			int id = 1;

			double costheta = cos(particles[i].theta);
			double sintheta = sin(particles[i].theta);

			// transform observation to map coordinates
			double x = particles[i].x;
			x += observations[m].x * costheta;
			x -= observations[m].y * sintheta;

			double y = particles[i].y;
			y += observations[m].x * sintheta;
			y += observations[m].y * costheta;

			t_ob.id = id;
			t_ob.x = x;
			t_ob.y = y;
			t_obs.push_back(t_ob);
		}

		dataAssociation(predictions, t_obs);

		double weight_update, delta_x, delta_y, std_x, std_y;
		std_x = (std_landmark[0] * std_landmark[0]);
		std_y = (std_landmark[1] * std_landmark[1]);
		weight_update = 1.;
		for (int j=0; j<t_obs.size(); j++){
			LandmarkObs observation = t_obs[j];
			LandmarkObs prediction  = predictions[observation.id];
			delta_x = pow(observation.x - prediction.x,2) / std_x;
			delta_y = pow(observation.y - prediction.y,2) / std_y;
			weight_update *= exp(-0.5*(delta_x + delta_y))/(2 * std_landmark[0] * std_landmark[1] * M_PI);
		}
		particles[i].weight = weight_update;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	double index, beta, mw;
	std::vector<Particle> new_particles;

	std::default_random_engine generator;
	std::uniform_int_distribution<int> RandomIndex(0, num_particles-1);
	index = RandomIndex(generator);
	beta = 0.;

	// find max weight
	mw = 0.;
	for (int i; i<num_particles; i++) {
		if (particles[i].weight > mw) {
			mw = particles[i].weight;
		}
	}

	// resampling wheel
	std::normal_distribution<double> RandomBeta(0., 2*mw);
	for (int i=0; i<num_particles; i++) {
		beta += RandomBeta(generator);
		while (beta > particles[index].weight) {
			beta -= particles[index].weight;
			if (index == num_particles-1) { // end of the world
				index = 0;
			}
			else {
				index ++;
			}
		}
		Particle particle;
		particle.id = 1;
		particle.x = particles[index].x;
		particle.y = particles[index].y;
		particle.theta = particles[index].theta;
		particle.weight = 1.;
    new_particles.push_back(particle);
  }
  particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
