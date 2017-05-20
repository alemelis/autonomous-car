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

	num_particles = 1000;

	std::default_random_engine generator;
	std::normal_distribution<double> DistributionX(x, std[0]);
	std::normal_distribution<double> DistributionY(y, std[1]);
	std::normal_distribution<double> DistributionTheta(theta, std[2]);

	particles = std::vector<Particle>(num_particles);
	for (int i=0; i<num_particles; i++) {
		particles[i].id = i;
		particles[i].x = x + DistributionX(generator);
		particles[i].y = y + DistributionY(generator);
		particles[i].theta = theta + DistributionTheta(generator);
		particles[i].weight = 1.;
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
		if (yaw_rate > 1E-5){
			theta_dot_dt = yaw_rate*delta_t;
			particles[i].x += velocity*(sin(theta_0+theta_dot_dt) - sin(theta_0))/yaw_rate;
			particles[i].y += velocity*(-cos(theta_0+theta_dot_dt) + cos(theta_0))/yaw_rate;
			particles[i].theta += theta_dot_dt;
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
	int closest_id;
	for (int i=0; i<observations.size(); i++){
		closest_id = 0;
		shortest_dist = 1E5;
		for (int j=0; j<predicted.size(); j++){
			obs_pred_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (obs_pred_dist < shortest_dist){
				shortest_dist = obs_pred_dist;
				closest_id = predicted[j].id;
			}
		}
		std::cout << closest_id << "\n";
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

	double weight_update;
	std::vector<LandmarkObs> t_obs = std::vector<LandmarkObs>(observations.size());
	for (int i=0; i<num_particles; i++) {
		weight_update = 1.;

		for (int m=0; m<observations.size(); m++) {

			// transform observation to map coordinates
			t_obs[m].x = 0.;
			t_obs[m].x += observations[m].x * cos(particles[i].theta);
			t_obs[m].x -= observations[m].y * sin(particles[i].theta);
			t_obs[m].x += particles[i].x;

			t_obs[m].y = 0.;
			t_obs[m].y += observations[m].x * sin(particles[i].theta);
			t_obs[m].y += observations[m].y * cos(particles[i].theta);
			t_obs[m].y += particles[i].x;

			// find closest landmark
			double dist_from_landm;
			double min_dist = 1.E15;
			int min_dist_id = 0;
			for (int l=0; l<map_landmarks.landmark_list.size(); l++) {
				dist_from_landm = dist(t_obs[m].x, t_obs[m].y, map_landmarks.landmark_list[l].x_f, map_landmarks.landmark_list[l].y_f);

				if (dist_from_landm < min_dist && dist_from_landm <= sensor_range) {
					min_dist = dist_from_landm;
					min_dist_id = map_landmarks.landmark_list[l].id_i;
				}
			}
			weight_update *= 0.5/(M_PI*std_landmark[0]*std_landmark[1])*exp(-0.5*(pow(t_obs[m].x-map_landmarks.landmark_list[min_dist_id].x_f,2)/(std_landmark[0]*std_landmark[0])+
																		 																           pow(t_obs[m].y-map_landmarks.landmark_list[min_dist_id].y_f,2)/(std_landmark[1]*std_landmark[1])));
		}
		particles[i].weight = weight_update;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	double index, beta, mw;

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
	for (int i; i<num_particles; i++) {
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
	}


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
