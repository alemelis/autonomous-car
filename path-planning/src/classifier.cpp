#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "classifier.h"

/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels, vector<vector<double>>& mus_c, vector<vector<double>>& sigmas_c)
{

	/*
		Trains the classifier with N data points and labels.

		INPUTS
		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d, 
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
	*/
	
	string current_label;
	double mu_c, sigma_c;
	int counter;
	
	// for each quantity
	for (int j=0; j<data[1].size(); j++){
	    
	    // for each label
	    for (int l=0; l<possible_labels.size(); l++){
	        current_label = possible_labels[l];
	        
	        vector<double> values_c;
	        counter = 0;
	        mu_c = 0;
	        sigma_c = 0;
	        
	        // check each training entry and compute mean
	        for (int i=0; i<data.size(); i++){
	            if (current_label == labels[i]){
	                values_c.push_back(data[i][j]);
	                mu_c += data[i][j];
	                counter += 1;
	            }
	        }
	        mu_c /= counter;
	        mus_c[l][j] = mu_c;
	        
	        // compute std
	        for (int k=0; k<values_c.size(); k++){
	            sigma_c += pow(values_c[k]-mu_c, 2);
	        }
	        sigma_c /= counter;
	        sigma_c = sqrt(sigma_c);
	        
	        sigmas_c[l][j] = sigma_c;
	    }
	}
}

string GNB::predict(vector<double> sample, vector<vector<double>> mus_c, vector<vector<double>> sigmas_c)
{
	/*
		Once trained, this method is called and expected to return 
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
		# TODO - complete this
	*/
	
	double mu_c, sigma_c;
	double p;
	double max_p=0;
	int idx = 0;
	
    // check each class
    for (int i=0; i<possible_labels.size(); i++){
        p = 1;
        
        // compute probability for each quantity
        for (int j=0; j<sample.size(); j++){
            mu_c = mus_c[i][j];
            sigma_c = sigmas_c[i][j];
            double sigma_c2 = pow(sigma_c, 2);
            
            p *= 1/sqrt(2*M_PI*sigma_c2)*exp(-0.5*pow(sample[j]-mu_c, 2)/sigma_c2);
        }
        
        if (p > max_p){
            max_p = p;
            idx = i;
        }
    }    

	return this->possible_labels[idx];

}
