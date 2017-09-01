#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

using namespace std;

class GNB {
public:

	vector<string> possible_labels = {"left","keep","right"};


	/**
  	* Constructor
  	*/
 	GNB();

	/**
 	* Destructor
 	*/
 	virtual ~GNB();

 	void train(vector<vector<double> > data, vector<string> labels, vector<vector<double> >& mus_c, vector<vector<double> >& sigmas_c);

  string predict(vector<double>, vector<vector<double> > mus_c, vector<vector<double> > sigmas_c);

};

#endif
