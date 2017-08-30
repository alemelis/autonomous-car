int main() {
    
    vector< vector<double> > X_train = Load_State("./train_states.txt");
    vector< vector<double> > X_test  = Load_State("./test_states.txt");
    vector< string > Y_train  = Load_Label("./train_labels.txt");
    vector< string > Y_test   = Load_Label("./test_labels.txt");
    
	cout << "X_train number of elements " << X_train.size() << endl;
	cout << "X_train element size " << X_train[0].size() << endl;
	cout << "Y_train number of elements " << Y_train.size() << endl;

	GNB gnb = GNB();
	
	vector<vector<double>> mus_c(3, vector<double>(X_train[0].size()));
	vector<vector<double>> sigmas_c(3, vector<double>(X_train[0].size()));
	gnb.train(X_train, Y_train, mus_c, sigmas_c);

	cout << "X_test number of elements " << X_test.size() << endl;
	cout << "X_test element size " << X_test[0].size() << endl;
	cout << "Y_test number of elements " << Y_test.size() << endl;
	
	int score = 0;
	for(int i = 0; i < X_test.size(); i++)
	{
		vector<double> coords = X_test[i];
		string predicted = gnb.predict(coords, mus_c, sigmas_c);
		if(predicted.compare(Y_test[i]) == 0)
		{
			score += 1;
		}
	}

	float fraction_correct = float(score) / Y_test.size();
	cout << "You got " << (100*fraction_correct) << " correct" << endl;

	return 0;
}
