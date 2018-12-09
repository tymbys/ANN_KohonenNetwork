#ifndef KOHONENNETWORK_H
#define KOHONENNETWORK_H

#include "KohonenNeuron.h"
#include<math.h>

class KohonenNetwork
{
public:

	KohonenNeuron * neurons;
	int size_of_input, amount_of_neurons;

	KohonenNetwork(int size_of_input, int amount_of_neurons);	

	double * ask_for_outputs(double * input);
	int get_winner(double * input);
	void normalize_vector(double * vector);
	// ----------------- WTA - Winner Take All -----------------
	void learn_WTA(double * input);

	// ----------------- WTM - Winner Take Most -----------------
	void learn_WTM(double *input, int promien_sasiedztwa);

};

#endif /* KOHONENNETWORK_H */

