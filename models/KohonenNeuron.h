#ifndef KOHONENNEURON_H
#define KOHONENNEURON_H

#include<iostream>
#include<math.h>
#include<stdlib.h>

class KohonenNeuron
{
public:
	double * weights;
	double learn_number, output;
	int size;

	KohonenNeuron();
        ~KohonenNeuron();
	void init(int size);
	void initialize_weight();
	double get_euclidean_length(double *input);
	double ask(double* input);
	void learn_WTA(double * input);

};

#endif /* KOHONENNEURON_H */

