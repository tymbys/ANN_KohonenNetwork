#include "KohonenNeuron.h"

using namespace std;

KohonenNeuron::KohonenNeuron(){ }

void KohonenNeuron::init(int size)
{
    learn_number = 0.1;
    this->size = size;
    weights = new double[size];
    initialize_weight();
}

KohonenNeuron::~KohonenNeuron()
{
    if (weights)
            delete[] weights;
}

void KohonenNeuron::initialize_weight()
{
    for (int i = 0; i < size; i++){
            weights[i] = (double)rand() / (double)RAND_MAX;
    }
    cout << endl;
}


double KohonenNeuron::get_euclidean_length(double *input)
{
    double tmp = 0.0, length = 0.0;
    for(int i=0; i< size; i++)
            tmp += (input[i] - weights[i]) * (input[i] - weights[i]);

    length = sqrt(tmp);
    return length;
}

double KohonenNeuron::ask(double* input)
{
        return get_euclidean_length(input);
}

void KohonenNeuron::learn_WTA(double * input)
{
    double dif;
    for(int i =0; i< size; i++)
    {
            dif = input[i] - weights[i];
            weights[i] += learn_number * dif;

    }
}