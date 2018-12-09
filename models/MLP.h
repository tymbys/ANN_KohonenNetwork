#ifndef MLP_H
#define MLP_H

typedef struct neuron{
	double output;
	int input_number;
	double *input_weights;
	double error;
} neuron;



class MLP {
public:
    MLP();
    MLP(int layer_count, int *neuron_count);
    MLP(const MLP& orig);
    virtual ~MLP();
    
    void create_neural_net(int layer_count, int *neuron_count);
    void run(double *input);
    void compute_output(neuron *n, int layer);
    void backpropagate(double *target);
    double* get_network_output();
    double sigmoid(double x);
    double get_random_weight();
    
private:
    int layer_num;
    int *neuron_num;
    neuron **neural_net;
};

#endif /* MLP_H */

