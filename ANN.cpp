#include "ANN.h"

/*
 * Получить результат обученной сети
 */
int KohonenNetwork::Handle(std::vector<int> input){
    for(auto i=0; i<_inputs.size(); i++){
        auto inputNeuron = _inputs[i];
        for(auto outgoingLink :  inputNeuron.outgoingLinks){
            outgoingLink->neuron->power += outgoingLink->weight*input[i];
        }
    }
    
    int maxInput = 0;
    
    
}