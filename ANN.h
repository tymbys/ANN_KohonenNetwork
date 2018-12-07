#ifndef ANN_H
#define ANN_H

//#include "Neuron.h"
//#include "Link.h"
//#include "Input.h"

#include <vector>

class Neuron;
class Input;
class Link;


class Neuron {
public:
    std::vector<Link*> incomingLinks;
    double power;
};

// Вход 
class Input {
public:
    // Связи с нейронами
    std::vector<Link*> outgoingLinks;
};

class Link {
public:
    Neuron * neuron;
    double weight;
};


class KohonenNetwork{
public:
    std::vector<Input> _inputs;
    std::vector<Neuron> _neuron;
    
    int Handle(std::vector<int> input);
    void Stady(std::vector<int> input, int correctAnswer);

};




#endif /* ANN_H */

