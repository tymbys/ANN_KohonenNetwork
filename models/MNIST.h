#ifndef MNIST_H
#define MNIST_H

#include <iostream>
#include <vector>
#include <fstream>
 
using namespace std;

enum DATA_TYPE {IS_LABEL=2, IS_IMG=4 };

class MNIST {
public:
    MNIST();
    virtual ~MNIST();
    
    int ReverseInt (int i);
    void ReadMNIST(string db, int NumberOfImages, int DataOfAnImage,vector<vector<double>> &arr, int DATA_TYPE);
    
private:

};

#endif /* MNIST_H */

