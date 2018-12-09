#ifndef NEURALNET_H
#define NEURALNET_H

#include <iostream>
#include <fstream>
#include <cstring>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include "math.h"

//https://github.com/sphilip/CSCI598D-Machine_Learning
//https://github.com/sphilip/CSCI598D-Machine_Learning/blob/2ecb5f37ece1f68f9fc25f4c2bb823740f205446/pattern_recognition/good_code/neuralNet.cpp

//#define RHO (double)0.1

typedef struct test_images_s {
    int *image;//[IMAGE_SIZE];
    int *output;//[OUTPUT_NEURONS];

} test_image_t;

class neuralNet {
public:

    neuralNet(const int inputs_neurons, const int hidden_neurons, const int outputs_neurons);
    virtual ~neuralNet();

    void setTestImage(test_image_t *test_image);

    int rand_test();
    void feed_forward(void);
    void set_network_inputs(int test, double noise_prob);
    
    void backpropagate_error(int test);
    double calculate_mse(int test);
    
private:

    int INPUT_NEURONS; // #define INPUT_NEURONS	322
    int HIDDEN_NEURONS; //#define HIDDEN_NEURONS	26
    int OUTPUT_NEURONS; //#define OUTPUT_NEURONS	26
    
    int MAX_TESTS;

    double *inputs; //[INPUT_NEURONS + 1];
    double *hidden; //[HIDDEN_NEURONS + 1];
    double *outputs; //[OUTPUT_NEURONS];

    
    double *w_h_i;
    double *w_o_h;
    //double *w_h_i; //[HIDDEN_NEURONS][INPUT_NEURONS + 1];
    //double *w_o_h; //[OUTPUT_NEURONS][HIDDEN_NEURONS + 1];

    int **alphabet;
    int image_size;
    int alphabet_count;
    
    double RHO;

    test_image_t *tests;

    double RAND_WEIGHT();
    double sigmoid(double x);
    double sigmoid_d(double x);
    
    int find_ans(int index);

    void init_network(void);
    
    
    
    int classifier(void);

};

#endif /* NEURALNET_H */

