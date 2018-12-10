#include "neuralNet.h"

neuralNet::neuralNet(const int inputs_neurons, const int hidden_neurons, const int outputs_neurons) : RHO(0.1) {

    srand(time(NULL));

    MAX_TESTS = 10;
    //MAX_TESTS = max_tests;

    INPUT_NEURONS = inputs_neurons;
    HIDDEN_NEURONS = hidden_neurons;
    OUTPUT_NEURONS = outputs_neurons;

    //inputs = new double[INPUT_NEURONS + 1];
    //hidden = new double[HIDDEN_NEURONS + 1];
    //outputs = new double[OUTPUT_NEURONS];
    inputs.resize(INPUT_NEURONS + 1);
    hidden.resize(HIDDEN_NEURONS + 1);
    outputs.resize(OUTPUT_NEURONS);

    //w_h_i = new double[HIDDEN_NEURONS][INPUT_NEURONS + 1];
    //w_o_h = new double[OUTPUT_NEURONS][HIDDEN_NEURONS + 1];

    //w_h_i = new double[HIDDEN_NEURONS*INPUT_NEURONS + 1];
    //w_o_h = new double[OUTPUT_NEURONS*HIDDEN_NEURONS + 1];

    w_h_i.resize(HIDDEN_NEURONS, vector<double>(INPUT_NEURONS + 1));
    w_o_h.resize(OUTPUT_NEURONS, vector<double>(HIDDEN_NEURONS + 1));

    //tests = new test_image_t[MAX_TESTS];


    //tests = new test_images_s(INPUT_NEURONS, OUTPUT_NEURONS);
    tests.resize(MAX_TESTS);

    for (int test_im = 0; test_im < tests.size(); test_im++) {

        tests[test_im].image.resize(INPUT_NEURONS);
        tests[test_im].output.resize(OUTPUT_NEURONS);
    }

    init_network();

    //alphabet[index] = new int[width*height];
    alphabet.resize(outputs_neurons, vector<double>(inputs_neurons));
}

neuralNet::~neuralNet() {
}

void neuralNet::setTestImage(vector<test_images_t> test_image, const int max_tests) {
    tests = test_image;
    MAX_TESTS = max_tests;
}

double neuralNet::sigmoid(double x) {
    return 1.0 / (1 + exp(-x));
}

double neuralNet::sigmoid_d(double x) {
    double g = sigmoid(x);
    return g * (1 - g);
}

double neuralNet::RAND_WEIGHT() {
    return (((double) rand() / (double) RAND_MAX) - 0.5);
}

void neuralNet::init_network(void) {
    int i, j;

    /* Set the input bias */
    inputs[INPUT_NEURONS] = 1.0;

    /* Set the hidden bias */
    hidden[HIDDEN_NEURONS] = 1.0;

    /* Initialize the input->hidden weights */
    for (j = 0; j < HIDDEN_NEURONS; j++) {
        for (i = 0; i < INPUT_NEURONS + 1; i++) {
            w_h_i[j][i] = RAND_WEIGHT();
        }
    }

    for (j = 0; j < OUTPUT_NEURONS; j++) {
        for (i = 0; i < HIDDEN_NEURONS + 1; i++) {
            w_o_h[j][i] = RAND_WEIGHT();
        }
    }

    return;
}

void neuralNet::feed_forward(void) {
    int i, j;

    /* Calculate outputs of the hidden layer */
    for (i = 0; i < HIDDEN_NEURONS; i++) {

        hidden[i] = 0.0;

        for (j = 0; j < INPUT_NEURONS + 1; j++) {
            hidden[i] += (w_h_i[i][j] * inputs[j]);
            //hidden[i] += (w_h_i[i*INPUT_NEURONS + j] * inputs[j]);
        }

        hidden[i] = sigmoid(hidden[i]);

    }

    /* Calculate outputs for the output layer */
    for (i = 0; i < OUTPUT_NEURONS; i++) {

        outputs[i] = 0.0;

        for (j = 0; j < HIDDEN_NEURONS + 1; j++) {
            outputs[i] += (w_o_h[i][j] * hidden[j]);
            //outputs[i] += (w_o_h[i*HIDDEN_NEURONS + j] * hidden[j] );
        }

        outputs[i] = sigmoid(outputs[i]);

    }

}

void neuralNet::backpropagate_error(int test) {
    int out, hid, inp;
    double err_out[OUTPUT_NEURONS];
    double err_hid[HIDDEN_NEURONS];

    /* Compute the error for the output nodes */
    for (out = 0; out < OUTPUT_NEURONS; out++) {

        err_out[out] = ((double) tests[test].output[out] - outputs[out]) * sigmoid_d(outputs[out]);

    }

    /* Compute the error for the hidden nodes */
    for (hid = 0; hid < HIDDEN_NEURONS; hid++) {

        err_hid[hid] = 0.0;

        /* Include error contribution for all output nodes */
        for (out = 0; out < OUTPUT_NEURONS; out++) {
            err_hid[hid] += err_out[out] * w_o_h[out][hid];
            //err_hid[hid] += err_out[out] * w_o_h[out*HIDDEN_NEURONS + hid];
        }

        err_hid[hid] *= sigmoid_d(hidden[hid]);

    }

    /* Adjust the weights from the hidden to output layer  */
    for (out = 0; out < OUTPUT_NEURONS; out++) {

        for (hid = 0; hid < HIDDEN_NEURONS; hid++) {
            w_o_h[out][hid] += RHO * err_out[out] * hidden[hid];
            //w_o_h[out*HIDDEN_NEURONS + hid] += RHO * err_out[out] * hidden[hid];
        }

    }

    /* Adjust the weights from the input to hidden layer  */
    for (hid = 0; hid < HIDDEN_NEURONS; hid++) {

        for (inp = 0; inp < INPUT_NEURONS + 1; inp++) {
            w_h_i[hid][inp] += RHO * err_hid[hid] * inputs[inp];
            //w_h_i[hid*(INPUT_NEURONS+1) + inp] += RHO * err_hid[hid] * inputs[inp];
        }

    }

    return;
}

double neuralNet::calculate_mse(int test) {
    double mse = 0.0;
    int i;

    for (i = 0; i < OUTPUT_NEURONS; i++) {
        double t = tests[test].output[i] - outputs[i];
        mse += t*t;
    }

    return ( mse / (double) i);
}

void neuralNet::set_network_inputs(int test, double noise_prob) {
    int i;

    /* Fill the network inputs vector from the test */
    for (i = 0; i < INPUT_NEURONS; i++) {

        inputs[i] = tests[test].image[i];

        /* In the given noise probability, negate the cell */
        if (rand() < noise_prob * RAND_MAX) {
            inputs[i] = (inputs[i]) ? 0 : 1;
        }

    }

    return;
}

int neuralNet::classifier(void) {
    int i, best;
    double max;

    best = 0;
    max = outputs[0];


    for (i = 1; i < OUTPUT_NEURONS; i++) {

        if (outputs[i] > max) {
            max = outputs[i];
            best = i;
        }

    }

    return best;
}

int neuralNet::rand_test() {
    int r = rand();
    int RR = RAND_MAX / MAX_TESTS;
    int rr = r / (RR);
    return rr;
}

int neuralNet::find_ans(int index) {
    int i = 0;
    for (i = 0; i < MAX_TESTS; i++)
        if (tests[index].output[i] == 1) break;

    return i;
}

bool neuralNet::saveNetAsInclude(string name) {
    //ifstream infile(file, ifstream::in);
    ofstream outfile(name, ifstream::out);
    if (!outfile) {
        cout << "Can't read from " << name << endl;
        return false;
    }

    //w_h_i

    //    vector<vector<double>> w = {
    //        {1,1,1,1},
    //        {2,2,2,2}
    //    };

    bool isFirst_x = true;
    bool isFirst_y = true;

    //save w_h_i
    outfile << "const double w_h_i["<< HIDDEN_NEURONS <<"][" <<INPUT_NEURONS + 1<<"] = {";

    for (auto y : w_h_i) {
        if (isFirst_y) {
            outfile << "\n{";
            isFirst_y = false;
        } else
            outfile << ",\n{";

        isFirst_x = true;
        for (double x : y) {
            if (isFirst_x) {
                outfile << x;
                isFirst_x = false;
            } else
                outfile << ",\t" << x;
        }
        outfile << "}";

    }
    outfile << "\n};\n\n";



    //save w_o_h
    outfile << "const double w_o_h["<<OUTPUT_NEURONS <<"]["<< HIDDEN_NEURONS + 1 <<"] = {";

    isFirst_x = true;
    isFirst_y = true;
    for (auto y : w_o_h) {
        if (isFirst_y) {
            outfile << "\n{";
            isFirst_y = false;
        } else
            outfile << ",\n{";

        isFirst_x = true;
        for (double x : y) {
            if (isFirst_x) {
                outfile << x;
                isFirst_x = false;
            } else
                outfile << ",\t" << x;
        }
        outfile << "}";

    }
    outfile << "\n};\n\n";

    return true;
}