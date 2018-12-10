#include <cstdlib>
#include <iomanip>
#include <vector>
#include <unistd.h>
#include "models/MNIST.h"
//#include "ANN.h"
#include "models/KohonenNetwork.h"
#include "models/MLP.h"
#include "models/neuralNet.h"

using namespace std;
//http://www.pvsm.ru/net/7339/print/
//https://neuronus.com/theory/nn/955-nejronnye-seti-kokhonena.html

MLP mlp;
MNIST mnist;

neuralNet *neural_net;

void Kohonen();
void MLP();
void MNIST_TEST();

void neuralNet_TEST();

int main(int argc, char** argv) {

    //Kohonen();

    //MNIST_TEST();
    neuralNet_TEST();

    return 0;
}

void Kohonen() {

    //uczace
    double* input1 = new double [2];
    input1[0] = 0.97;
    input1[1] = 0.2;
    double* input2 = new double [2];
    input2[0] = -0.72;
    input2[1] = 0.7;
    double* input3 = new double [2];
    input3[0] = -0.8;
    input3[1] = 0.6;
    double* input4 = new double [2];
    input4[0] = 0.2;
    input4[1] = -0.97;
    double* input5 = new double [2];
    input5[0] = 1.0;
    input5[1] = 0.0;
    double* input6 = new double [2];
    input6[0] = -0.67;
    input6[1] = 0.74;
    double* input7 = new double [2];
    input7[0] = 0.0;
    input7[1] = -1.0;
    double* input8 = new double [2];
    input8[0] = 0.3;
    input8[1] = -0.95;

    //testujace
    double* test1 = new double [2];
    test1[0] = 0.8;
    test1[1] = 0.1;
    double* test2 = new double [2];
    test2[0] = -0.47;
    test2[1] = 0.8;
    double* test3 = new double [2];
    test3[0] = 0.1;
    test3[1] = -0.65;


    int amount_of_neurons = 3;
    int size_of_input = 2;

    double *outs1 = new double[amount_of_neurons];
    double *outs2 = new double[amount_of_neurons];
    double *outs3 = new double[amount_of_neurons];

    KohonenNetwork * network = new KohonenNetwork(size_of_input, amount_of_neurons);

    outs1 = network->ask_for_outputs(test1);
    outs2 = network->ask_for_outputs(test2);
    outs3 = network->ask_for_outputs(test3);

    cout << "результаты теста перед обучением " << endl;
    for (int i = 0; i < amount_of_neurons; i++) {
        if (outs1[i] == 1)
            cout << "test: 1 \t" << "grupa: " << i + 1 << endl;
        if (outs2[i] == 1)
            cout << "test: 2 \t" << "grupa: " << i + 1 << endl;
        if (outs3[i] == 1)
            cout << "test: 3 \t" << "grupa: " << i + 1 << endl;
    }
    cout << endl;
    // ---------------------------------------------------
    int amount_of_iterations = 1, x = 0;

    for (int nr = 1; nr < 10; nr += 2) {
        x += nr;
        amount_of_iterations = nr;

        for (int i = 0; i < amount_of_iterations; i++) {
            network->learn_WTA(input1);
            network->learn_WTA(input2);
            network->learn_WTA(input3);
            network->learn_WTA(input4);
            network->learn_WTA(input5);
            network->learn_WTA(input6);
            network->learn_WTA(input7);
            network->learn_WTA(input8);
        }
        // ---------------------------------------------------

        outs1 = network->ask_for_outputs(test1);
        outs2 = network->ask_for_outputs(test2);
        outs3 = network->ask_for_outputs(test3);

        cout << endl << "результаты теста после" << x << " итерации обучения" << endl;
        for (int i = 0; i < amount_of_neurons; i++) {
            if (outs1[i] == 1)
                cout << "test: 1 \t" << "grupa: " << i + 1 << endl;
            if (outs2[i] == 1)
                cout << "test: 2 \t" << "grupa: " << i + 1 << endl;
            if (outs3[i] == 1)
                cout << "test: 3 \t" << "grupa: " << i + 1 << endl;
        }
        cout << endl;
    }
}

void MLP() {
    int neuron_count[4] = {2, 5, 4, 1};
    //MLP mlp;

    //create_neural_net(4, neuron_count);

    double inputs[4][2];
    double target[4];
    int data_num = 4;
    int input_size = 2;
    int target_size = 1;

    FILE *file = fopen("input.txt", "r");
    //if (!file) return 1;

    char buff[7];
    for (int i = 0; i < data_num; ++i) {
        fgets(buff, 7, file);
        char *ch = buff;
        for (int j = 0; j < input_size; ++j) {
            while (*ch == ',') ch++;
            inputs[i][j] = *ch - '0';
        }
        for (int j = 0; j < target_size; ++j) {
            while (*ch == ',') ch++;
            target[i] = *ch - '0';
        }
    }

    for (int j = 0; j < 1000000; ++j) {
        for (int i = 0; i < 4; ++i) {
            mlp.run(inputs[i]);
            mlp.backpropagate(&target[i]);
        }
    }

    printf("\nSaidas:\n");
    for (int i = 0; i < 4; ++i) {
        mlp.run(inputs[i]);
        double *out = mlp.get_network_output();
        printf("Entrada: %f %f - Saida: %f\n", inputs[i][0], inputs[i][1], *out);
    }
}

void MNIST_TEST() {
    const int count = 100;
    vector<vector<double>> ar_img;
    vector<vector<double>> ar_label;
    mnist.ReadMNIST("/media/tymbys/archive/progekt/Cpp/ANN/db/t10k-images.idx3-ubyte", count, 784, ar_img, IS_IMG); //28x28
    mnist.ReadMNIST("/media/tymbys/archive/progekt/Cpp/ANN/db/t10k-labels.idx1-ubyte", count, 1, ar_label, IS_LABEL);

    int i = 0;
    for (auto img : ar_img) {
        int w = 28, h = 28;
        int x = 0, y = 0;

        int label = ar_label[i][0];




        if (label == 1) {
            cout << endl << endl << "ind: " << i << "\tImg : " << label << endl;
            for (auto point : img) {
                if (x >= w) {
                    x = 0;
                    y++;
                    cout << endl;
                }
                x++;
                cout << setw(4) << point;
            }

        }

        i++;
    }


    cout << endl;
}

void Kohonen_MNIST_TEST() {
    const int count_data_train = 10; //количество выборок кажной цифры для тренировки

    int w = 28, h = 28;
    const int data_size = 784; //w*h; //784;

    int inc_train[10] = {0};


    const int count = 10;
    vector<vector<double>> ar_img;
    vector<vector<double>> ar_label;
    mnist.ReadMNIST("./db/t10k-images.idx3-ubyte", count, 784, ar_img, IS_IMG); //28x28
    mnist.ReadMNIST("./db/t10k-labels.idx1-ubyte", count, 1, ar_label, IS_LABEL); //1



    KohonenNetwork * network = new KohonenNetwork(data_size, 10);





    int i = 0, i_datda_train = 0;
    for (auto img : ar_img) {
        int x = 0, y = 0;

        int label = ar_label[i][0];


        if (inc_train[label] < count_data_train) {
            inc_train[label]++;

            //data_train.label[i_datda_train] = label;
            //data_train.data[i_datda_train] = img;
            i_datda_train++;
        }


        //проверяем кол-во тренировок
        int sum = 0;
        for (int num = 0; num < 10; num++) {
            sum += inc_train[num];
        }

        if (sum >= count_data_train * 10) break;



        i++;
    }


    cout << endl;

}

void neuralNet_TEST() {

    const int count_data_train = 10; //количество выборок кажной цифры для тренировки
    int inc_train[10] = {0};
    neural_net = new neuralNet(28 * 28, 10, 10);

    //sleep(3);

    const int count = 1000;
    vector<vector<double>> ar_img;
    vector<vector<double>> ar_label;
    mnist.ReadMNIST("./db/t10k-images.idx3-ubyte", count, 784, ar_img, IS_IMG); //28x28
    mnist.ReadMNIST("./db/t10k-labels.idx1-ubyte", count, 1, ar_label, IS_LABEL);


    int i = 0, i_datda_train = 0;

    for (vector<double> img : ar_img) {
        int w = 28, h = 28;
        int x = 0, y = 0;

        int label = ar_label[i][0];

        if (label >= 0 && label < 10  ) { //0..9
            

            if (inc_train[label] < count_data_train) {
                inc_train[label]++;
                
                neural_net->alphabet[label] = img;//.push_back(1);
                neural_net->tests[label].image = img;

                for(int j=0; j < 10; j++){
                    if(j==label)
                       neural_net->tests[label].output[j] = 1;
                    else
                       neural_net->tests[label].output[j] = 0;
                }


                if (inc_train[label] >= count_data_train)
                    cout << "count_data_train: " << inc_train[label] << " , label: " << label << endl;


                //data_train.label[i_datda_train] = label;
                //data_train.data[i_datda_train] = img;
                i_datda_train++;
            }

            cout << "Count image: " << i_datda_train << endl;


            //проверяем кол-во тренировок
            int sum = 0;
            for (int num = 0; num < 10; num++) {
                sum += inc_train[num];
            }

            if (sum >= count_data_train * 10) break;


        }

//        if (label == 1) {
//            cout << endl << endl << "ind: " << i << "\tImg : " << label << endl;
//            for (auto point : img) {
//                if (x >= w) {
//                    x = 0;
//                    y++;
//                    cout << endl;
//                }
//                x++;
//                cout << setw(4) << point;
//            }
//
//        }

            i++;
    }


    cout << endl;

    //exit(0);
    

    int test;
    double mse;

    int count_mse=0;
    //training
    do {



        /* Pick a test at random */
        test = neural_net->rand_test();

        /* Grab input image (with no noise) */
        neural_net->set_network_inputs(test, 0.0);

        /* Feed this data set forward */
        neural_net->feed_forward();

        /* Backpropagate the error */
        neural_net->backpropagate_error(test);

        /* Calculate the current MSE */
        mse = neural_net->calculate_mse(test);
        //     cout<<test<<","<<mse<<endl;

        count_mse++;
    } while (mse > 0.001);

    cout<<"count_mse: "<< count_mse << mse << endl;

    //cross validation 
}
