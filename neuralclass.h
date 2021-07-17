#ifndef NEURALCLASS_H
#define NEURALCLASS_H

#include <vector>
#include <math.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

class NeuralClass
{
public:
    NeuralClass();

    //initialize the weights
    double w_1 = 0.0;
    double w_2 = 0.0;
    double w_3 = 0.0;

    //learning rate
    double nL = 0.2;

    //bias
    double bias = 1;

    //inputs
    std::vector<std::vector<int>> inputs {

        {100,200},
        {200,-100},
        {50,50},
        {50,10},
        {-50,-20},
        {-70,-50},
        {200,200},
        {200,-50},
        {150,-50},
        {180,220},
        {0,0},
        {-100,-220}


    };

    //targets
    std::vector<int> targets{

        1,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1
    };


    // forward pass
    double getY(double x1, double x2);
    double sigmoidFunc(double y);
    double thresholdOut(double out);




    // backward pass
    double getError(double target, double result);
    void updateWeights(double x1, double x2, double error);





};

#endif // NEURALCLASS_H
