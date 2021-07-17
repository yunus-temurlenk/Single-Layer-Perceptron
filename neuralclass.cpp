#include "neuralclass.h"

NeuralClass::NeuralClass()
{
    std::cout<<"Neurall class is called..."<<std::endl;
}

double NeuralClass::getY(double x1, double x2)
{
    return w_1 * x1 + w_2 * x2 + bias * w_3;
}

double NeuralClass::sigmoidFunc(double y)
{
    return 1 / (1 + exp(-1 * y));
}

double NeuralClass::thresholdOut(double out)
{
    return out<0.5 ? 0 : 1;
}

double NeuralClass::getError(double target, double result)
{
    return target - result;
}

void NeuralClass::updateWeights(double x1, double x2, double error)
{
    w_1 = w_1 + nL * error * x1;
    w_2 = w_2 + nL * error * x2;
    w_3 = w_3 + nL * error * bias;

}
