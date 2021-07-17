#include <neuralclass.h>


int main()
{

    NeuralClass NN;
    int epochNum = 1;

    int inputSize = NN.inputs.size();
    int targetSize = NN.targets.size();

    if(inputSize != targetSize)
    {
        std::cout<<"Ooops!!! Input size does not match with the target size..."<<std::endl;
        return -1;
    }


    while (1) {

        int goodEstimationNumber = 0;

        for(int i=0; i<inputSize; i++)
        {
            double y = NN.getY(NN.inputs[i][0],NN.inputs[i][1]);

            double error = NN.getError(NN.targets[i],NN.thresholdOut(NN.sigmoidFunc(y)));

            if(error == 0)
                goodEstimationNumber++;

            NN.updateWeights(NN.inputs[i][0], NN.inputs[i][1],error);
        }

        if(goodEstimationNumber == inputSize)
        {
            std::cout<<"Training ended!!! All estimations are correct" <<std::endl;

            std::cout<<"Epoch Number is: "<< epochNum <<std::endl;

            std::cout<<"w1:  "<< NN.w_1 <<" w2: "<<NN.w_2<<" w3:  "<<NN.w_3<<std::endl;

            break;
        }

        if(epochNum>100000)
        {
            std::cout<<"Looks like not linearly seperable..."<<std::endl;
            return -1;
        }


        epochNum++;
    }

    cv::Mat graph = cv::Mat::zeros(cv::Size(500,500), CV_8UC3);
    cv::line(graph,cv::Point(250,0), cv::Point(250,500),cv::Scalar(255,255,255),5);
    cv::line(graph,cv::Point(0,250), cv::Point(500,250),cv::Scalar(255,255,255),5);
    cv::namedWindow("Plot",0);

    for(int i=0; i<targetSize; i++)
    {
        if(NN.targets[i] == 1)
        {
            cv::circle(graph,cv::Point(NN.inputs[i][0] + 250,250 - NN.inputs[i][1]),5, cv::Scalar(0,255,255), 5);
        }
        else {
            cv::circle(graph,cv::Point(NN.inputs[i][0] + 250,250 - NN.inputs[i][1]),5, cv::Scalar(0,0,255), 5);
        }


    }



    double yAxis250 = (NN.w_1 * 250  + NN.w_3) / (-1 * NN.w_2);
    double yAxisValue250 = 0.0;
    if(yAxis250<250 && yAxis250>-250)
    {
        yAxisValue250 = yAxis250;
    }
    else if (yAxis250>=250) {
        yAxisValue250 = 250;
    }
    else {
        yAxisValue250 = -250;
    }

    double yAxis_250 = (NN.w_1 * -250  + NN.w_3) / (-1 * NN.w_2);
    double yAxisValue_250 = 0.0;
    if(yAxis_250<250 && yAxis_250>-250)
    {
        yAxisValue_250 = yAxis_250;
    }
    else if (yAxis_250>=250) {
        yAxisValue_250 = 250;
    }
    else {
        yAxisValue_250 = -250;
    }

    if(NN.w_1 != 0)
    {
        double xAxis250 = (NN.w_2 * yAxisValue250  + NN.w_3) / (-1 * NN.w_1);
        double xAxis_250 = (NN.w_2 * yAxisValue_250  + NN.w_3) / (-1 * NN.w_1);
        cv::line(graph, cv::Point(xAxis250 + 250, 250 - yAxisValue250), cv::Point(xAxis_250 + 250, 250 - yAxisValue_250),cv::Scalar(255,0,0),3,cv::LINE_AA);
    }
    else {
        cv::line(graph, cv::Point(0, 250 - yAxisValue250), cv::Point(500, 250 - yAxisValue_250),cv::Scalar(255,0,0),3,cv::LINE_AA);

    }

    cv::imshow("Plot",graph);
    cv::waitKey(0);

    return 0;
}
