#ifndef LEAKYRELU_H
#define LEAKYRELU_H
#include "func.h"
using namespace std;

class leakyRelu: public func
{
public:
    leakyRelu(){};
    double operator () (double value)
    {
        if(value<0)
        {
            return 0.01*value;
        }
        else
        {
            return value;
        }
    }
    double grad (double value)
    {
        if(value<0)
        {
            return 0.01;
        }
        else
        {
            return 1.00;
        }
    }
    ~leakyRelu(){};
};

#endif
