#ifndef SIGMOID_H
#define SIGMOID_H
#include "func.h"
using namespace std;

class Sigmoid: public func
{
public:
    Sigmoid(){};
    double operator () (double value)
    {
        return (1.f/(1.f + exp(-value)));
    }
    double grad (double value)
    {
        return ((1.f-value)*value);
    }
    ~Sigmoid(){};
};

#endif
