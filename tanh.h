#ifndef TANH_H
#define TANH_H
#include "func.h"
using namespace std;

class Tanh: public func
{
public:
    Tanh(){};
    double operator () (double value)
    {
        return (2.f/(1.f + exp(-2*value))-1.f);
    }
    double grad (double value)
    {
        return (1.f-value*value);
    }
    ~Tanh(){};
};

#endif
