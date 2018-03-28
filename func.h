#ifndef FUNC_H
#define FUNC_H
using namespace std;

class func
{
public:
    func(){};
    virtual double operator () (double value) = 0;
    virtual double grad (double value) = 0;
    virtual ~ func(){};
};



#endif
