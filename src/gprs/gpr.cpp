#include "gprs/gpr.h"
#include <Eigen/Dense>
#include <iostream>
#include <chrono>

namespace GPRcpp
{

gpr::gpr()
{

}

gpr::gpr(std::shared_ptr<kernel_base> kernel)
    : kernel_(kernel)
{
    
};

gpr::gpr(std::shared_ptr<kernel_base> kernel, bool normalize_y)
    : kernel_(kernel), normalize_y_(normalize_y)
{
    
};

gpr::~gpr()
{
    // delete kernel_;
}


}