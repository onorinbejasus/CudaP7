#ifndef _AED_MATH_MOUSE_HPP_
#define _AED_MATH_MOUSE_HPP_

namespace aed {

struct MouseData {
    double srcx;
    double srcy;
    double relx;
    double rely;

    // XXX Would prefer these to be bools but that screws up alignment
    int buttonLeftDown;
    int buttonRightDown;
    int buttonMiddleDown;
};

}

#endif
