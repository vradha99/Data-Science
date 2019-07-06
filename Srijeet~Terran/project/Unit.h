#include <string>
#include  "Base.h"
#ifndef UNI_H_
#define UNI_H_
class Unit:public Base{
private:

public:
    std::string get_typ(){

    	return "unit";
    }

};
#endif // UNI_H_

