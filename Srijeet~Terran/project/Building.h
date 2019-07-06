
#include <string>
#include  "Base.h"
#ifndef BUI_H_
#define BUI_H_

class Building:public Base {
public:

 std::string get_typ(){

	return "building";
}
};
#endif // BUI_H_

