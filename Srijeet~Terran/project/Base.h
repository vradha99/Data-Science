

#include<string>
#include  "Info.h"
using namespace std;
#ifndef BAS_H_
#define BAS_H_
class Base{

public:

         string naam;
	 string madeby;


	virtual string get_typ()=0;


        Info*   upd;
};
#endif

