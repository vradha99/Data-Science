#include <iostream>


#include <string>
#include <vector>

using namespace std;
#ifndef ARB_H_
#define ARB_H_
class Arbitary{
public:
vector<string> presentL;
bool search(const std::string itemnaam){
int j=0;
while(j!= presentL.size()){
if(presentL[j]==itemnaam){
return true;
	        	   }
	       ++j;
	        }
return false;
	    }
void addItem (const string itemnaam){
presentL.push_back(itemnaam);


}

};
#endif
