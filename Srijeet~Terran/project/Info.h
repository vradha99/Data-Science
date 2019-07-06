#include <string>

#include "Dependancy.cpp"
#include "ProduceBy.cpp"

#ifndef INF_H_
#define INF_H_
class Info {

protected:


	int minerals;
	int vespene;
	int build_time;
	int supply_cost;
	int supply_provided;
	int start_energy;
	int max_energy;

public:

std::string naam = "blank";

	const int getminerals(){return minerals;}
        const void setminerals(int m){minerals=m;}

	const int getvespene(){return vespene;}
        const void setvespene(int v){vespene=v;}

	const int getbuild_time(){return build_time;}
        const void setbuild_time(int b){build_time=b;}

 const int getsupply_cost(){return supply_cost;}
 const void setsupply_cost(int s){supply_cost=s;}

	const int getsupply_provided(){return supply_provided;}
	const void setsupply_provided(int sp){supply_provided=sp;}

       const int getstart_energy(){return start_energy;}
        const void setstart_energy(int se){start_energy=se;}

   const int getmax_energy(){return max_energy;}
        const void setmax_energy(int me){max_energy=me;}

	std::string race = "blank";

	 ProduceBy produced_by;
	 Dependancy dependancy;



};
#endif // INF_H

