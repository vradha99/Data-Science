#include<iostream>
class Resources {


protected :
	double minerals;
	double vespene;
	int supply;
	int energy;
	int used_supply;

public:

       const double getminerals(){ return minerals;}
       const void setminerals(double m){ minerals=m;}

	const double getvespene(){return vespene;}
       const  void setvespene(double v){vespene=v;}

	const int getsupply(){return supply;}
       const void setsupply(int s){supply=s;}

	const int  getenergy(){return energy;}
     const   void setenergy(int e){energy=e;}

	const int getused_supply(){return used_supply;}
      const  void setused_supply(int u){used_supply=u;}


	Resources(int minerals,int vespene,int supply,int energy){

		this->setminerals(50);
		this->setvespene(0);
		this->setsupply(11);
		this->setenergy(0);
        this->setused_supply(0);
	}

};
