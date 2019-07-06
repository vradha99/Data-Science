

#include "common.h"


using namespace std;
#ifndef Building_Hash_H_
#define Building_Hash_H_

class Building_Hash {

protected:
  struct item {
	        Building  fact;

	    };

	    vector<item> presentL;

public:
 string flnaam;



	    Building_Hash(string race){

	   flnaam=race;
	   readFile(flnaam);
   }



    bool search(string itemname){



      int i=0;
        while(i!= presentL.size())
       {
        	   if(presentL[i].fact.naam==itemname){


        		   return true;
        	   }
        ++i;
        }

        return false;

    }

    Building getItem(const string itemname){

Building build;

         int i=0;
         while(i!= presentL.size())
         {

         	   if(presentL[i].fact.naam==itemname){

         		 return  presentL[i].fact;
         	   }
        ++i;
         }

         return build;

     }



     void readFile(string race)
    {
       flnaam = race;
        ifstream file;
        file.open(flnaam);


        vector<string> vecUnits;

         string line;


        string linetem;
        while (!file.eof())
        {

        	getline(file,line);


                    std::stringstream ss2(line);

                    std::string item1;
                    std::vector<std::string> tokens;
                    int i=0;



            item * newitem = new item;


                    Info * upd=new Info;

            std::stringstream ss(line);
                  getline(ss, item1, ',');

                upd->naam = item1;
        getline(ss, item1, ',');
        if(item1=="")
        	break;

   upd->setminerals(stoi(item1));
   getline(ss, item1, ',');
   upd->setvespene(stoi(item1));
    getline(ss, item1, ',');
    upd->setbuild_time(stoi(item1));
   getline(ss, item1, ',');
   upd->setsupply_cost(stoi(item1));
     getline(ss, item1, ',');
     upd->setsupply_provided(stoi(item1));
    getline(ss, item1, ',');
    upd->setstart_energy(stoi(item1));
     getline(ss, item1, ',');
     upd->setmax_energy(stoi(item1));
     getline(ss, item1, ',');
     upd->race =  item1;
     getline(ss, item1, ',');


     ProduceBy p;
         p= * new ProduceBy();


             	string temps;


             	  std::stringstream tempstring(item1);



             	  while(getline(tempstring,temps,'/')){

             		p.addItem(temps);

             	  }



             upd->produced_by= p;


    getline(ss, item1, ',');
    Dependancy d;
    d= * new Dependancy();


        	string temps2;


        	  std::stringstream tempstring2(item1);



        	  while(getline(tempstring2,temps2,'/')){

        		d.addItem(temps2);

        	  }



        upd->dependancy= d;


    Building fact =* new Building();
   fact.naam=upd->naam;
  fact.upd=upd;


    item it= *new item;
    it.fact=fact;
 presentL.push_back(it);

   }


        file.close();
        int i=0;
        while(i<presentL.size()){


        	Building  factemp=presentL.at(i).fact;

			++i;
			}

    }








};


#endif
