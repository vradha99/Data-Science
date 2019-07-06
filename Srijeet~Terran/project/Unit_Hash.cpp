#include "common.h"



using namespace std;




class Unit_Hash {

protected:
vector<Unit> presentL;

public:


 Unit_Hash(const string race){

 fillHashTable(race);
   }



 const bool search(const string itemnaam){



          int i=0;
        while(i< presentL.size()){

        	   if( presentL[i].naam==itemnaam){

        		   return true;
        	   }
        i++;
        }

        return false;
    }


   const string searchProduceBy(const ProduceBy itemnaam){


        int j=0;
     while(j<itemnaam. presentL.size()){
           for(int i=0; i< presentL.size(); ++i){

           	   if( presentL[i].naam==itemnaam. presentL[j]){

           		   return  presentL[i].naam;
           	   }
           }

    ++j; }

           return "no";
       }

   const Unit getItem(const string itemnaam){

Unit u;

         int i=0;
         while(i!=  presentL.size()){

         	   if( presentL[i].naam==itemnaam){


         		return   presentL[i];
         	   }
        ++i; }

         return u;
     }



  const    void fillHashTable(const string race)
    {

        ifstream file;
        file.open(race);




         string line;


        while (!file.eof())
        {

        	getline(file,line);


                    std::stringstream ss2(line);

                    std::string item1;

                    Info * upd=new Info;

            std::stringstream ss(line);
                  getline(ss, item1, ',');

  if(item1=="")
        	continue;

                  upd->naam = item1;
        getline(ss, item1, ',');


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
d=* new Dependancy();


    	string temps2;


    	  std::stringstream tempstring2(item1);



    	 while(getline(tempstring2,temps2,'/')){

    		d.addItem(temps2);

    	  }



    upd->dependancy= d;

    Unit u=* new Unit();
    u.naam=upd->naam;
    u.upd=upd;
     presentL.push_back(u);



                }


        file.close();


    }


};
