

#include "common.h"

using namespace std;
class Current_Units{

protected:

 struct item {
	        Unit unit;
	        int quanta;
	        int labour;
	        int gas_collector;
	        int mineral_collector;
	    };

	    vector<item> presentL;



public:

   const void useworker(const string workernaam){

            int i=0;
    	  while(i!= presentL.size()){

    	        	   if(presentL[i].labour==1){

    	        		 presentL[i].quanta--;
    	        		 presentL[i].mineral_collector--;
    	        	   }
    	        ++i;
    	        }


    }

   const void freewoker(const string workernaam){

           int i=0;
    	  while(i!= presentL.size()){

    	        	   if(presentL[i].unit.naam==workernaam){

    	        		 presentL[i].quanta++;

    	        		 if(presentL[i].labour==1){

    	        			 presentL[i].mineral_collector++;

    	        		 }

    	        	   }
    	        ++i;
    	        }

    }

 const void addGasCollector(const int quanta){

         int i=0;
     	while(i!= presentL.size()){

    	    	        	   if(presentL[i].labour==1){

    	    	        		 presentL[i].gas_collector+=quanta;
    	    	        		 presentL[i].mineral_collector-=quanta;
    	    	        	   }
    	    	        ++i;
    	    	        }


    }

  const  void removeGasCollector(const int quanta){

             int i=0;
        	while(i!=  presentL.size()){

        	    	        	   if( presentL[i].labour==1){

        	    	        		  presentL[i].gas_collector-=quanta;
        	    	        		  presentL[i].mineral_collector+=quanta;
        	    	        	   }
        	    	        ++i;
        	    	        }


        }

   const int getworkernumber(){




                 int i=0;
        	  while(i!= presentL.size()){

        	        	   if( presentL[i].labour==1){

        	        		 return  presentL[i].quanta;
        	        	   }

      ++i;
        }
        return 0;

    }




 const  bool search(const string itemnaam){



          int i=0;
        while(i!=  presentL.size()){

        	   if( presentL[i].unit.naam==itemnaam){

        		   return true;
        	   }
        ++i;
        }

        return false;
    }

  const  int numUnuits(const string itemnaam){


 int num=0;
           int i=0;
           while(i!=  presentL.size()){

           	   if( presentL[i].unit.naam==itemnaam){

           		   num++;
           	   }
          ++i;
           }

           return num;
       }
   const void addItem (const Unit  unit,const int lab){

        bool fonund=false;
         int i=0;
        while(i!=  presentL.size()){
            if( presentL[i].unit.naam==unit.naam){
                 presentL[i].quanta++;
                if(lab){
                 presentL[i].mineral_collector++;
                }
                fonund=true;
                break;
            }
        ++i;
        }

        if(!fonund){

            item * newitem = new item;

            newitem->unit = unit;

            newitem->quanta = 1;
            newitem->labour=lab;
            newitem->gas_collector=0;
            if(lab){
            newitem->mineral_collector=1;
            }else{
            	newitem->mineral_collector=0;

            }


             presentL.push_back(*newitem);

        }
    }

 const int getMineralCollectorNumber(){

    	 int i=0;
    	  while(i!=  presentL.size()){

    	        	        	   if( presentL[i].labour==1){

    	        	        		 return  presentL[i].mineral_collector;
    	        	        	   }


    	      ++i;  }

    }


  const  int getGasCollectorNumber(){

          int i=0;
    	  while (i!= presentL.size()){

    	        	        	   if(presentL[i].labour==1){

    	        	        		 return presentL[i].gas_collector;
    	        	        	   }


    	     ++i;
    	        }

    }



};
