#include "common.h"

class Current_Buildings{

protected:
    struct item {
        Building build;

       int work;
    };

    std::vector<item> presentL;

public:
    const std::string get_typ(){

    	return "building";
    }




    const bool availableBuilding(const std::string buildnaam){

    	int r=false;

         int i=0;

    	 while(i<presentL.size()){




    	   if(presentL[i].build.naam==buildnaam and presentL[i].work==0){


    	    	        		 return true;
    	    	        	   }
    	    	       ++i;
    	    	        }
    	   return false;
    }


    const std::string aviableBuildingProduceBy(const ProduceBy fact){


int i=0;
while(i<fact.presentL.size()){


	  for(int j=0; j<presentL.size(); ++j){




	     	   if(presentL[j].build.naam==fact.presentL[i] and presentL[j].work==0){


	     	    	        		 return presentL[j].build.naam;
	     	    	        	   }

	     	    	        }

	  return "no";
	  i++;
  }

}
    const bool useBuilding(const std::string buildnaam){

           int i=0;
    	  while(i!= presentL.size()){

    	        	   if(presentL[i].build.naam==buildnaam and presentL[i].work==0){

    	        		presentL[i].work=1;

    	        		 return true;
    	        	   }
    	       ++i;
    	        }
   return false;

    }

    const void freeBuilding(const std::string buildnaam){

            int i=0;
    	  while (i!= presentL.size()){

    	        	   if(presentL[i].build.naam==buildnaam and presentL[i].work==1){

    	        		   presentL[i].work=0;

    	        }
                ++i;

                }

    }


const bool searchDependancy(const Dependancy dependany){






	bool result=false;

     int j=0;
	while(j<dependany.presentL.size()){

		for(int i=0;i<presentL.size();i++){

		if(dependany.presentL[j]==presentL[i].build.naam){


			return true;
		}

	}
	++j;

	}

	return result;


}

    const bool search(const std::string itemnaam){



         int i=0;
        while(i!= presentL.size()){

        	   if(presentL[i].build.naam==itemnaam){

        		   return true;
        	   }
       ++i;
        }

        return false;
    }



    const void delet(const std::string itemnaam){



               int i=0;
           while(i!= presentL.size()){

           	   if(presentL[i].build.naam==itemnaam){

           		   presentL.erase(presentL.begin()+i);
 return;
           	   }
          ++i;
           }


       }
   const void addItem (const Building itemnaam){



            item * newitem = new item;

            newitem->build = itemnaam;


            newitem->work=0;
            presentL.push_back(*newitem);


    }

    };


