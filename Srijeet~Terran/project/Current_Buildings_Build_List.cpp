#include "common.h"


class Current_Buildings_Build_List{

public:

  	struct item {
        Building build;
        int time;
        int finish;

    };



    std::vector<item> presentL;
   const void addItem (const Building & works){



            item * newitem = new item;
          newitem->build=works;
          newitem->time=works.upd->getbuild_time();
          newitem->finish=0;
            presentL.push_back(*newitem);



    }


  const  int get_number_current_building(){

    	int number=0;
    	int i=0;
    	while(i<presentL.size()){

    	    		 if(presentL[i].finish==0){
    	    		number++;

    	    		 }
    	i++;
    	}

    	return number;
    }
  const  std::string update(){

    	//cout<<currentList.size();

    	std::string result="";
            int i=0;
    	 while(i<presentL.size()){

    		 if(presentL[i].finish==0){
    		 presentL[i].time--;

     	           if(presentL[i].time==0){

    	           result+="Finish Build"+presentL[i].build.naam;
presentL[i].finish=1;
    	           }

    	        }
    	        i++;
    	 }
return  result;
    }

    const std::vector<Building> get_finish_building(){


    	std::vector<Building> v=*new std::vector<Building>();
    	int i=0;
    	while(i<presentL.size()){

    		if(presentL[i].finish==1){

        v.push_back(presentL[i].build);
           presentL[i].finish=2;
    		}
    	i++;
    	}

    	return v;
    }


};


