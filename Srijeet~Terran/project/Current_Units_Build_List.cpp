#include "common.h"


class Current_Units_Build_List {
protected:
struct item {
        Unit unit;
        int time;
        int finish;

    };

    std::vector<item> presentL;




public:

  const  void addItem (const Unit & work){



            item * newitem = new item;
          newitem->unit=work;
          newitem->time=work.upd->getbuild_time();
          newitem->finish=0;
            presentL.push_back(*newitem);



    }


  const  int get_number_current_unit(){

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
   const std::string update(){

    	//cout<<currentList.size();

    	std::string result="";
            int i=0;
    	 while(i<presentL.size()){

    		 if(presentL[i].finish==0){
    		 presentL[i].time--;
     	           if(presentL[i].time==0){

    	           result+="Finish Build Unit :"+presentL[i].unit.naam+"\n";
presentL[i].finish=1;
    	           }

    	        }
    	 i++;
    	 }
return  result;
    }

  const  std::vector<Unit> get_finish_unit(){


    	std::vector<Unit> v=*new std::vector<Unit>();
    	int i=0;
    	while(i<presentL.size()){

    		if(presentL[i].finish==1){

          v.push_back(presentL[i].unit);
           presentL[i].finish=2;
    		}
    	i++;
    	}

    	return v;
    }


};
