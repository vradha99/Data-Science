#include"common.h"

class Wait_List{

protected:
    struct item {
        std::string naam;

        std::string type;
      int start_working;
    };



public:
std::vector<item> presentL;

   const void addItem (const std::string itemnaam,const std::string type){


            item * newitem = new item;
            newitem->naam = itemnaam;
            newitem->type=type;

            newitem->start_working = 0;
         presentL.push_back(*newitem);



    }



   const bool searchDependancy(const Dependancy dependany){


    	bool result=false;

          int j=0;
    	while(j<dependany.presentL.size()){

    		for(int i=0;i<presentL.size();i++){

    		if(dependany.presentL[j]==presentL[i].naam){


    			return true;
    		}

    	}

    	j++;
    	}

    	return result;

    }


  const  bool searc(const std::string key){


        	bool result=false;



                int i=0;
        		while(i<presentL.size()){

        		if(key==presentL[i].naam){


        			return true;
        		}

        	i++;
        	}



        	return result;

        }


  const  int get_number_wait(){
int number=0;
int i=0;
            while(i<presentL.size()){

     	    	          if(presentL[i].start_working==0){

     	    	        number++;

     	    	          }

           ++i;
     	    	  }

     	 return number;
     }

 const   std::string get_first_wait()
    {
                int i=0;
    	  while(i<presentL.size()){

    	          if(presentL[i].start_working==0){


    	        	  return presentL[i].naam;

    	          }

             i++;
    	  }




    }

  const  std::string get_first_wait_list_type(){
          int i=0;
    	while(i<presentL.size()){

    	    	          if(presentL[i].start_working==0){


    	    	        	  return presentL[i].type;

    	    	          }


    	    	i++;
    	    	  }


    }

    const void remove_first_wait(){
           int i=0;
    	 while(i<presentL.size()){

    	    	          if(presentL[i].start_working==0){

    	    	        	presentL[i].start_working=1;

    	    	        	return;
    	    	          }


    	    	i++;
    	    	  }


    }


  const  void start_work(const std::string itemnaam){

           int i=0;
        while(i<presentL.size()){

if(presentL[i].start_working==1)
{

	if(presentL[i].naam==itemnaam){


		presentL[i].start_working=2;
	}

}
i++;
}

    }



};


