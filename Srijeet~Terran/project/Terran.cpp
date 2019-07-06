#include"common.h"
#include "Unit_Hash.cpp"
#include "Resources.h"
#include "Building.h"
#include "Building_Hash.cpp"
#include "Wait_List.cpp"
#include "Current_Buildings_Build_List.cpp"
#include "Current_Units_Build_List.cpp"
#include "Current_Buildings.cpp"
#include "Current_Units.cpp"
using namespace std;

int main(const int argc, const char *argv[]) {

 	const string listfile=argv[2];
 	const string race=argv[1];

////////////////////////////////////////////////////////////////////////////
	 Current_Units currentUnits = *new Current_Units();
	Current_Buildings currentBuilding = *new Current_Buildings();
	Wait_List wait = *new Wait_List();
	Unit_Hash terranUnits = Unit_Hash("terrunits.csv");
	Building_Hash terranBuilding = Building_Hash("terrbuild.csv");
	Resources resources = *new Resources(50, 0, 11, 0);
	Current_Buildings_Build_List currentBuildingBuildList =
			*new Current_Buildings_Build_List();
	Current_Units_Build_List currentUnitBuildList =
			*new Current_Units_Build_List();

	////////////////////////////////////////////////////////////////////////

	Unit unit;
	ofstream savefile("teran.txt");


	unit = terranUnits.getItem("scv");

	for(int i=0;i<6;i++)
	{
	currentUnits.addItem(unit, 1);
	}
	Building nex = *new Building();
	resources.setused_supply(6);


	nex = terranBuilding.getItem("command_center");
	currentBuilding.addItem(nex);

	///////////////////////////////////////////////////////////////////
	ifstream file2;
	file2.open(listfile);

	string lineitem;

	//terranUnits.print();


	bool validList = true;

	while (!file2.eof()) {


		getline(file2, lineitem);


		if(lineitem=="")
			continue;
		std::stringstream ss2(lineitem);

		getline(ss2, lineitem);

		string type = "";
		Dependancy dependancy;

		bool depancytest = true;

		if (terranUnits.search(lineitem)) {

			type = "unit";
			Unit u = terranUnits.getItem(lineitem);

			dependancy = u.upd->dependancy;




		} else if (terranBuilding.search(lineitem)) {

			type = "building";

			Building b = terranBuilding.getItem(lineitem);

			dependancy = b.upd->dependancy;
			dependancy.print();

		} else {


			depancytest = false;

		}

		if (!dependancy.search("no")) {

			depancytest = false;

			for (int i = 0; i < wait.presentL.size(); i++) {

				if (wait.searchDependancy(dependancy)) {

					depancytest = true;
					break;
				}

			}

		}

		if (depancytest) {


			wait.addItem(lineitem, type);


		} else {

			dependancy.print();
		//	cout<<"NOW "<<lineitem<<" This";
			//wait.print();

			cout << "{" << "\n";
			cout << "\"buildlistValid\":0," << "\n";
			cout << "\"game\":\"sc2-hots-terran\"" << "\n";

			cout << "}";

			return 0;
		}

	}


	cout << "{" << "\n";
			cout << "\"buildlistValid\":1," << "\n";
			cout << "\"game\":\""<<race<<"\"" << ",\n";
			cout << "\"messages\":[" << "\n";

	file2.close();
////////////////////////////////////////////////////////////////////////////////////////

	int t = 1;
	int isfirst = 1;

	while (wait.get_number_wait() > 0
			or currentBuildingBuildList.get_number_current_building() > 0
			or currentUnitBuildList.get_number_current_unit() > 0)

    {


		string result_events = "";
		resources.setminerals(resources.getminerals()+(0.7 * currentUnits.getMineralCollectorNumber()));


		resources.setvespene(resources.getvespene()+ (0.35 * currentUnits.getGasCollectorNumber()));

		if (t > 1) {

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


			string resultB = currentBuildingBuildList.update();

			if (resultB != "") {


				vector<Building> finish =
						currentBuildingBuildList.get_finish_building();
				for (int i = 0; i < finish.size(); i++) {

					Building buildtemp = finish[i];
					if (result_events != "") {
						result_events += ",";
					}

					result_events += "\n";
					result_events += "{";
					result_events += "\"type\":\"build-end\",";
					result_events += "\n";
					result_events += "\"name\":";
					result_events += "\"";
					result_events += buildtemp.naam;
					result_events += "\"";
					result_events += "\n";
					result_events += "}";

					string buildby = buildtemp.madeby;

					if (terranBuilding.search(buildby)) {

						if (buildby == "command_center") {

							currentBuilding.delet(buildby);

						} else {

							currentBuilding.freeBuilding(buildby);

						}

					} else {

						currentUnits.freewoker(buildby);

					}

					if (buildtemp.naam == "refinery") {

						currentUnits.addGasCollector(3);

					}
					currentBuilding.addItem(buildtemp);

					if (buildtemp.upd->getsupply_provided() > 0
							& buildby != "command_center") {

                    resources.setsupply(resources.getsupply()+buildtemp.upd->getsupply_provided());
					}

				}

			}
			/////////////////////////////////////////////////////////////////////////////////////////////////////////


			string resultU = currentUnitBuildList.update();

			if (resultU != "") {



				vector<Unit> finish = currentUnitBuildList.get_finish_unit();

                  int i = 0;
				while ( i < finish.size()) {

					Unit u = finish[i];

					if (result_events != "") {

						result_events += ",";
					}
					result_events += "\n";
					result_events += "{";
					result_events += "\"type\":\"build-end\",";
					result_events += "\n";
					result_events += "\"name\":";
					result_events += "\"";
					result_events += u.naam;
					result_events += "\"";
					result_events += "\n";
					result_events += "}";
					currentBuilding.freeBuilding(u.madeby);
					if (u.naam == "scv") {

						currentUnits.addItem(u, 1);

					} else {

						currentUnits.addItem(u, 0);

					}

				++i;}

			}
		}

		bool start_event = false;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		if (wait.get_number_wait() > 0
				and wait.get_first_wait_list_type() == "building") {

			string namefirst = wait.get_first_wait();

			Building tempbuild = terranBuilding.getItem(namefirst);

			if (tempbuild.upd->getminerals() <= resources.getminerals()
					&& tempbuild.upd->getvespene() <= resources.getvespene()) {

				if (tempbuild.upd->dependancy.search("no")
						or currentBuilding.searchDependancy(
								tempbuild.upd->dependancy))

								{

					bool canUsed = false;
					int type = 0; // mean unit

					ProduceBy produce_by = tempbuild.upd->produced_by;

					if (terranBuilding.search(produce_by.presentL[0])) {

						type = 1;

						canUsed = currentBuilding.availableBuilding(
								produce_by.presentL[0]);

					} else {

						canUsed = (currentUnits.numUnuits(
								produce_by.presentL[0]) > 0);

					}
					if (canUsed) {

						tempbuild.madeby = produce_by.presentL[0];
						currentBuildingBuildList.addItem(tempbuild);

						if (type == 0) {

							currentUnits.useworker(produce_by.presentL[0]);

						} else {

							currentBuilding.useBuilding(
									produce_by.presentL[0]);

						}

						if (result_events != "") {

							result_events += ",";
						}
						result_events += "\n";
						result_events += "{";
						result_events += "\"type\":\"build-start\",";
						result_events += "\n";
						result_events += "\"name\":";
						result_events += "\"";
						result_events += tempbuild.naam;
						result_events += "\"";
						result_events += "\n";
						result_events += "}";

						resources.setminerals(resources.getminerals()-tempbuild.upd->getminerals());
						resources.setvespene( resources.getvespene()- tempbuild.upd->getvespene() );
						wait.remove_first_wait();
						start_event = true;
					}
				}

			}

		}
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		if (!start_event and wait.get_number_wait() > 0
				and wait.get_first_wait_list_type() == "unit") {

			string namefirst = wait.get_first_wait();

			Unit tempbuild = terranUnits.getItem(namefirst);

			if (tempbuild.upd->getminerals() <= resources.getminerals()
					&& tempbuild.upd->getvespene() <= resources.getvespene()) {

				if (tempbuild.upd->dependancy.search("no")
						or currentBuilding.searchDependancy(
								tempbuild.upd->dependancy)) {

					ProduceBy produce_by_u = tempbuild.upd->produced_by;

					string buildby = currentBuilding.aviableBuildingProduceBy(
							produce_by_u);

					if (buildby != "no") {

						tempbuild.madeby = buildby;
						currentUnitBuildList.addItem(tempbuild);
						resources.setused_supply(resources.getused_supply()+tempbuild.upd->getsupply_cost());
						if (result_events != "") {

							result_events += ",";
						}
						result_events += "\n";

						result_events += char(123);
						result_events += +" \"type\": \"build-start\",";
						//	result_events += "\n";
						result_events += "  \"name\":";

						result_events += "\"";
						result_events += tempbuild.naam;
						result_events += "\"";

						// result_events+="\"";
						result_events += "\n";
						result_events += "}";
						currentBuilding.useBuilding(buildby);
						resources.setminerals(resources.getminerals()-tempbuild.upd->getminerals());
						resources.setvespene( resources.getvespene()- tempbuild.upd->getvespene() );
                        wait.remove_first_wait();

					}

				}

			}

		}

		if (result_events != "") {

			if (isfirst == 1) {

				isfirst = 0;
			} else {

				cout << ",";
			}
			cout << "{\n";
			cout << "\"time\":" << t << ",\n";
			cout << "\"status\":{\n";
			cout << "\"workers\":{\n";
			cout << "\"vespene\":" << currentUnits.getGasCollectorNumber();
			cout << ",\n";
			cout << "\"minerals\":" << currentUnits.getMineralCollectorNumber()<< "\n";
			cout << "},\n";

			cout << "\"resources\":{\n";
			cout << "\"vespene\":" << floor(resources.getvespene() + 0.0001)	<< ",\n";
			cout << "\"minerals\":" << floor(resources.getminerals() + 0.0001)	<< ",\n";
			cout << "\"supply-used\":" << resources.getused_supply() << ",\n";
			cout << "\"supply\":" << resources.getsupply() << "\n";
			cout << "}\n";
			cout << "},\n";

			cout << "\"events\":[\n";
			cout << result_events << "\n";
			cout << "] \n";
			cout << "}";

		}
		t++;

	}

	cout << "]" << "\n";

	cout << "}" << "\n";

	return 0;
}
