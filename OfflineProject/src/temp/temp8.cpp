				cout <<"id_device: " << id_device << endl;
				cout <<"id_class 0: " << myRes->getString(1) << endl;
				//myStmt1 = myConn->createStatement();
				myRes1 = myStmt1->executeQuery("SELECT studentattendance.mssv_student FROM studentattendance WHERE studentattendance.id_schedule = '" + id_device + "' AND studentattendance.id_class = '" + myRes->getString(1) + "' ;");	
				cout <<"id_class 1: " << myRes->getString(1) << endl;
				while (myRes1->next())
				{
					//std::vector<int> year = cvtStr2Int(myRes->getString(3),"-");
					cout <<"id_class 2: " << myRes->getString(1) << endl;
					cout <<"mssv_student: " << myRes1->getString(1) << endl;
					cout <<"---------------------------------------"<< endl;

					/*fstream file;
					std::string line;
					std::vector<std::string> tmp_data;

					file.open("../data/data_temp_from_sqlserver.csv", ios::in);
					while (getline( file, line,'\n'))
					{
						  istringstream templine(line); 
						  string data;
						  while (getline( templine, data,','))
						  {
						  	tmp_initsql.push_back(data.c_str());
						  }
					}
					file.close();*/
					std::this_thread::sleep_for (std::chrono::seconds(2));
				}
				delete myStmt1;
				delete myRes1;	




		//cout <<"date: " << now->tm_mday << endl;
		//cout <<"month: " << now->tm_mon + 1 << endl;
		//cout <<"year: " << now->tm_year+ 1900 << endl;

				/*file << i + ",";
				file << myRes->getString(1) + ",";
				file << myRes->getString(2) + ",";
				file << myRes->getString(3) + ",";
        			file << myRes->getString(4) + "\n";*/
				i++;
