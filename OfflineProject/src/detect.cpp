#include "../header/UltraFace.hpp"
#include "../header/student.hpp"
#include <time.h>
#include <ctime>
#include <thread>

#include <atomic>
#include <deque>

#include "mysql_connection.h"
#include <unistd.h>
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>

#include <iostream>
#include <stdio.h>
#include "mysql_driver.h"

using namespace sql;

//Mutex for thread synchronization
static pthread_mutex_t foo_mutex = PTHREAD_MUTEX_INITIALIZER;
atomic<int> status(-3);
atomic<int> command(0);

struct thread_data 
{
  std::string gst_pipeline;
  std::vector<student>* temp_lst;
  int  thread_id;
  string window_title; //Unique window title for each thread
};

/**************************************************************************************************************************************************************************/
/*                                                                        Capture And Detec MULTITHREAD                                                                   */
/**************************************************************************************************************************************************************************/

void *CaptureAndDetec(void *threadarg)
{
	struct thread_data *data;
	data = (struct thread_data *) threadarg;

	//Safely open video stream
	pthread_mutex_lock(&foo_mutex);
	cv::VideoCapture cap(data->gst_pipeline, cv::CAP_GSTREAMER);
	pthread_mutex_unlock(&foo_mutex);

	if( !cap.isOpened())
	{
	std::cout<<"Not good, open camera failed"<<std::endl;
	return 0;
	}
	std::cout<< "Opened camera successfully!"<<std::endl;

	shape_predictor sp;
	deserialize("../Model/shape_predictor_68_face_landmarks.dat") >> sp;
	anet_type net;
	deserialize("../Model/dlib_face_recognition_resnet_model_v1.dat") >> net;
	student temp_std;
	UltraFace ultraface("../Model/RFB-320.bin", "../Model/RFB-320.param", 432, 240, 64, 0.82);
	std::map<std::string, dlib::matrix<float, 0, 1>> data_faces; 
	deserialize("../dataface/data_faces.dat") >> data_faces;

	auto m_StartTime = std::chrono::system_clock::now();
	double FPS = cap.get(cv::CAP_PROP_FPS);
	std::cout << "Capture  " << FPS << " FPS " <<std::endl;
	image_window win;
	win.set_title(data->window_title);
	int framesSkipping = 0;
	while (true)
	{
		cv::Mat img;
		if (!cap.read(img))
		{
			std::cout << "Capture read error" << std::endl;
			break;
		}
		cv::resize(img, img, cv::Size(720,405));
		double fps = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count();
		m_StartTime = std::chrono::system_clock::now();
		cv::putText(img, to_string(static_cast<int>(1000/fps)) + " FPS", cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255), 1, false);

		cv::Mat image_clone = img.clone();
		ncnn::Mat inmat = ncnn::Mat::from_pixels(image_clone.data, ncnn::Mat::PIXEL_BGR2RGB, image_clone.cols, image_clone.rows);
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		std::vector<FaceInfo> face_info;
		//ultraface.detect(inmat, face_info);
		ncnn::Mat score_blob32; 
		ncnn::Mat bbox_blob32;
		ncnn::Mat score_blob16;
		ncnn::Mat bbox_blob16;
		if(framesSkipping == 2)
		{
			ultraface.detect(inmat, score_blob32,bbox_blob32,score_blob16,bbox_blob16);

			const float prob_threshold = 0.8f;
			const float nms_threshold = 0.4f;
			std::vector<FaceInfo> faceproposals;

			//32
			int base_size = 16;
			int feat_stride = 32;
			ncnn::Mat ratios(1);
			ratios[0] = 1.f;
			ncnn::Mat scales(2);
			scales[0] = 32.f;
			scales[1] = 16.f;
			ncnn::Mat anchors = ultraface.generate_anchors(base_size, ratios, scales);
			std::vector<FaceInfo> faceobjects32;
			ultraface.generate_proposals(anchors, feat_stride, score_blob32, bbox_blob32, prob_threshold, faceobjects32);

			faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());

			//16
			feat_stride = 16;
			scales[0] = 8.f;
			scales[1] = 4.f;
			anchors = ultraface.generate_anchors(base_size, ratios, scales);
			std::vector<FaceInfo> faceobjects16;
			ultraface.generate_proposals(anchors, feat_stride, score_blob16, bbox_blob16, prob_threshold, faceobjects16);

			faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());
		    
			ultraface.nms(faceproposals, face_info);
			framesSkipping = -1;
		}
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		cv_image<bgr_pixel> cimg(img);
		matrix<rgb_pixel> matrix;
		assign_image(matrix, cimg);
		win.clear_overlay();
		win.set_image(cimg);
		std::vector<dlib::matrix<rgb_pixel>> faces;
		for (int i = 0; i < face_info.size(); i++)
		{
			auto face = face_info[i];
			rectangle rect(point(face.x1,face.y1), point(face.x2, face.y2));
			auto shape = sp(matrix,rect);
			dlib::matrix<rgb_pixel> face_chip;
			extract_image_chip(matrix, get_face_chip_details(shape, 150, 0.25), face_chip);
			faces.push_back(move(face_chip));
	    		std::vector<dlib::matrix<float, 0, 1>> face_descriptors = net(faces);
			string _ID = "Unknow";
			rgb_pixel Bound_style = rgb_pixel(0, 0, 255);
			double distance = 0.15f;
			for(auto& x:data_faces )
			{
			    double prob = ultraface.SubVector(face_descriptors[0], x.second);
			    if(prob < distance)
			    {
			        distance = prob;
				_ID = x.first;
				Bound_style = rgb_pixel(0, 255, 0);
			    }
			}
			faces.clear();
			image_window::overlay_rect orect(rect, Bound_style, _ID);
			win.add_overlay(orect);
		}
		face_info.clear();
		framesSkipping++;
	}
	//Release VideoCapture object
	cap.release();
	//Destroy previously created window
	cv::destroyAllWindows();

	//Exit thread
	pthread_exit(NULL);
}

std::vector<std::string> split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == string::npos) pos = str.length();
        string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

std::vector<int> cvtStr2Int(const std::string& str, const std::string& delim)
{
    std::vector<string> vt_tmp = split(str,delim);
    return std::vector<int>{std::stoi(vt_tmp[0]), std::stoi(vt_tmp[1]), std::stoi(vt_tmp[2])};
    
}

void get_and_update_mysql(  string& host_sv,
                            string& user_sv,
                            string& password_sv,
                            string& schema_sv, 
                            string& id_device,
                            string& host_lc,
                            string& user_lc,
                            string& password_lc,
                            string& schema_lc)
{
    
    while(true)
    {
        try 
	{
		std::time_t t = std::time(0);
		std::tm* now = std::localtime(&t);

		sql::Driver *myDriver;
		sql::Connection *myConn;
		sql::Statement *myStmt;
		sql::ResultSet *myRes;
		sql::PreparedStatement *ps;

		fstream file;
		fstream file_studentattendance;
		std::string line;
		std::vector<std::string> tmp_data;

		myDriver = get_driver_instance();
		myConn = myDriver->connect(host_lc, user_lc, password_lc);
		myConn->setSchema(schema_lc);
		myStmt = myConn->createStatement();
		myRes = myStmt->executeQuery("SELECT diemdanh.date FROM diemdanh WHERE diemdanh.id_room = '" + id_device + "';");
		
		if(!myRes->next())
		{
			myConn->close();
			delete myConn;
			delete myStmt;
			delete myRes;
			int i = 1;
			myDriver = get_driver_instance();
			myConn = myDriver->connect(host_sv, user_sv, password_sv);
			myConn->setSchema(schema_sv);
			myStmt = myConn->createStatement();
			myRes = myStmt->executeQuery("SELECT schedule.id_class, schedule.date, schedule.start, schedule.end FROM schedule WHERE schedule.id_room = '" + id_device + "';");
			file.open("../data/data_schedule.csv", ios::out);
			while (myRes->next())
			{
				file << std::to_string(i)   + ",";
				file << myRes->getString(1) + ",";
				file << myRes->getString(2) + ",";
				file << myRes->getString(3) + ",";
        			file << myRes->getString(4) + "\n";
				i++;
			}
			file.close();
			myConn->close();
			delete myConn;
			delete myStmt;
			delete myRes;
			i = 1;
			file.open("../data/data_schedule.csv", ios::in);
			file_studentattendance.open("../data/studentattendance.csv", ios::out);
			while (getline( file, line,'\n'))
			{
				istringstream templine(line); 
				string data;
				while (getline( templine, data,','))
				{
					tmp_data.push_back(data.c_str());
				}
				myDriver = get_driver_instance();
				myConn = myDriver->connect(host_sv, user_sv, password_sv);
				myConn->setSchema(schema_sv);
				myStmt = myConn->createStatement();
				myRes = myStmt->executeQuery("SELECT studentattendance.mssv_student FROM studentattendance WHERE studentattendance.id_schedule = '" + id_device + "' AND studentattendance.id_class = '" + tmp_data[1] + "' ;");
				while (myRes->next())
				{
					file_studentattendance << std::to_string(i) + ",";
					file_studentattendance << tmp_data[1] + ",";
					file_studentattendance << tmp_data[2] + ",";
					file_studentattendance << tmp_data[3] + ",";
					file_studentattendance << tmp_data[4] + ",";
					file_studentattendance << myRes->getString(1) + "\n";
					i++;	
				}
				tmp_data.clear();
				myConn->close();
				delete myConn;
				delete myStmt;
				delete myRes;
			}
			file.close();
			file_studentattendance.close();

			cout << "DONE GET DATA FROM SERVER " << endl;

			file.open("../data/studentattendance.csv", ios::in);
			while (getline( file, line,'\n'))
			{
				istringstream templine(line); 
				string data;
				while (getline( templine, data,','))
				{
					tmp_data.push_back(data.c_str());
				}
				try 
				{
					myDriver = get_driver_instance();
					myConn = myDriver->connect(host_lc, user_lc, password_lc);
					myConn->setSchema(schema_lc);
					ps = myConn->prepareStatement("INSERT INTO diemdanh(number,id_room, id_class, date, start, end, mssv_student, status_0, status_1, status_2, status_3, status_4, status_5, status_6, status_7, status_8, status_9, status_10, status_11, status_12, status_13, status_14) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)");

					ps->setInt(1,std::stoi(tmp_data[0]));
					ps->setString(2,id_device);
					ps->setString(3,tmp_data[1]);
					ps->setString(4,tmp_data[2]);
					ps->setString(5,tmp_data[3]);
					ps->setString(6,tmp_data[4]);
					ps->setString(7,tmp_data[5]);
					ps->setInt(8,0);
					ps->setInt(9,0);
					ps->setInt(10,0);
					ps->setInt(11,0);
					ps->setInt(12,0);
					ps->setInt(13,0);
					ps->setInt(14,0);
					ps->setInt(15,0);
					ps->setInt(16,0);
					ps->setInt(17,0);
					ps->setInt(18,0);
					ps->setInt(19,0);
					ps->setInt(20,0);
					ps->setInt(21,0);
					ps->setInt(22,0);

					ps->executeUpdate();
					tmp_data.clear();
					myConn->close();
					delete ps;
					delete myConn;
				}
				catch (sql::SQLException &e) 
				{
				    cout << "# ERR: " << e.what();
				    cout << " (MySQL error code: " << e.getErrorCode();
				    cout << ", SQLState: " << e.getSQLState() << " )" << endl;
				}
			}
			file.close();
			cout << "DONE UPDATE DATA TO LOCAL " << endl;
		}
		std::this_thread::sleep_for (std::chrono::seconds(10));
        } 
        catch (sql::SQLException &e) {
		cout << "# ERR: " << e.what();
		cout << " (MySQL error code: " << e.getErrorCode();
		cout << ", SQLState: " << e.getSQLState() << " )" << endl;
        }

    }
    
}

bool updata_status_mysql_server(    string& host,
		                    string& user,
		                    string& password,
		                    string& schema,
		                    const string name, 
		                    cv::Mat& img, 
		                    double distace, 
		                    const string date)
{
    

	try 
	{

	} 
        catch (sql::SQLException &e) 
	{
		cout << "# ERR: " << e.what();
		cout << " (MySQL error code: " << e.getErrorCode();
		cout << ", SQLState: " << e.getSQLState() << " )" << endl;
		return false;
        }
    return true;
}

bool check_internet()
{

    if (system("ping -c1 -s1 www.google.com"))
    {
        int tmp = status.load(std::memory_order_acquire);
        if(tmp != 4 || tmp != 5 || tmp != 6 || tmp != -2)
        {
            cout<<"No internet connection  \n";
            status.store(4, std::memory_order_release);
            return false;
        }
    }
    else
    {
        int tmp = status.load(std::memory_order_acquire);
        if(tmp != 0 || tmp != -1 || tmp != 1 || tmp != 2 || tmp != 3|| tmp != -2)
        {
            cout<<"Internet connection  \n";
            status.store(0, std::memory_order_release);
            return true;
        }
    }
    usleep(10000000);
}

/**************************************************************************************************************************************************************************/
/*                                                                                   MAIN                                                                                 */
/**************************************************************************************************************************************************************************/

int main(void)
{
	//Thread check internet
	thread check_internet_thread(check_internet);

	//Init config
	std::string initsql;
	std::string host_sv;
	std::string user_sv;
	std::string password_sv;
	std::string schema_sv;
    	std::string id_device;
	std::string host_lc;
	std::string user_lc;
	std::string password_lc;
	std::string schema_lc;


	sql::Driver *myDriver;
	sql::Connection *myConn;
	sql::Statement *myStmt;
	sql::ResultSet *myRes;
	sql::PreparedStatement *ps;

	if (!fs::exists("../data/initsql.csv"))
	{
		cout << "Please check initsql file " << endl;
	}
	else
	{
		fstream file;
		std::string line;
		std::vector<std::string> tmp_initsql;

		file.open("../data/initsql.csv", ios::in);
		while (getline( file, line,'\n'))
		{
			  istringstream templine(line); 
			  string data;
			  while (getline( templine, data,','))
			  {
			  	tmp_initsql.push_back(data.c_str());
			  }
		}
		file.close();

		host_sv = tmp_initsql[0];
		user_sv = tmp_initsql[1];
		password_sv = tmp_initsql[2];
		schema_sv = tmp_initsql[3];
		id_device = tmp_initsql[4];
		host_lc = tmp_initsql[5];
		user_lc = tmp_initsql[6];
		password_lc = tmp_initsql[7];
		schema_lc = tmp_initsql[8];
	}
	cout<< "Host sever: "<<host_sv<<endl;
	cout<< "User sever: "<<user_sv<<endl;
	cout<< "Pass sever: "<<password_sv<<endl;
	cout<< "Schema sever: "<<schema_sv<<endl;
	cout<< "id_device: "<<id_device<<endl;
	cout<< "Host local: "<<host_lc<<endl;
	cout<< "User local: "<<user_lc<<endl;
	cout<< "Pass local: "<<password_lc<<endl;
	cout<< "Schema local: "<<schema_lc<<endl;

	//Wait to check internet
	while(status.load(std::memory_order_acquire) == -3);
	check_internet_thread.join();

	thread get_and_update_mysql_thread (get_and_update_mysql, std::ref(host_sv),
		                                            	  std::ref(user_sv),
		                                            	  std::ref(password_sv),
		                                            	  std::ref(schema_sv),
		                                            	  std::ref(id_device),
								  std::ref(host_lc),
		                                            	  std::ref(user_lc),
		                                            	  std::ref(password_lc),
		                                            	  std::ref(schema_lc));


/*
	std::time_t t = std::time(0);
	std::tm* now = std::localtime(&t);

	cout <<"date: " << now->tm_mday << endl;
	cout <<"month: " << now->tm_mon + 1 << endl;
	cout <<"year: " << now->tm_year+ 1900 << endl;

	myDriver = get_driver_instance();
        myConn = myDriver->connect("tcp://156.67.222.106:3306", "u477501821_duyle", "Duyle22697");
        myConn->setSchema("u477501821_diemdanhuit");
        myStmt = myConn->createStatement();
        myRes = myStmt->executeQuery("SELECT schedule.id_class, schedule.id_room, schedule.date, schedule.start, schedule.end FROM schedule WHERE schedule.id_room = 1 ;");
        while (myRes->next())
        {
            std::vector<int> year = cvtStr2Int(myRes->getString(3),"-");
	    cout <<"year: " << year[0] - (now->tm_year+ 1900) << endl;
            //class_list_today.push_back(temp_class);
        }
        myConn->close();
        delete myConn;
        delete myStmt;
        delete myRes;
*/
/*try {
	myDriver = get_driver_instance();
	myConn = myDriver->connect("tcp://127.0.0.1:3306", "root", "1997"); //IP Address, user name, password
	myConn->setSchema("LOCAL_DB");

	ps = myConn->prepareStatement("INSERT INTO schedule(number,id_room, id_class, date, start, end) VALUES (?,?,?,?,?,?)");

	ps->setString(1,"test");
	ps->setDouble(2,2);
	ps->setDouble(3,4);
	ps->setString(4,"test");
	ps->setString(5,"test");
	ps->setString(6,"test");
	ps->executeUpdate();
	delete ps;
	delete myConn;
}
        catch (sql::SQLException &e) {
            cout << "# ERR: " << e.what();
            cout << " (MySQL error code: " << e.getErrorCode();
            cout << ", SQLState: " << e.getSQLState() << " )" << endl;
}
	myDriver = get_driver_instance();
	myConn = myDriver->connect("tcp://156.67.222.106:3306", "u477501821_duyle", "Duyle22697");
	myConn->setSchema("u477501821_diemdanhuit");
	myStmt = myConn->createStatement();

	std::string test = "UPDATE studentattendance SET status = 0 WHERE mssv_student = '15520145' AND id_class = 'IT001' ;";
	myStmt->executeUpdate(test);

	myConn->close();
	delete myConn;
	delete myStmt;

	myDriver = get_driver_instance();
        myConn = myDriver->connect("tcp://127.0.0.1:3306", "root", "1997");
        myConn->setSchema("LOCAL_DB");
        myStmt = myConn->createStatement();
        myRes = myStmt->executeQuery("SELECT schedule.number FROM schedule WHERE schedule.number = A.101 ;");
        while (myRes->next())
        {
            temp_class.set_date(myRes->getString(1));
            //class_list_today.push_back(temp_class);
        }
        myConn->close();
        delete myConn;
        delete myStmt;
        delete myRes;*/
/*
    const int thread_count = 2;

    pthread_t threads[thread_count];
    struct thread_data td[thread_count];

    //Read list Student
    student temp_student;
    std::vector<student> m_temp_lst;
    //temp_student.ReadListStu(temp_student, m_temp_lst);

    //Initialize thread data beforehand
    //td[0].gst_pipeline = 1;
    td[0].gst_pipeline = "v4l2src device=/dev/video1 ! image/jpeg, width=(int)1280, height=(int)720, framerate=30/1 ! jpegdec ! videoconvert ! appsink";
    td[0].window_title = "CAM 1 ";
    //td[0].temp_lst = &m_temp_lst;

    //td[1].gst_pipeline = 0;61612/513
    td[1].gst_pipeline = "v4l2src device=/dev/video0 ! image/jpeg, width=(int)1280, height=(int)720, framerate=60/1 ! jpegdec ! videoconvert ! appsink";
    td[1].window_title = "CAM 2 ";
    //td[1].temp_lst = &m_temp_lst;


    int rc = 0;
    for( int i = 0; i < thread_count; i++ ) 
    {
        cout <<"main() : creating thread, " << i << endl;
        td[i].thread_id = i;

        rc = pthread_create(&(threads[i]), NULL, CaptureAndDetec, (void *)& (td[i]));

        if (rc) 
        {
            cout << "Error:unable to create thread," << rc << endl;
            exit(-1);
        }
    }

    //Wait for the previously spawned threads to complete execution
    for( int i = 0; i < thread_count; i++ )
        pthread_join(threads[i], NULL);

    pthread_exit(NULL);
*/
	get_and_update_mysql_thread.join();
	command.store(-1, std::memory_order_release);
	return 0;
}









































/*
host = "tcp://156.67.222.106:3306";
user = "u477501821_duyle";
password = "Duyle22697";
schema = "u477501821_diemdanhuit";
id_device = "1";
initsql = host + "," + user + "," + password + "," + schema + "," + id_device;
serialize("../init/initsql.txt") << initsql;
*/
