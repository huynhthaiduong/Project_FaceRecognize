/*#include "../header/UltraFace.hpp"
#include "../header/student.hpp"
#include <time.h>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <iostream>
#include <fstream>

#include <dlib/threads.h>
#include <dlib/misc_api.h>  // for dlib::sleep
#include <dlib/logger.h>
#include <deque>

#include <map>

using namespace std;
using namespace dlib;

struct Match_object
{
    std::string ID;
    double Avg_value;
    double m_checkAvgValue_d = 0.15f;
    cv::Scalar Bound_style;
    int check_index;
};

class my_object : public multithreaded_object
{
public:
	//Read list Student
	student temp_student;
	std::vector<student> temp_lst;
	//std::map<std::string, std::vector<float>> data_faces;
	bool done = false;
	cv::VideoCapture cap1, cap2;
	my_object(cv::VideoCapture t_cap1, cv::VideoCapture t_cap2)
	{
		cap1 = t_cap1;
		cap2 = t_cap2;
		// register which functions we want to run as threads.  We want one thread running
		// thread1() and two threads to run thread2().  So we will have a total of 2 threads
		// running.
		//initStu(temp_student,temp_lst,data_faces);
		clear();
		register_thread(*this,&my_object::thread1);
		register_thread(*this,&my_object::thread2);

		// start all our registered threads going by calling the start() function
		start();
	}	

	~my_object()
	{
		// Tell the thread() function to stop.  This will cause should_stop() to 
		// return true so the thread knows what to do.
		stop();

		// Wait for the threads to stop before letting this object destruct itself.
		// Also note, you are *required* to wait for the threads to end before 
		// letting this object destruct itself.
		wait();
	}

private:
	void check_cam(std::string& gst_pipeline_1, std::string& gst_pipeline_2)
	{

	}

	void check_face()
	{

	}
	
	void initStu(student& temp_student, std::vector<student>& temp_lst, std::map<std::string, std::vector<float>> &data_faces)
	{
		temp_student.ReadListStu(temp_student, temp_lst, data_faces);
	}

	void checkstu(std::vector<student> temp_lst)
	{
		for (int i = 0; i < temp_lst.size(); i++)
		{
			std::cout << "Sinh vien: " << temp_lst[i].student_name << std::endl;
			std::cout << "     MSSV: " << temp_lst[i].student_id << std::endl;
			if (temp_lst[i].checked == 1)
			{
		    		std::cout << "      Co" << std::endl;
			}
			else
			{
		    		std::cout << "      Vang" << std::endl;
			}
		}	
	}

	float calc_prob(std::vector<float>& a, std::vector<float>& b)
	{
		float prob = 0;
		for(int i =0; i < b.size();i++)
		{
			prob += a[i]*b[i]; 
		}
		return prob;
	}

	float calc_prob2(std::vector<float>& a, std::vector<float>& b)
	{
		float prob = 0;
		for(int i =0; i < b.size();i++)
		{
			prob += a[i]*b[i]; 
		}
		return prob;
	}
    
	void thread1()
	{
		std::cout<<"  ***************************************************************"<<std::endl;
    		std::cout<<" *                                                               *"<<std::endl;
    		std::cout<<"*                  Opened camera 1 successfully!                  *"<<std::endl;
    		std::cout<<" *                                                               *"<<std::endl;
    		std::cout<<"  ***************************************************************"<<std::endl;

		//Create window with unique title
		cv::namedWindow("Camera 1", cv::WINDOW_AUTOSIZE);

		shape_predictor sp;
		deserialize("../Model/shape_predictor_68_face_landmarks.dat") >> sp;
		anet_type net;
		deserialize("../Model/dlib_face_recognition_resnet_model_v1.dat") >> net;
		UltraFace ultraface("../Model/RFB-320.bin", "../Model/RFB-320.param", 432, 240, 64, 0.82);
		std::map<std::string, std::vector<float>> data_faces;
		deserialize("../dataface/data_faces.dat") >> data_faces;

		auto m_StartTime = std::chrono::system_clock::now();
		double FPS = cap1.get(cv::CAP_PROP_FPS);
		std::cout << "Capture  " << FPS << " FPS " <<std::endl;
		double juge = 0;
		Match_object temp_obj;
		while (true)
		{
			cv::Mat img;
			if (!cap1.read(img))
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

			std::vector<FaceInfo> face_info;
			ultraface.detect(inmat, face_info);

			cv_image<bgr_pixel> cimg(img);
			matrix<rgb_pixel> matrix;
			assign_image(matrix, cimg);
			std::vector<dlib::matrix<rgb_pixel>> faces;
			for (int i = 0; i < face_info.size(); i++)
			{
				temp_obj.Bound_style = cv::Scalar(0, 0, 255);
				temp_obj.ID = "Unknow";
				auto face = face_info[i];
				rectangle rect(point(face.x1,face.y1), point(face.x2, face.y2));
				auto shape = sp(matrix,rect);

				dlib::matrix<rgb_pixel> face_chip;
				extract_image_chip(matrix, get_face_chip_details(shape,112,0.15), face_chip);
				faces.push_back(move(face_chip));
				for (size_t i = 0; i < faces.size(); ++i)
		    		{
					ncnn::Mat tmp_ncnnmat = ncnn::Mat::from_pixels(toMat(faces[i]).data, ncnn::Mat::PIXEL_RGB, 112, 112);
					std::vector<float> out;
					ultraface.face_embedding(tmp_ncnnmat, out);
					string _ID = "";
					double distance = 0;

					for(auto& x:data_faces )
					{
					    float prob = calc_prob(x.second,out);
					    if(prob > distance)
					    {
						distance = prob;
						_ID = x.first;
					    }
		        		}
					if(distance >= 0.5)
					{
				    		temp_obj.ID = _ID;
				    		temp_obj.Bound_style = cv::Scalar(0, 255, 0);
					}
		    		}
				faces.clear();
				cv::rectangle(img, cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2), temp_obj.Bound_style, 1);
				//cv::putText(img, temp_obj.ID, cv::Point(face.x1, face.y2 - 10), cv::FONT_HERSHEY_DUPLEX, 1, temp_obj.Bound_style, 2, false);
			}
			face_info.clear();
			cv::imshow("Camera 1", img);
			if (cv::waitKey(1) == 27)
			{
				break;
			}
		}
//		checkstu(temp_lst);
		//Release VideoCapture object
		cap1.release();
		//Destroy previously created window
		cv::destroyWindow("Camera 1");
		done = true;
	}

	void thread2()
	{
		std::cout<<"  ***************************************************************"<<std::endl;
    		std::cout<<" *                                                               *"<<std::endl;
    		std::cout<<"*                  Opened camera 2 successfully!                  *"<<std::endl;
    		std::cout<<" *                                                               *"<<std::endl;
    		std::cout<<"  ***************************************************************"<<std::endl;

		//Create window with unique title
		cv::namedWindow("Camera 2", cv::WINDOW_AUTOSIZE);

		shape_predictor sp;
		deserialize("../Model/shape_predictor_68_face_landmarks.dat") >> sp;
		anet_type net;
		deserialize("../Model/dlib_face_recognition_resnet_model_v1.dat") >> net;
		UltraFace ultraface("../Model/RFB-320.bin", "../Model/RFB-320.param", 432, 240, 64, 0.82);
		std::map<std::string, std::vector<float>> data_faces;
		deserialize("../dataface/data_faces.dat") >> data_faces;
		cv::Mat outImg;
		auto m_StartTime = std::chrono::system_clock::now();
		double FPS = cap2.get(cv::CAP_PROP_FPS);
		std::cout << "Capture  " << FPS << " FPS " <<std::endl;
		double juge = 0;
		Match_object temp_obj;
		while (true)
		{
			cv::Mat img;
			if (!cap2.read(img))
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

			std::vector<FaceInfo> face_info;
			ultraface.detect(inmat, face_info);

			cv_image<bgr_pixel> cimg(img);
			matrix<rgb_pixel> matrix;
			assign_image(matrix, cimg);
			std::vector<dlib::matrix<rgb_pixel>> faces;
			for (int i = 0; i < face_info.size(); i++)
			{
				temp_obj.Bound_style = cv::Scalar(0, 0, 255);
				temp_obj.ID = "Unknow";
				auto face = face_info[i];
				rectangle rect(point(face.x1,face.y1), point(face.x2, face.y2));
				auto shape = sp(matrix,rect);

				dlib::matrix<rgb_pixel> face_chip;
				extract_image_chip(matrix, get_face_chip_details(shape,112,0.15), face_chip);
				faces.push_back(move(face_chip));

				for (size_t i = 0; i < faces.size(); ++i)
		    		{
					ncnn::Mat tmp_ncnnmat = ncnn::Mat::from_pixels(toMat(faces[i]).data, ncnn::Mat::PIXEL_RGB, 112, 112);
					std::vector<float> out;
					ultraface.face_embedding(tmp_ncnnmat, out);
					string _ID = "";
					double distance = 0;

					for(auto& x:data_faces )
					{
					    float prob = calc_prob2(x.second,out);
					    if(prob > distance)
					    {
						distance = prob;
						_ID = x.first;
					    }
		        		}
					if(distance >= 0.5)
					{
				    		temp_obj.ID = _ID;
				    		temp_obj.Bound_style = cv::Scalar(0, 255, 0);
					}
		    		}
				faces.clear();
				cv::rectangle(img, cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2), temp_obj.Bound_style, 1);
				//cv::putText(img, temp_obj.ID, cv::Point(face.x1, face.y2 - 10), cv::FONT_HERSHEY_DUPLEX, 1, temp_obj.Bound_style, 2, false);
				cv::imshow("Camera 2", img);
			}
			face_info.clear();
			cv::imshow("Camera 2", img);
			if (cv::waitKey(1) == 27)
			{
				break;
			}
		}
//		checkstu(temp_lst);
		//Release VideoCapture object
		cap2.release();
		//Destroy previously created window
		cv::destroyWindow("Camera 2");
		done = true;
	}

	bool check_internet()
	{

	    if (system("ping -c1 -s1 www.google.com"))
	    {
		int tmp = status.load(std::memory_order_acquire);
		if(tmp != 4 || tmp != 5 || tmp != 6 || tmp != -2)
		{
		    cout<<"There is no internet connection  \n";
		    status.store(4, std::memory_order_release);
		    return false;
		}
	    }
	    else
	    {
		int tmp = status.load(std::memory_order_acquire);
		if(tmp != 0 || tmp != -1 || tmp != 1 || tmp != 2 || tmp != 3|| tmp != -2)
		{
		    cout<<"There is internet connection  \n";
		    status.store(0, std::memory_order_release);
		    return true;
		}
	    }
	    usleep(10000000);
	}

};

void SetSystemTime()
{
    string newtime = "'10:42:43'";
    string cmdstop ="timedatectl set-ntp 0";
    string cmd ="timedatectl set-time ";
    string cmdstart ="timedatectl set-ntp 1";
    cmd += newtime;
    system(cmdstop.c_str()); //stop Automatic Date & Time
    system(cmd.c_str()); //set sysytem time
    //system(cmdstart.c_str()); //start Automatic Date & Time
}

std::string get_tegra_pipeline(int width, int height, int fps,int display_width,int display_height, int flip_method) {
return "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)" + std::to_string(width) + ", height=(int)" + std::to_string(height) + ",format=(string)NV12, framerate=(fraction)" + std::to_string(fps) + "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" + std::to_string(display_height) + ", format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink";
}
*/
/**************************************************************************************************************************************************************************/
/*                                                                                   MAIN                                                                                 */
/**************************************************************************************************************************************************************************/
/*int main(void)
{
    int capture_width = 640 ;
    int capture_height = 360 ;
    int display_width = 640 ;
    int display_height = 360 ;
    int framerate = 21 ;
    int flip_method = 2 ;

    std::string pipeline = gstreamer_pipeline(capture_width,
	capture_height,
	display_width,
	display_height,
	framerate,
	flip_method);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";
    std::string gst_pipeline_1 = "v4l2src device=/dev/video1 ! image/jpeg, width=(int)640, height=(int)360, framerate=30/1 ! jpegdec ! videoconvert ! appsink";
    std::string gst_pipeline_2 = "v4l2src device=/dev/video0 ! image/jpeg, width=(int)1280, height=(int)720, framerate=60/1 ! jpegdec ! videoconvert ! appsink";
 
    cv::VideoCapture cap1(gst_pipeline_1, cv::CAP_GSTREAMER);
    cv::VideoCapture cap2(gst_pipeline_2, cv::CAP_GSTREAMER);
    if((!cap1.isOpened()) || (!cap2.isOpened()) ) {
	std::cout<<"Failed to open camera."<<std::endl;
	return (-1);
    }
    UltraFace ultraface("../Model/RFB-320.bin", "../Model/RFB-320.param", 432, 240, 64, 0.82);
    my_object test(cap1,cap2);

    return 0;
}
*/


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/*
#include "../header/UltraFace.hpp"
#include "../header/student.hpp"
#include <time.h>
#include <thread>

//Mutex for thread synchronization
static pthread_mutex_t foo_mutex = PTHREAD_MUTEX_INITIALIZER;

struct thread_data 
{
  std::string gst_pipeline;
  std::vector<student>* temp_lst;
  int  thread_id;
  string window_title; //Unique window title for each thread
};

struct Match_object
{
    std::string ID;
    double Avg_value;
    double m_checkAvgValue_d = 0.15f;
    cv::Scalar Bound_style;
    int check_index;
};

float calc_prob(std::vector<float>& a, std::vector<float>& b)
{
	float prob = 0;
	for(int i =0; i < b.size();i++)
	{
		prob += a[i]*b[i]; 
	}
	return prob;
}
*/
/**************************************************************************************************************************************************************************/
/*                                                                        Capture And Detec MULTITHREAD                                                                   */
/**************************************************************************************************************************************************************************/
/*
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

	//Create window with unique title
	cv::namedWindow(data->window_title, cv::WINDOW_AUTOSIZE);

	shape_predictor sp;
	deserialize("../Model/shape_predictor_68_face_landmarks.dat") >> sp;
	anet_type net;
	deserialize("../Model/dlib_face_recognition_resnet_model_v1.dat") >> net;
	student temp_std;
	UltraFace ultraface("../Model/RFB-320.bin", "../Model/RFB-320.param", 432, 240, 64, 0.82);

	std::map<std::string, std::vector<float>> data_faces;
	deserialize("../dataface/data_faces.dat") >> data_faces;
	auto m_StartTime = std::chrono::system_clock::now();
	double FPS = cap.get(cv::CAP_PROP_FPS);
	std::cout << "Capture  " << FPS << " FPS " <<std::endl;
	double juge = 0;
	Match_object temp_obj;
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

		std::vector<FaceInfo> face_info;
		ultraface.detect(inmat, face_info);

		cv_image<bgr_pixel> cimg(img);
		matrix<rgb_pixel> matrix;
		assign_image(matrix, cimg);
		std::vector<dlib::matrix<rgb_pixel>> faces;
		for (int i = 0; i < face_info.size(); i++)
		{
			temp_obj.Bound_style = cv::Scalar(0, 0, 255);
			temp_obj.ID = "Unknow";
			auto face = face_info[i];
			rectangle rect(point(face.x1,face.y1), point(face.x2, face.y2));
			auto shape = sp(matrix,rect);

			dlib::matrix<rgb_pixel> face_chip;
			extract_image_chip(matrix, get_face_chip_details(shape,112,0.15), face_chip);
			faces.push_back(move(face_chip));

			for (size_t i = 0; i < faces.size(); ++i)
	    		{
				ncnn::Mat tmp_ncnnmat = ncnn::Mat::from_pixels(toMat(faces[i]).data, ncnn::Mat::PIXEL_RGB, 112, 112);
				std::vector<float> out;
				ultraface.face_embedding(tmp_ncnnmat, out);
				string _ID = "";
				double distance = 0;

				for(auto& x:data_faces )
				{
				    float prob = calc_prob(x.second,out);
				    if(prob > distance)
				    {
					distance = prob;
					_ID = x.first;
				    }
				}
				if(distance >= 0.5)
				{
			    		temp_obj.ID = _ID;
			    		temp_obj.Bound_style = cv::Scalar(0, 255, 0);
				}
	    		}
			faces.clear();
			cv::rectangle(img, cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2), temp_obj.Bound_style, 1);
			cv::putText(img, temp_obj.ID, cv::Point(face.x1, face.y2 - 10), cv::FONT_HERSHEY_DUPLEX, 1, temp_obj.Bound_style, 2, false);

		}
		face_info.clear();
		cv::imshow(data->window_title, img);
		if (cv::waitKey(1) == 27)
		{
			break;
		}


	}
    for (int i = 0; i < (*(data->temp_lst)).size(); i++)
    {
        std::cout << "Sinh vien: " << (*(data->temp_lst))[i].student_name << std::endl;
        std::cout << "      MSSV: " << (*(data->temp_lst))[i].student_id << std::endl;
        if ((*(data->temp_lst))[i].checked == 1)
        {
            std::cout << "      Co" << std::endl;
        }
        else
        {
            std::cout << "      Vang" << std::endl;
        }
    }
    //Release VideoCapture object
    cap.release();
    //Destroy previously created window
    cv::destroyWindow(data->window_title);

  //Exit thread
  pthread_exit(NULL);
}

*/
/**************************************************************************************************************************************************************************/
/*                                                                                   MAIN                                                                                 */
/**************************************************************************************************************************************************************************/
/*
int main(void)
{
    const int thread_count = 2;

    pthread_t threads[thread_count];
    struct thread_data td[thread_count];

    //Read list Student
    student temp_student;
    std::vector<student> m_temp_lst;
    //temp_student.ReadListStu(temp_student, m_temp_lst);

    //Initialize thread data beforehand
    //td[0].gst_pipeline = 1;
    td[0].gst_pipeline = "v4l2src device=/dev/video1 ! image/jpeg, width=(int)640, height=(int)360, framerate=30/1 ! jpegdec ! videoconvert ! appsink";
    td[0].window_title = "CAM 1 ";
    //td[0].temp_lst = &m_temp_lst;

    //td[1].gst_pipeline = 0;
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

    return 0;
}
*/


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#include <opencv2/opencv.hpp>

#include "mysql_connection.h"
#include <unistd.h>
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>

#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <map>
#include <string>
#include "../header/UltraFace.hpp"
#include "../header/student.hpp"
#include <time.h>
#include <thread>
#include <deque>

#include <math.h>
using namespace dlib;
using namespace std;

std::mutex m1;
std::mutex m2;

//Mutex for thread synchronization
static pthread_mutex_t foo_mutex = PTHREAD_MUTEX_INITIALIZER;

struct thread_data 
{
  std::string gst_pipeline;
  int  thread_id;
  shape_predictor sp;
  std::map<std::string, std::vector<float>> data_faces;
  std::deque<std::vector<matrix<rgb_pixel>>> buffer_faces;
  string window_title; //Unique window title for each thread
};

struct Match_object
{
    std::string ID;
    double Avg_value;
    double m_checkAvgValue_d = 0.15f;
    cv::Scalar Bound_style;
    int check_index;
};

float calc_prob(std::vector<float>& a, std::vector<float>& b)
{
    float prob = 0;
    for(int i =0; i < b.size();i++)
    {
        prob += a[i]*b[i]; 
    }
    return prob;
}

void capture_and_detect_1(cv::VideoCapture& cap, UltraFace& ultraface, shape_predictor& sp, anet_type& net, std::map<std::string, dlib::matrix<float, 0, 1>>& data_faces)
{
	//Create window with unique title
	cv::namedWindow("CAM 1", cv::WINDOW_AUTOSIZE);
	auto m_StartTime = std::chrono::system_clock::now();
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

		ncnn::Mat inmat = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);

		std::vector<FaceInfo> face_info;
		ultraface.detect(inmat, face_info);

		cv_image<bgr_pixel> cimg(img);
		matrix<rgb_pixel> matrix;
		assign_image(matrix, cimg);
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
		        double distance = 0.15f;
			cv::Scalar Bound_style = cv::Scalar(0, 0, 255);

		        for(auto& x:data_faces )
		        {
			    double prob = ultraface.SubVector(face_descriptors[0], x.second);
		            if(prob < distance)
		            {
		                distance = prob;
				Bound_style = cv::Scalar(0, 255, 0);
		                _ID = x.first;
		            }
		        }
			faces.clear();
			cv::rectangle(img, cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2), Bound_style, 1);
			cv::putText(img, _ID, cv::Point(face.x1, face.y2 - 10), cv::FONT_HERSHEY_DUPLEX, 1, Bound_style, 2, false);
		}
		face_info.clear();
		cv::imshow("CAM 1", img);
		if (cv::waitKey(1) == 27)
		{
			break;
		}
		//auto lock1 = std::unique_lock<std::mutex>(m1);
		//buffer_faces.push_back(faces);
		//lock1.unlock();
        	//std::this_thread::sleep_for (std::chrono::milliseconds(buffer_faces.size()));
	}
	//Destroy previously created window
	cv::destroyWindow("CAM 1");
}

void capture_and_detect_2(cv::VideoCapture& cap, UltraFace& ultraface, shape_predictor& sp, std::deque<std::vector<matrix<rgb_pixel>>>& buffer_faces)
{
	//Create window with unique title
	cv::namedWindow("CAM 2", cv::WINDOW_AUTOSIZE);
	auto m_StartTime = std::chrono::system_clock::now();
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

		std::vector<FaceInfo> face_info;
		ultraface.detect(inmat, face_info);

		cv_image<bgr_pixel> cimg(img);
		matrix<rgb_pixel> matrix;
		assign_image(matrix, cimg);
		std::vector<dlib::matrix<rgb_pixel>> faces;
		for (int i = 0; i < face_info.size(); i++)
		{
			auto face = face_info[i];
			rectangle rect(point(face.x1,face.y1), point(face.x2, face.y2));
			auto shape = sp(matrix,rect);

			dlib::matrix<rgb_pixel> face_chip;
			extract_image_chip(matrix, get_face_chip_details(shape,112,0.15), face_chip);
			faces.push_back(move(face_chip));
			cv::rectangle(img, cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2), cv::Scalar(0, 255, 0), 1);

		}
		face_info.clear();
		cv::imshow("CAM 2", img);
		if (cv::waitKey(1) == 27)
		{
			break;
		}
		//auto lock2 = std::unique_lock<std::mutex>(m2);
		//buffer_faces.push_back(faces);
		//lock2.unlock();
        	//std::this_thread::sleep_for (std::chrono::milliseconds(buffer_faces.size()));
	}
	//Destroy previously created window
	cv::destroyWindow("CAM 2");
}

void face_recognize_1(UltraFace& ultraface, std::deque<std::vector<matrix<rgb_pixel>>>& buffer_faces, std::map<std::string, std::vector<float>>& data_faces)
{
	while(true)
	{
		if(buffer_faces.size() <= 1)
		    continue;

		if(buffer_faces[0].size() == 0)
		{
		    auto lock = std::unique_lock<std::mutex>(m1);
		    buffer_faces.pop_front();
		    lock.unlock();        
		    continue;
		}

		for (size_t i = 0; i < buffer_faces[0].size(); ++i)
		{
		    ncnn::Mat tmp_ncnnmat = ncnn::Mat::from_pixels(toMat(buffer_faces[0][i]).data, ncnn::Mat::PIXEL_RGB, 112, 112);
		    std::vector<float> out;
		    ultraface.face_embedding(tmp_ncnnmat, out);
		    
		    for(auto&x: data_faces)
		    {
			float prob = calc_prob(x.second,out);
			if(prob > 0.6)
			{
			    cout<<"1_Name: "<<x.first<<endl;
			    cout<<"1_Prob: "<<prob<<endl;
			}
		    }
		}
		auto lock = std::unique_lock<std::mutex>(m1);
		buffer_faces.clear();
		lock.unlock();
	}
}

void face_recognize_2(UltraFace& ultraface, std::deque<std::vector<matrix<rgb_pixel>>>& buffer_faces, std::map<std::string, std::vector<float>>& data_faces)
{
	while(true)
	{
		if(buffer_faces.size() <= 1)
		    continue;

		if(buffer_faces[0].size() == 0)
		{
		    auto lock = std::unique_lock<std::mutex>(m2);
		    buffer_faces.pop_front();
		    lock.unlock();        
		    continue;
		}

		for (size_t i = 0; i < buffer_faces[0].size(); ++i)
		{
		    ncnn::Mat tmp_ncnnmat = ncnn::Mat::from_pixels(toMat(buffer_faces[0][i]).data, ncnn::Mat::PIXEL_RGB, 112, 112);
		    std::vector<float> out;
		    ultraface.face_embedding(tmp_ncnnmat, out);
		    
		    for(auto&x: data_faces)
		    {
			float prob = calc_prob(x.second,out);
			if(prob > 0.6)
			{
			    cout<<"2_Name: "<<x.first<<endl;
			    cout<<"2_Prob: "<<prob<<endl;
			}
		    }
		}
		auto lock = std::unique_lock<std::mutex>(m2);
		buffer_faces.pop_front();
		lock.unlock();

	}
}

int main(void)
{
	std::string gst_pipeline_1 = "v4l2src device=/dev/video1 ! image/jpeg, width=(int)1280, height=(int)720, framerate=30/1 ! jpegdec ! videoconvert ! appsink";
	std::string gst_pipeline_2 = "v4l2src device=/dev/video0 ! image/jpeg, width=(int)1280, height=(int)720, framerate=60/1 ! jpegdec ! videoconvert ! appsink";

	shape_predictor sp;
	deserialize("../Model/shape_predictor_68_face_landmarks.dat") >> sp;
	anet_type net;
	deserialize("../Model/dlib_face_recognition_resnet_model_v1.dat") >> net;

	//Load know faces
	std::map<std::string, dlib::matrix<float, 0, 1>> data_faces;    
	deserialize("../dataface/data_faces.dat") >> data_faces;

	//Template database
	UltraFace ultraface("../Model/RFB-320.bin", "../Model/RFB-320.param", 426, 240, 64, 0.82);

	//vector buffer
	std::deque<std::vector<matrix<rgb_pixel>>> buffer_faces_1;
	std::deque<std::vector<matrix<rgb_pixel>>> buffer_faces_2;

	cv::VideoCapture cap1(gst_pipeline_1, cv::CAP_GSTREAMER);
	cv::VideoCapture cap2(gst_pipeline_2, cv::CAP_GSTREAMER);

	if((!cap1.isOpened()) || (!cap2.isOpened()) ) {
		std::cout<<"Failed to open camera."<<std::endl;
		return (-1);
	}

	thread capture_and_detect_thread_1(capture_and_detect_1,   std::ref(cap1),
		                                                   std::ref(ultraface),
		                                                   std::ref(sp),
								   std::ref(net),
		                                                   std::ref(data_faces));

/*	thread capture_and_detect_thread_2(capture_and_detect_2,   std::ref(cap2),
		                                                   std::ref(ultraface),
		                                                   std::ref(sp),
		                                                   std::ref(buffer_faces_2));*/


/*	thread face_recognize_thread_1(face_recognize_1, std::ref(ultraface),
		                                   	 std::ref(buffer_faces_1),
		                                    	 std::ref(data_faces));
*/
/*	thread face_recognize_thread_2(face_recognize_2, std::ref(ultraface),
		                                   	 std::ref(buffer_faces_2),
		                                    	 std::ref(data_faces));*/

	capture_and_detect_thread_1.join();
	//capture_and_detect_thread_2.join();
//	face_recognize_thread_1.join();
	//face_recognize_thread_2.join();
	cap1.release();
	cap2.release();
	return 0;
}


















