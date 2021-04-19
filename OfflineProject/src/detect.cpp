#include "../header/UltraFace.hpp"
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

using namespace std;
using namespace dlib;

struct Match_object
{
    std::string Name_detected;
    double Avg_value;
    double m_checkAvgValue_d = 0.15f;
    cv::Scalar Bound_style;
    int check_index;
};

class my_object : public multithreaded_object
{
public:
	std::string gst_pipeline_1;
	std::string gst_pipeline_2;
	//Read list Student
	student temp_student;
	std::vector<student> temp_lst;
	bool done = false;

    my_object()
    {
        // register which functions we want to run as threads.  We want one thread running
        // thread1() and two threads to run thread2().  So we will have a total of 2 threads
        // running.
	initStu(temp_student,temp_lst);
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
	
	void initStu(student& temp_student, std::vector<student>& temp_lst)
	{
		temp_student.ReadListStu(temp_student, temp_lst);
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
    
    void thread1()
    {
		std::string gst_pipeline = "v4l2src device=/dev/video1 ! image/jpeg, width=(int)1280, height=(int)720, framerate=30/1 ! jpegdec ! videoconvert ! appsink";
		cv::VideoCapture cap(gst_pipeline, cv::CAP_GSTREAMER);

		if( !cap.isOpened())
		{
			std::cout<<"Not good, open camera failed"<<std::endl;
			exit(-1);
		}

		std::cout<<"  ***************************************************************"<<std::endl;
    		std::cout<<" *                                                               *"<<std::endl;
    		std::cout<<"*                  Opened camera 1 successfully!                  *"<<std::endl;
    		std::cout<<" *                                                               *"<<std::endl;
    		std::cout<<"  ***************************************************************"<<std::endl;

		//Create window with unique title
		cv::namedWindow("Camera 1", cv::WINDOW_AUTOSIZE);

		shape_predictor sp;
		deserialize("../Model/shape_predictor_5_face_landmarks.dat") >> sp;
		anet_type net;
		deserialize("../Model/dlib_face_recognition_resnet_model_v1.dat") >> net;
		UltraFace ultraface("../Model/RFB-320.bin", "../Model/RFB-320.param", 432, 240, 64, 0.82);
		student temp_std;
		std::vector<matrix<rgb_pixel>> faces;
		cv::Mat img;
		cv::Mat outImg;
		auto m_StartTime = std::chrono::system_clock::now();
		double FPS = cap.get(cv::CAP_PROP_FPS);
		std::cout << "Capture  " << FPS << " FPS " <<std::endl;
		while (true)
		{cout << "starting thread 1" << endl;
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
			/************************************************/
			/*		      DETECT		        */
			/************************************************/
			std::vector<FaceInfo> face_info;
			ultraface.detect(inmat, face_info);

			cv_image<bgr_pixel> cimg(img);
			matrix<rgb_pixel> matrix;
			assign_image(matrix, cimg);
			faces.clear();
			for (int i = 0; i < face_info.size(); i++)
			{
				auto face = face_info[i];
				rectangle rect(point(face.x1, face.y1), point(face.x2, face.y2));
				auto shape = sp(matrix, rect);
				dlib::matrix<rgb_pixel> face_chip;
				extract_image_chip(matrix, get_face_chip_details(shape, 150, 0.25), face_chip);
				faces.push_back(move(face_chip));
				std::vector<dlib::matrix<float, 0, 1>> face_descriptors = net(faces);
				Match_object temp_obj;
				temp_obj.Name_detected = "Unknow";
				temp_obj.Avg_value = 0.15F;
				temp_obj.Bound_style = cv::Scalar(0, 0, 255);
				temp_obj.check_index = -1;

				for (int j = 0; j < temp_lst.size(); j++)
				{
					temp_obj.m_checkAvgValue_d = ultraface.SubVector(face_descriptors[0], temp_lst[j].student_features);
					if (temp_obj.m_checkAvgValue_d < temp_obj.Avg_value)
					{
				    		temp_obj.Avg_value = temp_obj.m_checkAvgValue_d;
				    		temp_obj.Name_detected = temp_lst[j].student_name;
				    		temp_obj.Bound_style = cv::Scalar(0, 255, 0);
				    		temp_obj.check_index = j;
					}
				}
				if (temp_obj.check_index != -1)
				{
				    temp_lst[temp_obj.check_index].checked = 1;
				}
				cv::rectangle(img, cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2), temp_obj.Bound_style, 1);
				cv::putText(img, temp_obj.Name_detected, cv::Point(face.x1, face.y2 - 10), cv::FONT_HERSHEY_DUPLEX, 1, temp_obj.Bound_style, 2, false);
				faces.clear();
				//cv::resize(img, outImg, cv::Size(720,405));
				cv::imshow("Camera 1", img);
				cout << "********************** FACE 1 *********************" << endl;
			}
			//cv::resize(img, outImg, cv::Size(720,405));
			cv::imshow("Camera 1", img);

			if (cv::waitKey(1) == 27)
			{
				cv::destroyAllWindows();
				break;
			}
		}
		checkstu(temp_lst);
		//Release VideoCapture object
		cap.release();
		//Destroy previously created window
		cv::destroyWindow("Camera 1");
		done = true;
    }

    void thread2()
    {
    		std::string gst_pipeline ="v4l2src device=/dev/video0 ! image/jpeg, width=(int)1280, height=(int)720, framerate=60/1 ! jpegdec ! videoconvert ! appsink";
    		cv::VideoCapture cap(gst_pipeline, cv::CAP_GSTREAMER);
	    
		if( !cap.isOpened())
		{
			std::cout<<"Not good, open camera failed"<<std::endl;
			exit(-1);
		}
		std::cout<<"  ***************************************************************"<<std::endl;
    		std::cout<<" *                                                               *"<<std::endl;
    		std::cout<<"*                  Opened camera 2 successfully!                  *"<<std::endl;
    		std::cout<<" *                                                               *"<<std::endl;
    		std::cout<<"  ***************************************************************"<<std::endl;

		//Create window with unique title
		cv::namedWindow("Camera 2", cv::WINDOW_AUTOSIZE);

		shape_predictor sp;
		deserialize("../Model/shape_predictor_5_face_landmarks.dat") >> sp;
		anet_type net;
		deserialize("../Model/dlib_face_recognition_resnet_model_v1.dat") >> net;
		UltraFace ultraface("../Model/RFB-320.bin", "../Model/RFB-320.param", 432, 240, 64, 0.82);
		student temp_std;
		std::vector<matrix<rgb_pixel>> faces;
		cv::Mat img;
		cv::Mat outImg;
		auto m_StartTime = std::chrono::system_clock::now();
		double FPS = cap.get(cv::CAP_PROP_FPS);
		std::cout << "Capture  " << FPS << " FPS " <<std::endl;
		while (true)
		{cout << "starting thread 2" << endl;
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
			/************************************************/
			/*		      DETECT		        */
			/************************************************/
			std::vector<FaceInfo> face_info;
			ultraface.detect(inmat, face_info);

			cv_image<bgr_pixel> cimg(img);
			matrix<rgb_pixel> matrix;
			assign_image(matrix, cimg);
			faces.clear();
			for (int i = 0; i < face_info.size(); i++)
			{
				auto face = face_info[i];
				rectangle rect(point(face.x1, face.y1), point(face.x2, face.y2));
				auto shape = sp(matrix, rect);
				dlib::matrix<rgb_pixel> face_chip;
				extract_image_chip(matrix, get_face_chip_details(shape, 150, 0.25), face_chip);
				faces.push_back(move(face_chip));
				std::vector<dlib::matrix<float, 0, 1>> face_descriptors = net(faces);
				Match_object temp_obj;
				temp_obj.Name_detected = "Unknow";
				temp_obj.Avg_value = 0.15F;
				temp_obj.Bound_style = cv::Scalar(0, 0, 255);
				temp_obj.check_index = -1;

				for (int j = 0; j < temp_lst.size(); j++)
				{
				temp_obj.m_checkAvgValue_d = ultraface.SubVector(face_descriptors[0], temp_lst[j].student_features);
				if (temp_obj.m_checkAvgValue_d < temp_obj.Avg_value)
				{
				    temp_obj.Avg_value = temp_obj.m_checkAvgValue_d;
				    temp_obj.Name_detected = temp_lst[j].student_name;
				    temp_obj.Bound_style = cv::Scalar(0, 255, 0);
				    temp_obj.check_index = j;
				}
				}
				if (temp_obj.check_index != -1)
				{
				    temp_lst[temp_obj.check_index].checked = 1;
				}
				cv::rectangle(img, cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2), temp_obj.Bound_style, 1);
				cv::putText(img, temp_obj.Name_detected, cv::Point(face.x1, face.y2 - 10), cv::FONT_HERSHEY_DUPLEX, 1, temp_obj.Bound_style, 2, false);
				faces.clear();
				//cv::resize(img, outImg, cv::Size(720,405));
				cv::imshow("Camera 2", img);
				cout << "********************** FACE 2 *********************" << endl;
			}
			//cv::resize(img, outImg, cv::Size(720,405));
			cv::imshow("Camera 2", img);

			if (cv::waitKey(1) == 27)
			{
				cv::destroyAllWindows();
				break;
			}

		}
		checkstu(temp_lst);
		//Release VideoCapture object
		cap.release();
		//Destroy previously created window
		cv::destroyWindow("Camera 2");
		done = true;
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

/**************************************************************************************************************************************************************************/
/*                                                                                   MAIN                                                                                 */
/**************************************************************************************************************************************************************************/
int main(void)
{
    my_object t;

    return 0;
}

