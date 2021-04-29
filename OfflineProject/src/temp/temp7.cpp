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

void capture_and_detect_1(cv::VideoCapture& cap, shape_predictor& sp, anet_type& net, std::map<std::string, dlib::matrix<float, 0, 1>>& data_faces)
{
	//Template database
	UltraFace ultraface("../Model/RFB-320.bin", "../Model/RFB-320.param", 426, 240, 64, 0.82);
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

		cv::Mat image_clone = img.clone();
		ncnn::Mat inmat = ncnn::Mat::from_pixels(image_clone.data, ncnn::Mat::PIXEL_BGR2RGB, image_clone.cols, image_clone.rows);
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		ncnn::Mat score_blob32; 
		ncnn::Mat bbox_blob32;
		ncnn::Mat score_blob16;
		ncnn::Mat bbox_blob16;
		ultraface.detect(inmat, score_blob32,bbox_blob32,score_blob16,bbox_blob16);
		std::vector<FaceInfo> face_info;
		//ultraface.detect(inmat, face_info);
		const float prob_threshold = 0.8f;
		const float nms_threshold = 0.4f;
		std::vector<FaceInfo> faceproposals;
		//std::vector<FaceInfo> face_info;
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
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
			for(auto& x:data_faces )
			{
			    double prob = ultraface.SubVector(face_descriptors[0], x.second);
			    if(prob < distance)
			    {
			        distance = prob;
				cout<<"1_Name: "<<x.first<<endl;
			    }
			}
			faces.clear();
			cv::rectangle(img, cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2), cv::Scalar(0, 255, 0), 1);
		}
		face_info.clear();
		cv::imshow("CAM 1", img);
		if (cv::waitKey(1) == 27)
		{
			break;
		}
	}
	//Destroy previously created window
	cv::destroyWindow("CAM 1");
}

void capture_and_detect_2(cv::VideoCapture& cap, shape_predictor& sp, anet_type& net, std::map<std::string, dlib::matrix<float, 0, 1>>& data_faces)
{
	//Template database
	UltraFace ultraface("../Model/RFB-320.bin", "../Model/RFB-320.param", 426, 240, 64, 0.82);
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
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		ncnn::Mat score_blob32; 
		ncnn::Mat bbox_blob32;
		ncnn::Mat score_blob16;
		ncnn::Mat bbox_blob16;
		ultraface.detect(inmat, score_blob32,bbox_blob32,score_blob16,bbox_blob16);
		std::vector<FaceInfo> face_info;
		//ultraface.detect(inmat, face_info);
		const float prob_threshold = 0.8f;
		const float nms_threshold = 0.4f;
		std::vector<FaceInfo> faceproposals;
		//std::vector<FaceInfo> face_info;
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
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
			for(auto& x:data_faces )
			{
			    double prob = ultraface.SubVector(face_descriptors[0], x.second);
			    if(prob < distance)
			    {
			        distance = prob;
				cout<<"2_Name: "<<x.first<<endl;
			    }
			}
			faces.clear();
			cv::rectangle(img, cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2), cv::Scalar(0, 255, 0), 1);
		}
		face_info.clear();
		cv::imshow("CAM 2", img);
		if (cv::waitKey(1) == 27)
		{
			break;
		}
	}
	//Destroy previously created window
	cv::destroyWindow("CAM 2");
}

int main(void)
{
	std::string gst_pipeline_1 = "v4l2src device=/dev/video1 ! image/jpeg, width=(int)1280, height=(int)720, framerate=30/1 ! jpegdec ! videoconvert ! appsink";
	std::string gst_pipeline_2 = "v4l2src device=/dev/video0 ! image/jpeg, width=(int)1280, height=(int)720, framerate=60/1 ! jpegdec ! videoconvert ! appsink";

	cv::VideoCapture cap1(gst_pipeline_1, cv::CAP_GSTREAMER);
	cv::VideoCapture cap2(gst_pipeline_2, cv::CAP_GSTREAMER);

	if((!cap1.isOpened()) || (!cap2.isOpened()) ) {
		std::cout<<"Failed to open camera."<<std::endl;
		return (-1);
	}

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


	thread capture_and_detect_thread_1(capture_and_detect_1,   std::ref(cap1),
		                                                   std::ref(sp),
								   std::ref(net),
		                                                   std::ref(data_faces));

	thread capture_and_detect_thread_2(capture_and_detect_2,   std::ref(cap2),
		                                                   std::ref(sp),
								   std::ref(net),
		                                                   std::ref(data_faces));


	capture_and_detect_thread_1.join();
	capture_and_detect_thread_2.join();
	cap1.release();
	cap2.release();
	return 0;
}


















