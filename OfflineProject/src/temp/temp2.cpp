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
#include <deque>

#include <map>

using namespace std;
using namespace dlib;

std::mutex m1;
std::mutex m2;

struct Match_object
{
    std::string ID;
    std::string Name_detected;
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
	std::map<std::string, std::vector<float>> data_faces;
	bool done = false;
	image_window win;
	cv::VideoCapture cap1, cap2;
	my_object(cv::VideoCapture t_cap1, cv::VideoCapture t_cap2)
	{
		cap1 = t_cap1;
		cap2 = t_cap2;
		// register which functions we want to run as threads.  We want one thread running
		// thread1() and two threads to run thread2().  So we will have a total of 2 threads
		// running.
		initStu(temp_student,temp_lst,data_faces);
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

		cv::Mat outImg;
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
			//cv::resize(img, img, cv::Size(720,405));
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
			std::vector<dlib::matrix<rgb_pixel>> faces;
			for (int i = 0; i < face_info.size(); i++)
			{
				/*auto face = face_info[i];
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
				    		temp_obj.ID = temp_lst[j].student_name;
				    		temp_obj.Bound_style = cv::Scalar(0, 255, 0);
				    		temp_obj.check_index = j;
					}
				}
				if (temp_obj.check_index != -1)
				{
				    temp_lst[temp_obj.check_index].checked = 1;
				}*/

				temp_obj.Bound_style = cv::Scalar(0, 0, 255);
				temp_obj.ID = "Unknow";
				auto face = face_info[i];
				rectangle rect(point(face.x1,face.y1), point(face.x2, face.y2));
				//image_window::overlay_rect orect(rect, rgb_pixel(255,0,0),"abc");
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
					    //cout << _ID <<": "<< distance<< endl;
					    //if(ID ==_ID)
					    //{
						juge += distance;
				    		temp_obj.ID = _ID;
				    		temp_obj.Bound_style = cv::Scalar(0, 255, 0);
						//cout_img++;
					    //}
					}
		    		}
				//win.add_overlay(orect);
				faces.clear();
				cv::rectangle(img, cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2), temp_obj.Bound_style, 1);
				cv::putText(img, temp_obj.ID, cv::Point(face.x1, face.y2 - 10), cv::FONT_HERSHEY_DUPLEX, 1, temp_obj.Bound_style, 2, false);
				cv::imshow("Camera 1", img);
			}
			face_info.clear();
			cv::imshow("Camera 1", img);
			if (cv::waitKey(1) == 27)
			{
				//cv::destroyAllWindows();
				break;
			}
		}
		checkstu(temp_lst);
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
			/************************************************/
			/*		      DETECT		        */
			/************************************************/
			std::vector<FaceInfo> face_info;
			ultraface.detect(inmat, face_info);

			cv_image<bgr_pixel> cimg(img);
			matrix<rgb_pixel> matrix;
			assign_image(matrix, cimg);
			std::vector<dlib::matrix<rgb_pixel>> faces;
			for (int i = 0; i < face_info.size(); i++)
			{
				/*auto face = face_info[i];
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
				    		temp_obj.ID = temp_lst[j].student_name;
				    		temp_obj.Bound_style = cv::Scalar(0, 255, 0);
				    		temp_obj.check_index = j;
					}
				}
				if (temp_obj.check_index != -1)
				{
				    temp_lst[temp_obj.check_index].checked = 1;
				}*/

				temp_obj.Bound_style = cv::Scalar(0, 0, 255);
				temp_obj.ID = "Unknow";
				auto face = face_info[i];
				rectangle rect(point(face.x1,face.y1), point(face.x2, face.y2));
				//image_window::overlay_rect orect(rect, rgb_pixel(255,0,0),"abc");
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
					    //cout << _ID <<": "<< distance<< endl;
					    //if(ID ==_ID)
					    //{
						juge += distance;
				    		temp_obj.ID = _ID;
				    		temp_obj.Bound_style = cv::Scalar(0, 255, 0);
						//cout_img++;
					    //}
					}
		    		}
				//win.add_overlay(orect);
				faces.clear();
				cv::rectangle(img, cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2), temp_obj.Bound_style, 1);
				cv::putText(img, temp_obj.ID, cv::Point(face.x1, face.y2 - 10), cv::FONT_HERSHEY_DUPLEX, 1, temp_obj.Bound_style, 2, false);
				cv::imshow("Camera 2", img);
			}
			face_info.clear();
			cv::imshow("Camera 2", img);
			if (cv::waitKey(1) == 27)
			{
				//cv::destroyAllWindows();
				break;
			}
		}
		checkstu(temp_lst);
		//Release VideoCapture object
		cap2.release();
		//Destroy previously created window
		cv::destroyWindow("Camera 2");
		done = true;
	}

	/*bool check_internet()
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
	}*/

};


/*
void capture_and_detect_func_1(cv::VideoCapture& cap,UltraFace& ultraface, std::deque<cv::Mat>& buffer_img,
                                                                        std::deque<ncnn::Mat>& buffer_score_blob32,
                                                                        std::deque<ncnn::Mat>& buffer_bbox_blob32,
                                                                        std::deque<ncnn::Mat>& buffer_score_blob16,
                                                                        std::deque<ncnn::Mat>& buffer_bbox_blob16)
{
    while(true)
    {
        //cout<<"capture_and_detect_func "<< buffer_img.size()<<endl;
        cv::Mat img;
        if (!cap.read(img)) {
            std::cout<<"Capture read error"<<std::endl;
            break;
        }
        ncnn::Mat inmat = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);
        ncnn::Mat score_blob32; 
        ncnn::Mat bbox_blob32;
        ncnn::Mat score_blob16;
        ncnn::Mat bbox_blob16;

        ultraface.detect(inmat, score_blob32,bbox_blob32,score_blob16,bbox_blob16);

        auto lock = std::unique_lock<std::mutex>(m1);
        buffer_img.push_back(img);
        buffer_score_blob32.push_back(score_blob32);
        buffer_bbox_blob32.push_back(bbox_blob32);
        buffer_score_blob16.push_back(score_blob16);
        buffer_bbox_blob16.push_back(bbox_blob16);
        lock.unlock();
        
        std::this_thread::sleep_for (std::chrono::milliseconds(buffer_img.size()));
    }
}

void landmark_func_1(image_window& win,
                        shape_predictor& sp,
                        UltraFace& ultraface,
                        std::deque<cv::Mat>& buffer_img,
                        std::deque<ncnn::Mat>& buffer_score_blob32,
                        std::deque<ncnn::Mat>& buffer_bbox_blob32,
                        std::deque<ncnn::Mat>& buffer_score_blob16,
                        std::deque<ncnn::Mat>& buffer_bbox_blob16,
                        std::deque<std::vector<matrix<rgb_pixel>>>& buffer_faces)
{
    while(true)
    {
        //cout<<"landmark_ali_func"<<buffer_faces.size()<<endl;
        if(buffer_img.size() <= 1)
            continue;
        
        const float prob_threshold = 0.8f;
        const float nms_threshold = 0.4f;
        std::vector<FaceInfo> faceproposals;
        std::vector<FaceInfo> face_info;
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
        ultraface.generate_proposals(anchors, feat_stride, buffer_score_blob32[0], buffer_bbox_blob32[0], prob_threshold, faceobjects32);

        faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());

        //16
        
        feat_stride = 16;
        //ncnn::Mat ratios(1);
        //ratios[0] = 1.f;
        //ncnn::Mat scales(2);
        scales[0] = 8.f;
        scales[1] = 4.f;
        anchors = ultraface.generate_anchors(base_size, ratios, scales);
        std::vector<FaceInfo> faceobjects16;
        ultraface.generate_proposals(anchors, feat_stride, buffer_score_blob16[0], buffer_bbox_blob16[0], prob_threshold, faceobjects16);

        faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());
    
        ultraface.nms(faceproposals, face_info);

        cv_image<bgr_pixel> cimg(buffer_img[0]);
        matrix<rgb_pixel> matrix;
        assign_image(matrix, cimg);
        std::vector<dlib::matrix<rgb_pixel>> faces;

        win.clear_overlay();
        for (int i = 0; i < face_info.size(); i++) {
            auto face = face_info[i];
            rectangle rect(point(face.x1,face.y1), point(face.x2, face.y2));
            image_window::overlay_rect orect(rect, rgb_pixel(255,0,0));
            auto shape = sp(matrix,rect);        
            dlib::matrix<rgb_pixel> face_chip;
            extract_image_chip(matrix, get_face_chip_details(shape,112,0.25), face_chip);
            faces.push_back(move(face_chip));
            win.add_overlay(orect);
        }
        win.set_image(matrix);

        auto lock1 = std::unique_lock<std::mutex>(m2);
        buffer_faces.push_back(faces);
        lock1.unlock();

        auto lock = std::unique_lock<std::mutex>(m1);
        buffer_img.pop_front();
        buffer_score_blob32.pop_front();
        buffer_bbox_blob32.pop_front();
        buffer_score_blob16.pop_front();
        buffer_bbox_blob16.pop_front();
        lock.unlock();
        //std::this_thread::sleep_for (std::chrono::milliseconds(buffer_faces.size()));
    }
}

void capture_and_detect_func_2(cv::VideoCapture& cap,UltraFace& ultraface, std::deque<cv::Mat>& buffer_img,
                                                                        std::deque<ncnn::Mat>& buffer_score_blob32,
                                                                        std::deque<ncnn::Mat>& buffer_bbox_blob32,
                                                                        std::deque<ncnn::Mat>& buffer_score_blob16,
                                                                        std::deque<ncnn::Mat>& buffer_bbox_blob16)
{
    while(true)
    {
        //cout<<"capture_and_detect_func "<< buffer_img.size()<<endl;
        cv::Mat img;
        if (!cap.read(img)) {
            std::cout<<"Capture read error"<<std::endl;
            break;
        }
        ncnn::Mat inmat = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);
        ncnn::Mat score_blob32; 
        ncnn::Mat bbox_blob32;
        ncnn::Mat score_blob16;
        ncnn::Mat bbox_blob16;

        ultraface.detect(inmat, score_blob32,bbox_blob32,score_blob16,bbox_blob16);

        auto lock = std::unique_lock<std::mutex>(m1);
        buffer_img.push_back(img);
        buffer_score_blob32.push_back(score_blob32);
        buffer_bbox_blob32.push_back(bbox_blob32);
        buffer_score_blob16.push_back(score_blob16);
        buffer_bbox_blob16.push_back(bbox_blob16);
        lock.unlock();
        
        std::this_thread::sleep_for (std::chrono::milliseconds(buffer_img.size()));
    }
}

void landmark_func_2(image_window& win,
                        shape_predictor& sp,
                        UltraFace& ultraface,
                        std::deque<cv::Mat>& buffer_img,
                        std::deque<ncnn::Mat>& buffer_score_blob32,
                        std::deque<ncnn::Mat>& buffer_bbox_blob32,
                        std::deque<ncnn::Mat>& buffer_score_blob16,
                        std::deque<ncnn::Mat>& buffer_bbox_blob16,
                        std::deque<std::vector<matrix<rgb_pixel>>>& buffer_faces)
{
    while(true)
    {
        //cout<<"landmark_ali_func"<<buffer_faces.size()<<endl;
        if(buffer_img.size() <= 1)
            continue;
        
        const float prob_threshold = 0.8f;
        const float nms_threshold = 0.4f;
        std::vector<FaceInfo> faceproposals;
        std::vector<FaceInfo> face_info;
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
        ultraface.generate_proposals(anchors, feat_stride, buffer_score_blob32[0], buffer_bbox_blob32[0], prob_threshold, faceobjects32);

        faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());

        //16
        
        feat_stride = 16;
        //ncnn::Mat ratios(1);
        //ratios[0] = 1.f;
        //ncnn::Mat scales(2);
        scales[0] = 8.f;
        scales[1] = 4.f;
        anchors = ultraface.generate_anchors(base_size, ratios, scales);
        std::vector<FaceInfo> faceobjects16;
        ultraface.generate_proposals(anchors, feat_stride, buffer_score_blob16[0], buffer_bbox_blob16[0], prob_threshold, faceobjects16);

        faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());
    
        ultraface.nms(faceproposals, face_info);

        cv_image<bgr_pixel> cimg(buffer_img[0]);
        matrix<rgb_pixel> matrix;
        assign_image(matrix, cimg);
        std::vector<dlib::matrix<rgb_pixel>> faces;

        win.clear_overlay();
        for (int i = 0; i < face_info.size(); i++) {
            auto face = face_info[i];
            rectangle rect(point(face.x1,face.y1), point(face.x2, face.y2));
            image_window::overlay_rect orect(rect, rgb_pixel(255,0,0));
            auto shape = sp(matrix,rect);        
            dlib::matrix<rgb_pixel> face_chip;
            extract_image_chip(matrix, get_face_chip_details(shape,112,0.25), face_chip);
            faces.push_back(move(face_chip));
            win.add_overlay(orect);
        }
        win.set_image(matrix);

        auto lock1 = std::unique_lock<std::mutex>(m2);
        buffer_faces.push_back(faces);
        lock1.unlock();

        auto lock = std::unique_lock<std::mutex>(m1);
        buffer_img.pop_front();
        buffer_score_blob32.pop_front();
        buffer_bbox_blob32.pop_front();
        buffer_score_blob16.pop_front();
        buffer_bbox_blob16.pop_front();
        lock.unlock();
        //std::this_thread::sleep_for (std::chrono::milliseconds(buffer_faces.size()));
    }
}
*/

void capture_and_detect_func_1(cv::VideoCapture& cap,UltraFace& ultraface, std::deque<cv::Mat>& buffer_img, std::deque<std::vector<FaceInfo>>& buffer_faces)
{
    while(true)
    {
        //cout<<"capture_and_detect_func "<< buffer_img.size()<<endl;
        cv::Mat img;
        if (!cap.read(img)) {
            std::cout<<"Capture read error"<<std::endl;
            break;
        }

	ncnn::Mat inmat = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);
	std::vector<FaceInfo> face_info;

	ultraface.detect(inmat, face_info);

        auto lock = std::unique_lock<std::mutex>(m1);
        buffer_img.push_back(img);
	buffer_faces.push_back(face_info);
        lock.unlock();

        std::this_thread::sleep_for (std::chrono::milliseconds(buffer_img.size()));
    }
}
/*
void capture_and_detect_func_2(cv::VideoCapture& cap,UltraFace& ultraface, std::deque<cv::Mat>& buffer_img, std::deque<std::vector<matrix<rgb_pixel>>>& buffer_faces)
{
    while(true)
    {
        //cout<<"capture_and_detect_func "<< buffer_img.size()<<endl;
        cv::Mat img;
        if (!cap.read(img)) {
            std::cout<<"Capture read error"<<std::endl;
            break;
        }

	ncnn::Mat inmat = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);
	std::vector<FaceInfo> face_info;

	ultraface.detect(inmat, face_info);

        auto lock = std::unique_lock<std::mutex>(m2);
        buffer_img.push_back(img);
        lock.unlock();
        
        std::this_thread::sleep_for (std::chrono::milliseconds(buffer_img.size()));
    }
}*/
void face_recognize_1(cv::VideoCapture& cap, 
		      image_window& win,
                      shape_predictor& sp,
		      anet_type& net,
                      UltraFace& ultraface)
{
	std::cout<<"  ***************************************************************"<<std::endl;
	std::cout<<" *                                                               *"<<std::endl;
	std::cout<<"*                  Opened camera 1 successfully!                  *"<<std::endl;
	std::cout<<" *                                                               *"<<std::endl;
	std::cout<<"  ***************************************************************"<<std::endl;

	//Create window with unique title
	cv::namedWindow("Camera 1", cv::WINDOW_AUTOSIZE);
        student temp_student;
        std::vector<student> temp_lst;
        //temp_student.ReadListStu(temp_student, temp_lst);
	auto m_StartTime = std::chrono::system_clock::now();
	while (true)
	{
		cv::Mat img;
		if (!cap.read(img)) {
		    std::cout<<"Capture read error"<<std::endl;
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
		win.clear_overlay();
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
			//image_window::overlay_rect dets(rect, rgb_pixel(255,0,0));
			cv::rectangle(img, cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2), temp_obj.Bound_style, 1);
			cv::putText(img, temp_obj.Name_detected, cv::Point(face.x1, face.y2 - 10), cv::FONT_HERSHEY_DUPLEX, 1, temp_obj.Bound_style, 2, false);
			faces.clear();
			assign_image(matrix, cimg);
			//cv::resize(img, outImg, cv::Size(720,405));
			//cv::imshow("Camera 1", img);
			//win.add_overlay(dets);
			//cout << "********************** FACE 1 *********************" << endl;
		}
		face_info.clear();
		win.set_image(matrix);
		//auto lock = std::unique_lock<std::mutex>(m1);
		//buffer_img.pop_front();
		//buffer_faces.pop_front();
		//lock.unlock();
		//cv::resize(img, outImg, cv::Size(720,405));
		//cv::imshow("Camera 1", img);
		//win.add_overlay(img);
		if (cv::waitKey(1) == 27)
		{
			cv::destroyAllWindows();
			break;
		}
	}
	//checkstu(temp_lst);
	//Release VideoCapture object
	//cap.release();
	//Destroy previously created window
	//cv::destroyWindow("Camera 1");
	//done = true;*/
}

void face_recognize_2(cv::VideoCapture& cap, 
		      image_window& win,
                      shape_predictor& sp,
		      anet_type& net,
                      UltraFace& ultraface)
{
	std::cout<<"  ***************************************************************"<<std::endl;
	std::cout<<" *                                                               *"<<std::endl;
	std::cout<<"*                  Opened camera 2 successfully!                  *"<<std::endl;
	std::cout<<" *                                                               *"<<std::endl;
	std::cout<<"  ***************************************************************"<<std::endl;

	//Create window with unique title
	cv::namedWindow("Camera 2", cv::WINDOW_AUTOSIZE);
        student temp_student;
        std::vector<student> temp_lst;
        //temp_student.ReadListStu(temp_student, temp_lst);
	auto m_StartTime = std::chrono::system_clock::now();
	while (true)
	{
		cv::Mat img;
		if (!cap.read(img)) {
		    std::cout<<"Capture read error"<<std::endl;
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
		win.clear_overlay();
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
			//image_window::overlay_rect dets(rect, rgb_pixel(255,0,0));
			cv::rectangle(img, cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2), temp_obj.Bound_style, 1);
			cv::putText(img, temp_obj.Name_detected, cv::Point(face.x1, face.y2 - 10), cv::FONT_HERSHEY_DUPLEX, 1, temp_obj.Bound_style, 2, false);
			faces.clear();
			assign_image(matrix, cimg);
			//cv::resize(img, outImg, cv::Size(720,405));
			//cv::imshow("Camera 1", img);
			//win.add_overlay(dets);
			//cout << "********************** FACE 1 *********************" << endl;
		}
		face_info.clear();
		win.set_image(matrix);
		//auto lock = std::unique_lock<std::mutex>(m1);
		//buffer_img.pop_front();
		//buffer_faces.pop_front();
		//lock.unlock();
		//cv::resize(img, outImg, cv::Size(720,405));
		//cv::imshow("Camera 2", img);
		//win.add_overlay(img);
		if (cv::waitKey(1) == 27)
		{
			cv::destroyAllWindows();
			break;
		}
	}
	//checkstu(temp_lst);
	//Release VideoCapture object
	//cap.release();
	//Destroy previously created window
	//cv::destroyWindow("Camera 1");
	//done = true;*/
}

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
/*    int capture_width = 640 ;
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
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";*/
    std::string gst_pipeline_1 = "v4l2src device=/dev/video1 ! image/jpeg, width=(int)640, height=(int)360, framerate=30/1 ! jpegdec ! videoconvert ! appsink";
    std::string gst_pipeline_2 = "v4l2src device=/dev/video0 ! image/jpeg, width=(int)1280, height=(int)720, framerate=60/1 ! jpegdec ! videoconvert ! appsink";
 
    cv::VideoCapture cap1(gst_pipeline_1, cv::CAP_GSTREAMER);
    cv::VideoCapture cap2(gst_pipeline_2, cv::CAP_GSTREAMER);
    if((!cap1.isOpened()) || (!cap2.isOpened()) ) {
	std::cout<<"Failed to open camera."<<std::endl;
	return (-1);
    }
    
    //Load know faces
    //std::map<std::string, std::vector<float>> data_faces;    
    //deserialize("data_faces.dat") >> data_faces;
    //Read list Student
    //student temp_student;
    //std::vector<student> temp_lst;
    //temp_student.ReadListStu(temp_student, temp_lst);
    //image_window win1;
    //image_window win2;
    
    //Template database

    UltraFace ultraface("../Model/RFB-320.bin", "../Model/RFB-320.param", 432, 240, 64, 0.82);


    my_object test(cap1,cap2);

    //vector buffer
    std::deque<cv::Mat> buffer_img_1;
    std::deque<std::vector<FaceInfo>> buffer_faces_1;
    std::deque<ncnn::Mat> buffer_score_blob16_1;
    std::deque<ncnn::Mat> buffer_bbox_blob16_1;
    std::deque<ncnn::Mat> buffer_score_blob32_1;
    std::deque<ncnn::Mat> buffer_bbox_blob32_1;

    std::deque<cv::Mat> buffer_img_2;
    std::deque<std::vector<matrix<rgb_pixel>>> buffer_faces_2;
    std::deque<ncnn::Mat> buffer_score_blob16_2;
    std::deque<ncnn::Mat> buffer_bbox_blob16_2;
    std::deque<ncnn::Mat> buffer_score_blob32_2;
    std::deque<ncnn::Mat> buffer_bbox_blob32_2;

/*
    thread capture_and_detect_thread_1(capture_and_detect_func_1,   std::ref(cap1),
                                                        	    std::ref(ultraface),
                                                                    std::ref(buffer_img_1),
                                                                    std::ref(buffer_score_blob32_1),
                                                                    std::ref(buffer_bbox_blob32_1),
                                                                    std::ref(buffer_score_blob16_1),
                                                                    std::ref(buffer_bbox_blob16_1));

    thread landmark_thread_1(landmark_func_1,   std::ref(win1),
                                                std::ref(sp),
                                                std::ref(ultraface),
                                                std::ref(buffer_img_1),
                                                std::ref(buffer_score_blob32_1),
                                                std::ref(buffer_bbox_blob32_1),
                                                std::ref(buffer_score_blob16_1),
                                                std::ref(buffer_bbox_blob16_1),
                                                std::ref(buffer_faces_1));

    thread capture_and_detect_thread_2(capture_and_detect_func_2,   std::ref(cap2),
                                                        	    std::ref(ultraface),
                                                                    std::ref(buffer_img_2),
                                                                    std::ref(buffer_score_blob32_2),
                                                                    std::ref(buffer_bbox_blob32_2),
                                                                    std::ref(buffer_score_blob16_2),
                                                                    std::ref(buffer_bbox_blob16_2));

    thread landmark_thread_2(landmark_func_2,   std::ref(win2),
                                                std::ref(sp),
                                                std::ref(ultraface),
                                                std::ref(buffer_img_2),
                                                std::ref(buffer_score_blob32_2),
                                                std::ref(buffer_bbox_blob32_2),
                                                std::ref(buffer_score_blob16_2),
                                                std::ref(buffer_bbox_blob16_2),
                                                std::ref(buffer_faces_2));

    capture_and_detect_thread_1.join();
    landmark_thread_1.join();
    capture_and_detect_thread_2.join();
    landmark_thread_2.join();
*/

/*    thread capture_and_detect_thread_1(capture_and_detect_func_1,   std::ref(cap1),
                                                        	    std::ref(ultraface),
                                                                    std::ref(buffer_img_1),
								    std::ref(buffer_faces_1));
*/
/*
    thread face_recognize_thread_1(face_recognize_1,   std::ref(cap1), 
						       std::ref(win1),
                                                       std::ref(sp),
                                                       std::ref(net),
                                                       std::ref(ultraface));

    thread face_recognize_thread_2(face_recognize_2,   std::ref(cap2), 
						       std::ref(win2),
                                                       std::ref(sp),
                                                       std::ref(net),
                                                       std::ref(ultraface));
//    capture_and_detect_thread_1.join();
    face_recognize_thread_1.join();
    face_recognize_thread_2.join();
    cap1.release();
    cap2.release();
*/
    return 0;
}

























