#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <dlib/image_transforms.h>
#include <dlib/image_processing.h>
#include <experimental/filesystem>
#include <map>
#include "../header/student.hpp"
#include "../header/TrainModel.hpp"
#include <string.h>
#include <fstream>
#include "gpu.h"
#include "net.h"

using namespace dlib;
using namespace std;
namespace fs = std::experimental::filesystem;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
                                                  alevel0<
                                                      alevel1<
                                                          alevel2<
                                                              alevel3<
                                                                  alevel4<
                                                                      max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>>;
/**/
/*
Chuong trinh nay lay mau cho cac sinh vien theo luong sau:
    Yeu cau nhap ten sinh vien
    Yeu cau nhap ma sinh vien
    Kiem tra su ton tai cua file "student_List.txt"
        Neu da ton tai:
            Lay het du lieu cua file "student_List.txt" ra 1 vector
            Kiem tra xem du lieu vua nhan da co trong "student_List.txt" hay chua:
                Neu co:
                    Cap nhat du lieu cua file "student_List.txt" tai vi tri du lieu cu
                Neu khong:
                    Them du lieu vua nhap vao file "student_List.txt" o vi tri cuoi cung
        Neu khong ton tai:
            Tao va them du lieu vua nhap vao "student_List.txt"
    Tao file dac trung dua vao duong dan sinh ra sau khi nhap thong tin

*/

int main()
{
    student temp_student;
    cout << "Enter Student Name: " << std::endl;
    getline(cin, temp_student.student_name);
    cout << "Enter Student ID: " << std::endl;
    getline(cin, temp_student.student_id);
    temp_student.dat_path = "../dat/" + temp_student.student_id + ".dat";
    std::string std_lst_path = "../list/studentList.txt";
    fstream f;
    int status = 0;
    std::vector<std::string> std_list;
    if (fs::exists(std_lst_path))
    {
        f.open(std_lst_path, ios::in);
        while (!f.eof())
        {
            std::string data;
            getline(f, data);
            std_list.push_back(data);
        }
        f.close();
        for (int i = 0; i < std_list.size(); i += 4)
        {
            if (temp_student.student_id == std_list[i])
            {
                std_list[i] = temp_student.student_id;
                std_list[i + 1] = temp_student.student_name;
                std_list[i + 2] = temp_student.dat_path;
                std_list.pop_back();
                status = 1;
            }
        }
        if (status == 0)
        {
            std_list.push_back(temp_student.student_id);
            std_list.push_back(temp_student.student_name);
            std_list.push_back(temp_student.dat_path);
        }
    }
    else
    {
        std_list.push_back(temp_student.student_id);
        std_list.push_back(temp_student.student_name);
        std_list.push_back(temp_student.dat_path);
    }
    f.open(std_lst_path, ios::out);
    for (int i = 0; i < std_list.size(); i++)
    {
        f << std_list[i] + "\n";
    }
    f.close();
    std::map<std::string, dlib::matrix<float,0,1>> data_faces;
    if (!fs::exists(temp_student.dat_path))
    {
        serialize(temp_student.dat_path) << data_faces;
    }
    else
    {
        deserialize(temp_student.dat_path) >> data_faces;
    }
    /*v4l2-ctl -d /dev/video0 --list-formats-ext*/
    //const std::string gst_pipeline = "v4l2src device=/dev/video0 ! video/x-raw, format=YUY2, width=640, height=480, framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink";
    const std::string gst_pipeline = "v4l2src device=/dev/video0 ! image/jpeg, width=(int)1280, height=(int)720, framerate=30/1 ! jpegdec ! videoconvert ! appsink";    
    cv::VideoCapture cap(gst_pipeline, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) {
	std::cout<<"Failed to open camera."<<std::endl;
	return (-1);
    }
    cv::Mat img;
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("../Model/shape_predictor_68_face_landmarks.dat") >> sp;
    anet_type net;
    deserialize("../Model/dlib_face_recognition_resnet_model_v1.dat") >> net;
    image_window win;
    std::vector<matrix<rgb_pixel>> faces;
    int cout_img = 0;
    int cout_color = 0;
    int cout_percent = 1;
    std::vector<matrix<rgb_pixel>> array_face;
    auto m_StartTime = std::chrono::system_clock::now();
    double FPS = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Capture " << FPS << " FPS " <<std::endl;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    TrainModel trainmodel("../Model/RFB-320.bin", "../Model/RFB-320.param", 426, 240, 128, 0.82);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    while(true)
    {
    	if (!cap.read(img)) {
		std::cout<<"Capture read error"<<std::endl;
		break;
	    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        double fps = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count();
	m_StartTime = std::chrono::system_clock::now();
	cv::circle(img, cv::Point(1280/2, 720/2), 200, cv::Scalar(0, 255, 0), 5);
	cv::putText(img,to_string(cout_percent) + "%", cv::Point((1280/2)-20, (720/2)-215), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, cout_color, 0), 2);
	cv::putText(img, to_string(static_cast<int>(1000/fps)) + " FPS", cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255), 1, false);
        cv::Mat image_clone = img.clone();
        ncnn::Mat inmat = ncnn::Mat::from_pixels(image_clone.data, ncnn::Mat::PIXEL_BGR2RGB, image_clone.cols, image_clone.rows);
/************************************************/
/*		      DETECT		        */
/************************************************/
        std::vector<FaceInfo> face_info;
        trainmodel.detect(inmat, face_info);

        cv_image<bgr_pixel> cimg(img);
        matrix<rgb_pixel> matrix;
	win.clear_overlay();
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
	    win.add_overlay(rect);
        }
        if (faces.size() == 0)
        {
            /*cout << "No faces found in image!" << endl;*/
	    cv::circle(img, cv::Point(1280/2, 720/2), 200, cv::Scalar(0, 0, 255), 5);
	    cv::putText(img,to_string(cout_percent) + "%", cv::Point((1280/2)-20, (720/2)-215), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255), 2);
            continue;
        }
        cout_img++;
	cout_color+=2;
	if ((cout_img % 2) ==0)
	    cout_percent++;
        cout << cout_img <<endl;
        if (cout_img == 200)
            break;
        array_face.push_back(faces[0]);
	if (cv::waitKey(1) >= 0)
	{
	    cv::destroyAllWindows();
	    break;
	}
	cv::imshow("Detect", img);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }
    cap.release();
    cv::destroyAllWindows() ;
    std::cout<<"Please wait a moments..."<<std::endl;
    data_faces.erase(temp_student.student_id);
    data_faces.insert(std::pair<std::string, dlib::matrix<float,0,1>>(temp_student.student_id,mean(mat(net(array_face)))));
    cout << data_faces[temp_student.student_id];
    serialize(temp_student.dat_path) << data_faces;
    ncnn::destroy_gpu_instance();
    return 0;
}
