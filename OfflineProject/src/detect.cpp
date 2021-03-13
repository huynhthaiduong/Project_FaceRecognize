#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <map>
#include <string>
#include <fstream>
#include <experimental/filesystem>
#include "../header/student.hpp"
#include "../header/UltraFace.hpp"
#include <math.h>

using namespace dlib;
using namespace std;
namespace fs = std::experimental::filesystem;
using std::string;

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

struct Match_object
{
    std::string Name_detected;
    double Avg_value;
    cv::Scalar Bound_style;
    int check_index;
};
int main()
{
    fstream f;
    student temp_student;
    std::vector<student> temp_lst;
    std::vector<std::string> std_list;
    std::map<std::string, dlib::matrix<float, 0, 1>> data_faces;
    if (fs::exists("../list/studentList.txt"))
    {
        f.open("../list/studentList.txt", ios::in);
        while (!f.eof())
        {
            std::string data;
            getline(f, data);
            std_list.push_back(data);
        }
        f.close();
        for (int i = 0; i < std_list.size(); i += 4)
        {
            temp_student.student_id = std_list[i];
            temp_student.student_name = std_list[i + 1];
            temp_student.dat_path = std_list[i + 2];
            temp_student.checked = 0;
            if (fs::exists(temp_student.dat_path))
            {
                deserialize(temp_student.dat_path) >> data_faces;
            }
            else
            {
                std::cout << "Data couldn't be found. Please check list and dat folder" << std::endl;
                return 0;
            }
            temp_student.student_features = data_faces[temp_student.student_id];
            temp_lst.push_back(temp_student);
        }
    }
    else
    {
	std::cout << "Data couldn't be found. Please check list and dat folder" << std::endl;
    }
    /*v4l2-ctl -d /dev/video0 --list-formats-ext*/
    // gst-device-monitor-1.0
    /*const std::string gst_pipeline = "v4l2src device=/dev/video0 ! video/x-raw, format=YUY2, width=640 height=480, framerate=20/1 ! videoconvert ! video/x-raw, format=BGR ! appsink";*/
    const std::string gst_pipeline = "v4l2src device=/dev/video0 ! image/jpeg, width=(int)1280, height=(int)720, framerate=30/1 ! jpegdec ! videoconvert ! appsink";
    cv::VideoCapture cap(gst_pipeline, cv::CAP_GSTREAMER);
    /*cv::VideoCapture cap(0);*/
    if (!cap.isOpened())
    {
        std::cout << "Failed to open camera." << std::endl;
        return (-1);
    }
    shape_predictor sp;
    deserialize("../Model/shape_predictor_68_face_landmarks.dat") >> sp;
    anet_type net;
    deserialize("../Model/dlib_face_recognition_resnet_model_v1.dat") >> net;
    student temp_std;
    UltraFace ultraface("../Model/RFB-320.bin", "../Model/RFB-320.param", 426, 240, 4, 0.82);
    std::vector<matrix<rgb_pixel>> faces;
    cv::Mat img;
    cv::namedWindow("Detect", cv::WINDOW_AUTOSIZE);
    std::chrono::time_point<std::chrono::system_clock> m_StartTime = std::chrono::system_clock::now();
    while (true)
    {
        if (!cap.read(img))
        {
            std::cout << "Capture read error" << std::endl;
            break;
        }
//      cv::resize(img, img, cv::Size(800, 600));
        double fps = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count();
        int fps_int = static_cast<int>(1000 / fps);
        cv::putText(img, to_string(fps_int) + " FPS", cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255), 1, false);
        cv::imshow("Detect", img);
        if (cv::waitKey(1) >= 0)
        {
            cv::destroyAllWindows();
            break;
        }
        m_StartTime = std::chrono::system_clock::now();
        cv::Mat image_clone = img.clone();
        ncnn::Mat inmat = ncnn::Mat::from_pixels(image_clone.data, ncnn::Mat::PIXEL_BGR2RGB, image_clone.cols, image_clone.rows);
/************************************************/
//	ncnn::create_gpu_instance();
/************************************************/
        std::vector<FaceInfo> face_info;
        ultraface.detect(inmat, face_info);
/************************************************/
//      ncnn::destroy_gpu_instance();
/************************************************/
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
            temp_obj.Avg_value = 0.16f;
            temp_obj.Bound_style = cv::Scalar(0, 0, 255);
            temp_obj.check_index = -1;
            for (int j = 0; j < temp_lst.size(); j++)
            {
	/************************************************/
//		ncnn::create_gpu_instance();
	/************************************************/
                if ((length(face_descriptors[0] - temp_lst[j].student_features) * length(face_descriptors[0] - temp_lst[j].student_features)) < temp_obj.Avg_value)
                {
                    temp_obj.Avg_value = length(face_descriptors[0] - temp_lst[j].student_features) * length(face_descriptors[0] - temp_lst[j].student_features);
                    temp_obj.Name_detected = temp_lst[j].student_name;
                    temp_obj.Bound_style = cv::Scalar(0, 255, 0);
                    temp_obj.check_index = j;
                }
	/************************************************/
//		ncnn::destroy_gpu_instance();
	/************************************************/
            }
            if (temp_obj.check_index != -1)
            {
                temp_lst[temp_obj.check_index].checked = 1;
            }
            cv::rectangle(img, cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2), temp_obj.Bound_style, 1);
            cv::putText(img, temp_obj.Name_detected, cv::Point(face.x1, face.y2 - 10), cv::FONT_HERSHEY_DUPLEX, 1, temp_obj.Bound_style, 2, false);
            cv::imshow("Detect", img);
            if (cv::waitKey(10) >= 0)
                break;
            faces.clear();
        }
        if (cv::waitKey(1) >= 0)
            break;
    }
    for (int i = 0; i < temp_lst.size(); i++)
    {
        std::cout << "Sinh vien: " << temp_lst[i].student_name << std::endl;
        std::cout << "      MSSV: " << temp_lst[i].student_id << std::endl;
        if (temp_lst[i].checked == 1)
        {
            std::cout << "      Co" << std::endl;
        }
        else
        {
            std::cout << "      Vang" << std::endl;
        }
    }
    cap.release();
    return 0;
}
