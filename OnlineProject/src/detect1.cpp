#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "mysql_connection.h"
#include <unistd.h>
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>
#include <iostream>
#include <vector>
#include "class.hpp"
#include "student.hpp"
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
#include "UltraFace/UltraFace.hpp"
#include <fstream>
#include <experimental/filesystem>
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
int get_time();
std::string get_date();
int main()
{
    sql::Driver *myDriver;
    sql::Connection *myConn;
    sql::Statement *myStmt;
    sql::ResultSet *myRes;
    std::vector<id_class> class_list_today;
    id_class temp_class;
    std::string temp, date, time;
    const std::string gst_pipeline = "v4l2src ! image/jpeg, width = 1280, height = 720, framerate=60/1 ! jpegdec ! videoconvert ! appsink";
    int update_cycle = 0;
    int detect = 0;
    std::vector<student> temp_lst;
    student temp_student;
    std::map<std::string, dlib::matrix<float, 0, 1>> data_faces;
    std::cout << get_time() << std::endl;
    cv::VideoCapture cap(gst_pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened())
    {

        std::cout << "Failed to open camera." << std::endl;

        return (-1);
    }
    while ((get_time() > 60000) and (get_time() < 180000))
    {
        date = get_date();
        myDriver = get_driver_instance();
        myConn = myDriver->connect("tcp://156.67.222.106:3306", "u477501821_duyle", "Duyle22697");
        myConn->setSchema("u477501821_diemdanhuit");
        myStmt = myConn->createStatement();
        myRes = myStmt->executeQuery("SELECT tkb_day.id_tkb, tkb_day.date, tkb_day.start ,tkb_day.end, class.id_class, class.name_class FROM tkb_day LEFT JOIN class ON tkb_day.id_class = class.id_class WHERE tkb_day.date ='" + date + "';");
        while (myRes->next())
        {
            temp_class.set_date(myRes->getString(2));
            temp_class.set_time(myRes->getString(3), myRes->getString(4));
            temp_class.set_class_name(myRes->getString(5));
            class_list_today.push_back(temp_class);
        }
        for (int i = 0; i < class_list_today.size(); i++)
        {
            int time = get_time();
            if ((time >= class_list_today[i].start_time) and (time <= class_list_today[i].end_time))
            {
                temp_class = class_list_today[i];
                detect = 1;
            }
        }
        delete myRes;
        myRes = myStmt->executeQuery("select tkb_class.id, sinhvien.mssv, sinhvien.name , tkb_class.status from tkb_class left join sinhvien on tkb_class.id_sv = sinhvien.id where tkb_class.id_class = '" + temp_class.class_name + "';");
        while (myRes->next())
        {
            temp_student.id = myRes->getString(1);
            temp_student.student_name = myRes->getString(3);
            temp_student.student_id = myRes->getString(2);
            temp_student.checked = stoi(myRes->getString(4));
            temp_lst.push_back(temp_student);
        }
        myConn->close();
        delete myConn;
        delete myStmt;
        delete myRes;

        for (int i = 0; i < temp_lst.size(); i++)
        {
            temp_lst[i].dat_path = "../dat/" + temp_lst[i].student_id + ".dat";
            temp_lst[i].checked = 0;
            if (fs::exists(temp_lst[i].dat_path))
            {
                deserialize(temp_lst[i].dat_path) >> data_faces;
            }
            else
            {
                std::cout << "Data couldn't be found. Please check list and dat folder" << std::endl;
                return 0;
            }
            temp_lst[i].student_features = data_faces[temp_lst[i].student_id];
        }
        shape_predictor sp;
        deserialize("/home/nhan/data/shape_predictor_68_face_landmarks.dat") >> sp;
        anet_type net;
        deserialize("/home/nhan/data/dlib_face_recognition_resnet_model_v1.dat") >> net;
        student temp_std;
        UltraFace ultraface("RFB-320.bin", "RFB-320.param", 426, 240, 2, 0.82);
        std::vector<matrix<rgb_pixel>> faces;
        cv::Mat img;
        cv::namedWindow("Detect", cv::WINDOW_AUTOSIZE);
        std::chrono::time_point<std::chrono::system_clock> m_StartTime = std::chrono::system_clock::now();
        while (detect == 1)
        {
            update_cycle++;
            if (!cap.read(img))
            {
                std::cout << "Capture read error" << std::endl;
                break;
            }
            cv::resize(img, img, cv::Size(800, 480));
            double fps = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count();
            int fps_int = static_cast<int>(1000 / fps);
            cv::putText(img, to_string(fps_int) + " FPS", cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255), 1, false);
            cv::imshow("Detect", img);
            if (cv::waitKey(1) >= 0)
            {
                cv::destroyAllWindows();
                break;
            }
            m_StartTime = std::chrono::system_clock::now();
            cv::Mat image_clone = img.clone();
            ncnn::Mat inmat = ncnn::Mat::from_pixels(image_clone.data, ncnn::Mat::PIXEL_BGR2RGB, image_clone.cols, image_clone.rows);
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
                temp_obj.Avg_value = 0.16f;
                temp_obj.Bound_style = cv::Scalar(0, 0, 255);
                temp_obj.check_index = -1;
                for (int j = 0; j < temp_lst.size(); j++)
                {
                    if ((length(face_descriptors[0] - temp_lst[j].student_features) * length(face_descriptors[0] - temp_lst[j].student_features)) < temp_obj.Avg_value)
                    {
                        temp_obj.Avg_value = length(face_descriptors[0] - temp_lst[j].student_features) * length(face_descriptors[0] - temp_lst[j].student_features);
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
                cv::imshow("Detect", img);
                if (cv::waitKey(10) >= 0)
                    break;
                faces.clear();
            }
            if (update_cycle > 100)
            {
                myDriver = get_driver_instance();
                myConn = myDriver->connect("tcp://156.67.222.106:3306", "u477501821_duyle", "Duyle22697");
                myConn->setSchema("u477501821_diemdanhuit");
                myStmt = myConn->createStatement();
                for (int i = 0; i < temp_lst.size(); i++)
                {
                    std::string test = "UPDATE tkb_class SET status='" + std::to_string(temp_lst[i].checked) + "' WHERE id='" + temp_lst[i].id + "';";
                    myStmt->executeUpdate(test);
                    std::cout << test << std::endl;
                }
                myConn->close();
                delete myConn;
                delete myStmt;
                update_cycle = 0;
            }
            if ((get_time() > temp_class.end_time) or (cv::waitKey(1) >= 0))
                detect = 0;
        }
        std::cout << "Waiting" << std::endl;
    }
    cap.release();
    return 0;
}
int get_time()
{
    std::time_t time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    tm *gmtm = localtime(&time_now);
    int out = (gmtm->tm_hour * 10000) + (gmtm->tm_min * 100) + gmtm->tm_sec;
    return out;
}
std::string get_date()
{
    std::time_t time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    tm *gmtm = localtime(&time_now);
    std::string out = std::to_string(gmtm->tm_year + 1900) + "-" + std::to_string(gmtm->tm_mon + 1) + "-" + std::to_string(gmtm->tm_mday);
    return out;
}
