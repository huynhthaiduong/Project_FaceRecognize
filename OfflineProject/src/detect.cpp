#include "../header/UltraFace.hpp"
#include "../header/student.hpp"
#include <time.h>

struct Match_object
{
    std::string Name_detected;
    double Avg_value;
    double m_checkAvgValue_d = 0.15f;
    cv::Scalar Bound_style;
    int check_index;
};
int main()
{
    student temp_student;
    std::vector<student> temp_lst;
    temp_student.ReadListStu(temp_student, temp_lst);
    /*v4l2-ctl -d /dev/video0 --list-formats-ext*/
    /*gst-device-monitor-1.0*/
    //const std::string gst_pipeline1 = "v4l2src device=/dev/video0 ! video/x-raw, format=YUY2, width=640, height=480, framerate=25/1 ! videoconvert ! video/x-raw, format=BGR ! appsink";
    //const std::string gst_pipeline1 = "v4l2src device=/dev/video0 ! image/jpeg, width=(int)1280, height=(int)720, framerate=30/1 ! jpegdec ! videoconvert ! appsink";
    //cv::VideoCapture cap0(gst_pipeline0, cv::CAP_GSTREAMER);
    //cv::VideoCapture cap1(gst_pipeline1, cv::CAP_GSTREAMER);
    //cv::VideoCapture cap1("../videotest/60fps.mov");
    cv::VideoCapture cap1(0);
    
    if (!cap1.isOpened())//||!cap0.isOpened())
    {
        std::cout << "Failed to open camera." << std::endl;
        return (-1);
    }
    shape_predictor sp;
    deserialize("../Model/shape_predictor_68_face_landmarks.dat") >> sp;
    anet_type net;
    deserialize("../Model/dlib_face_recognition_resnet_model_v1.dat") >> net;
    student temp_std;
    UltraFace ultraface("../Model/RFB-320.bin", "../Model/RFB-320.param", 432, 240, 128, 0.7891011);
    std::vector<matrix<rgb_pixel>> faces;
    cv::Mat img;
 //   cv::Mat frameL, frameR;
//    cap0>>frameL;
//    cap1>>frameR;
//    cv::hconcat(frameL, frameR, img);
    cv::namedWindow("Detect", cv::WINDOW_AUTOSIZE);
    cv::resizeWindow("Detect", 1280, 720);
    auto m_StartTime = std::chrono::system_clock::now();
    auto m_EndTime = std::chrono::system_clock::now();
    //clock_t start, end;
    //double FPS0 = cap0.get(cv::CAP_PROP_FPS);
    //std::cout << "Capture 0 " << FPS0 << " FPS " <<std::endl;
    double FPS1 = cap1.get(cv::CAP_PROP_FPS);
    std::cout << "Capture 1 " << FPS1 << " FPS " <<std::endl;
    while (true)
    {
	//start = clock();
        if (!cap1.read(img))//||!cap1.read(frameR))
        {
            std::cout << "Capture read error" << std::endl;
            break;
        }

        //cv::hconcat(frameL, frameR, img);

	

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
            temp_obj.Avg_value = 0.15f;
            temp_obj.Bound_style = cv::Scalar(0, 0, 255);
            temp_obj.check_index = -1;
            for (int j = 0; j < temp_lst.size(); j++)
            {
                //temp_obj.m_checkAvgValue_d = (length(face_descriptors[0] - temp_lst[j].student_features) * length(face_descriptors[0] - temp_lst[j].student_features));
		temp_obj.m_checkAvgValue_d = ultraface.SubVector(face_descriptors[0], temp_lst[j].student_features);
                if (temp_obj.m_checkAvgValue_d < temp_obj.Avg_value)
                {
                    //temp_obj.Avg_value = length(face_descriptors[0] - temp_lst[j].student_features) * length(face_descriptors[0] - temp_lst[j].student_features);
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
        }
        if (cv::waitKey(1) == 27)
        {
            cv::destroyAllWindows();
            break;
        }
        //end = clock();
	//double elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;

        cv::imshow("Detect", img);
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
    cap1.release();
//    cap0.release();
    return 0;
}

