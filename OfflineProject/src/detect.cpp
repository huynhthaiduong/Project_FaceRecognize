#include "../header/UltraFace.hpp"
#include <time.h>
#include <experimental/filesystem>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
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

#include <vector>
#include <map>
#include <string>
#include <math.h>

using namespace dlib;
using namespace std;

int main()
{
    cv::VideoCapture cap1("../videotest/tng.mp4");
    
    if (!cap1.isOpened())
    {
        std::cout << "Failed to open camera." << std::endl;
        return (-1);
    }
    
    UltraFace ultraface("../Model/RFB-320.bin", "../Model/RFB-320.param", 432, 240, 2, 0.7891011); 
    
    cv::Mat img;
 
    cv::namedWindow("Detect", cv::WINDOW_AUTOSIZE);
    
    auto m_StartTime = std::chrono::system_clock::now();
    auto m_EndTime = std::chrono::system_clock::now();
   
    while (true)
    {
	
        if (!cap1.read(img))
        {
            std::cout << "Capture read error" << std::endl;
            break;
        }


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
       
        for (int i = 0; i < face_info.size(); i++)
        {
            auto face = face_info[i];
            
            cv::rectangle(img, cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2), cv::Scalar(0,255,0), 1);
            
        }
        if (cv::waitKey(1) == 27)
        {
            cv::destroyAllWindows();
            break;
        }
        

        cv::imshow("Detect", img);
    }
    
    cap1.release();

    return 0;
}

