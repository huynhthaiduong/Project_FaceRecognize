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
#include "../header/student.hpp"

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
    /* face landmark */
        shape_predictor sp;
		deserialize("../Model/shape_predictor_5_face_landmarks.dat") >> sp;
		anet_type net;
		deserialize("../Model/dlib_face_recognition_resnet_model_v1.dat") >> net;
    std::vector<matrix<rgb_pixel>> faces;
		cv::Mat img;
		cv::Mat outImg;
    /*****************/
    // CascadeClassifier facecascade;
    // facecascade.load("../Model/shape_predictor_68_face_landmarks.dat")
    UltraFace ultraface("../Model/RFB-320.bin", "../Model/RFB-320.param", 432, 240, 2, 0.7891011); 
    // UltraFace ultraface("../Model/shape_predictor_68_face_landmarks.dat"); 
    
    // cv::Mat img;
 
    // cv::namedWindow("Detect", cv::WINDOW_AUTOSIZE);
    
    // auto m_StartTime = std::chrono::system_clock::now();
    // auto m_EndTime = std::chrono::system_clock::now();
   
    while (true)
    {
	
        if (!cap1.read(img))
        {
            std::cout << "Capture read error" << std::endl;
            break;
        }


    //     double fps = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count();
	// m_StartTime = std::chrono::system_clock::now();
	// cv::putText(img, to_string(static_cast<int>(1000/fps)) + " FPS", cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255), 1, false);

        cv::Mat image_clone = img.clone();
        ncnn::Mat inmat = ncnn::Mat::from_pixels(image_clone.data, ncnn::Mat::PIXEL_BGR2RGB, image_clone.cols, image_clone.rows);
        /************************************************/
        /*		      DETECT		        */
        /************************************************/

        std::vector<FaceInfo> face_info;
        ultraface.detect(inmat, face_info); //face detection provide cordinate of the face
    
        // cv_image<bgr_pixel> cimg(img);
        // matrix<rgb_pixel> matrix;
        // assign_image(matrix, cimg);
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
            // Match_object temp_obj;
            // temp_obj.Name_detected = "Unknow";
            // temp_obj.Avg_value = 0.15F;
            // temp_obj.Bound_style = cv::Scalar(0, 0, 255);
            // temp_obj.check_index = -1;
            /* drawing bounding box */
            cv::rectangle(img, cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2), cv::Scalar(0,255,0), 1);
            for (int j = 0; j < face_descriptors.size(); j++)
            {
            printf("%d \n",face_descriptors[j]);
            }
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

