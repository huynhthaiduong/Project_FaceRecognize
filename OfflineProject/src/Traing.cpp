#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <experimental/filesystem>  
#include <map>
#include "../header/UltraFace.hpp"

using namespace dlib;
using namespace std;
namespace fs = std::experimental::filesystem;


std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "v4l2src device=/dev/video1 ! image/jpeg, width=(int)640, height=(int)360, framerate=30/1 ! jpegdec ! videoconvert ! appsink";//"nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           //std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           //"/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           //std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}
std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
);
float calc_prob(std::vector<float>& a, std::vector<float>& b)
{
    float prob = 0;
    for(int i =0; i < b.size();i++)
    {
        prob += a[i]*b[i]; 
    }
    return prob;
}

int main()
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
 
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) {
	std::cout<<"Failed to open camera."<<std::endl;
	return (-1);
    }

    cv::Mat img;

     frontal_face_detector detector = get_frontal_face_detector();
    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
    // And finally we load the DNN responsible for face recognition.

    std::map<std::string, std::vector<float>> data_faces;
    if (!fs::exists("data_faces.dat"))
    {
        serialize("data_faces.dat") << data_faces;
    }
    else
    {
        deserialize("data_faces.dat") >> data_faces;
    }


    image_window win;
    std::vector<matrix<rgb_pixel>> faces;
    
    string name;
    cout << "enter name: ";
    cin >> name;
    int cout_img = 0;
    std::vector<matrix<rgb_pixel>> array_face;

    UltraFace ultraface("RFB-320.bin", "RFB-320.param", 426, 240, 2, 0.82);

    while(true)
    {
    	if (!cap.read(img)) {
		std::cout<<"Capture read error"<<std::endl;
		break;
	    }
        std::chrono::time_point<std::chrono::system_clock> m_StartTime = std::chrono::system_clock::now();
        ncnn::Mat inmat = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);

        std::vector<FaceInfo> face_info;
        ultraface.detect(inmat, face_info);

        cv_image<bgr_pixel> cimg(img);

        matrix<rgb_pixel> matrix;
        assign_image(matrix, cimg);
        win.clear_overlay();
        win.set_image(matrix);
        faces.clear();

         for (int i = 0; i < face_info.size(); i++) {
            auto face = face_info[i];
            rectangle rect(point(face.x1,face.y1), point(face.x2, face.y2));
            image_window::overlay_rect orect(rect, rgb_pixel(255,0,0),"abc");
            auto shape = sp(matrix,rect);
            //double left_eye_ratio = get_blinking_ratio(l_eye_poits, shape);
            //double right_eye_ratio = get_blinking_ratio(r_eye_points, shape);
           
            dlib::matrix<rgb_pixel> face_chip;
            extract_image_chip(matrix, get_face_chip_details(shape,112,0.15), face_chip);
            faces.push_back(move(face_chip));
            win.add_overlay(orect);
        }
        
        
       

        if (faces.size() == 0)
        {
            cout << "No faces found in image!" << endl;
            continue;
        }

        cout_img++;
        cout << cout_img <<endl;
        if (cout_img == 100)
            break;
        array_face.push_back(faces[0]);
        
    }
    cout_img = 0;
    std::vector<float> mean_out;
    mean_out.assign(128,0);
    for (size_t i = 0; i < array_face.size(); ++i)
    {
        ncnn::Mat tmp_ncnnmat = ncnn::Mat::from_pixels(toMat(array_face[i]).data, ncnn::Mat::PIXEL_RGB, 112, 112);
        std::vector<float> out;
        ultraface.face_embedding(tmp_ncnnmat, out);

        for(int x = 0;x<mean_out.size();x++)
        {
            mean_out[x] +=  out[x];
        }
    }
    for(int x = 0;x<mean_out.size();x++)
    {
        mean_out[x] /= array_face.size();
    }
    data_faces.erase(name);
    data_faces.insert(std::pair<std::string, std::vector<float>>(name,mean_out));
    serialize("data_faces.dat") << data_faces;
    double juge = 0;
    while(true)
    {
    	if (!cap.read(img)) {
		std::cout<<"Capture read error"<<std::endl;
		break;
	    }
        std::chrono::time_point<std::chrono::system_clock> m_StartTime = std::chrono::system_clock::now();
        ncnn::Mat inmat = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);

        std::vector<FaceInfo> face_info;
        ultraface.detect(inmat, face_info);

        cv_image<bgr_pixel> cimg(img);

        matrix<rgb_pixel> matrix;
        assign_image(matrix, cimg);
        win.clear_overlay();
        win.set_image(matrix);
        faces.clear();

         for (int i = 0; i < face_info.size(); i++) {
            auto face = face_info[i];
            rectangle rect(point(face.x1,face.y1), point(face.x2, face.y2));
            image_window::overlay_rect orect(rect, rgb_pixel(255,0,0),"abc");
            auto shape = sp(matrix,rect);
            //double left_eye_ratio = get_blinking_ratio(l_eye_poits, shape);
            //double right_eye_ratio = get_blinking_ratio(r_eye_points, shape);
           
            dlib::matrix<rgb_pixel> face_chip;
            extract_image_chip(matrix, get_face_chip_details(shape,112,0.15), face_chip);
            faces.push_back(move(face_chip));
            win.add_overlay(orect);
        }
        
        
       

        if (faces.size() == 0)
        {
            cout << "No faces found in image!" << endl;
            continue;
        }
        for (size_t i = 0; i < faces.size(); ++i)
            {
                ncnn::Mat tmp_ncnnmat = ncnn::Mat::from_pixels(toMat(faces[i]).data, ncnn::Mat::PIXEL_RGB, 112, 112);
                std::vector<float> out;
                ultraface.face_embedding(tmp_ncnnmat, out);
                string _name = "";
                double distance = 0;
                for(auto& x:data_faces )
                {
                    float prob = calc_prob(x.second,out);
                    if(prob > distance)
                    {
                        distance = prob;
                        _name = x.first;
                    }
                }
                if(distance >= 0.7)
                {
                    cout << _name <<": "<< distance<< endl;
                    if(name ==_name)
                    {
                        juge += distance;
                        cout_img++;
                    }
                }
            }
        
        cout << cout_img <<endl;
        if (cout_img == 100)
            break;
    }
    cout<<"Avg Prob:"<<juge<<endl;
    cap.release();
    cv::destroyAllWindows() ;
    return 0;
}

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
)
{
    // All this function does is make 100 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently. They are also randomly
    // mirrored left to right.
    thread_local dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> crops; 
    for (int i = 0; i < 100; ++i)
        crops.push_back(jitter_image(img,rnd));

    return crops;
}
