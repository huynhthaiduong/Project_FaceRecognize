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
#include <deque>

#include <math.h>
using namespace dlib;
using namespace std;

std::mutex m1;
std::mutex m2;
//std::chrono::time_point<std::chrono::system_clock> m_StartTime = std::chrono::system_clock::now();
        
std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "v4l2src device=/dev/video1 ! image/jpeg, width=(int)640, height=(int)360, framerate=30/1 ! jpegdec ! videoconvert ! appsink";//"nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           //std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           //"/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           //std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true";
}


void updata_mysql(sql::PreparedStatement *ps, const string name, matrix<rgb_pixel>& img)
{
    cv::Mat tmp = toMat(img);

    std::vector<unsigned char> buff(22500);
    cv::imencode(".png", tmp, buff,{cv::IMWRITE_PNG_STRATEGY_DEFAULT,1});
    unsigned char* buffimg = &buff[0];

    std::string value(reinterpret_cast<char*>(buffimg),22500);
    std::istringstream tmp_blob(value);
    ps->setBlob(2,&tmp_blob);
    ps->setString(1,name);
    ps->executeUpdate();     
}

bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2){
    // treat two empty mat as identical as well
    if (mat1.empty() && mat2.empty()) {
        return true;
    }
    // if dimensionality of two mat is not identical, these two mat is not identical
    if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims) {
        return false;
    }
    cv::Mat diff;
    cv::compare(mat1, mat2, diff, cv::CMP_NE);
    int nz = cv::countNonZero(diff);
    return nz==0;
}


double get_blinking_ratio(int eye_points[],dlib::full_object_detection facial_landmarks)
{
    cv::Point left_point (facial_landmarks.part(eye_points[0]).x(), facial_landmarks.part(eye_points[0]).y());
    cv::Point right_point (facial_landmarks.part(eye_points[3]).x(), facial_landmarks.part(eye_points[3]).y());
    cv::Point top1 (facial_landmarks.part(eye_points[1]).x(),facial_landmarks.part(eye_points[1]).y());
    cv::Point top2 (facial_landmarks.part(eye_points[2]).x(),facial_landmarks.part(eye_points[2]).y());
    cv::Point bot1 (facial_landmarks.part(eye_points[5]).x(),facial_landmarks.part(eye_points[5]).y());
    cv::Point bot2 (facial_landmarks.part(eye_points[4]).x(),facial_landmarks.part(eye_points[4]).y());
    cv::Point center_top = (top1 + top2)/2;
    cv::Point center_bottom = (bot1 + bot2)/2;

    //hor_line = cv2.line(image, left_point, right_point, (0, 255, 0), 2)
    //ver_line = cv2.line(image, center_top, center_bottom, (0, 255, 0), 2)

    double hor_line_lenght = hypot((left_point.x - right_point.x), (left_point.y - right_point.y));
    double ver_line_lenght = hypot((center_top.x - center_bottom.x), (center_top.y - center_bottom.y));

    
    return hor_line_lenght / ver_line_lenght;
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

void capture_and_detect_func(cv::VideoCapture& cap,UltraFace& ultraface, std::deque<cv::Mat>& buffer_img,
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

void landmark_ali_func(image_window& win,
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
            //double left_eye_ratio = get_blinking_ratio(l_eye_poits, shape);
            //double right_eye_ratio = get_blinking_ratio(r_eye_points, shape);
           
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
        std::this_thread::sleep_for (std::chrono::milliseconds(buffer_faces.size()));
    }
}

void face_embedding_func(   UltraFace& ultraface,
                            std::deque<std::vector<matrix<rgb_pixel>>>& buffer_faces,
                            std::map<std::string, std::vector<float>>& data_faces)
{
    while(true)
    {
        if(buffer_faces.size() <= 1)
            continue;
        
        if(buffer_faces[0].size() == 0)
        {
            auto lock = std::unique_lock<std::mutex>(m2);
            buffer_faces.pop_front();
            lock.unlock();        
            continue;
        }

        for (size_t i = 0; i < buffer_faces[0].size(); ++i)
        {
            ncnn::Mat tmp_ncnnmat = ncnn::Mat::from_pixels(toMat(buffer_faces[0][i]).data, ncnn::Mat::PIXEL_RGB, 112, 112);
            std::vector<float> out;
            ultraface.face_embedding(tmp_ncnnmat, out);
            
            for(auto&x: data_faces)
            {
                float prob = calc_prob(x.second,out);
                if(prob > 0.6)
                {
                    cout<<"Name: "<<x.first<<endl;
                    cout<<"Prob: "<<prob<<endl;
                }
            }
        }
        auto lock = std::unique_lock<std::mutex>(m2);
        buffer_faces.pop_front();
        lock.unlock();
        
    }
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

    shape_predictor sp;
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
    int r_eye_points[] = {42, 43, 44, 45, 46, 47};
    int l_eye_poits[] = {36, 37, 38, 39, 40, 41};
    // And finally we load the DNN responsible for face recognition.
    

    
    //Load know faces
    std::map<std::string, std::vector<float>> data_faces;    
    deserialize("data_faces.dat") >> data_faces;
    
    //Template database

    UltraFace ultraface("RFB-320.bin", "RFB-320.param", 426, 240, 2, 0.82);
    image_window win;
    


    
    //vector buffer
    std::deque<cv::Mat> buffer_img;
    std::deque<ncnn::Mat> buffer_score_blob16;
    std::deque<ncnn::Mat> buffer_bbox_blob16;
    std::deque<ncnn::Mat> buffer_score_blob32;
    std::deque<ncnn::Mat> buffer_bbox_blob32;
    std::deque<std::vector<matrix<rgb_pixel>>> buffer_faces;

    thread capture_and_detect_thread(capture_and_detect_func,   std::ref(cap),
                                                                std::ref(ultraface),
                                                                std::ref(buffer_img),
                                                                std::ref(buffer_score_blob32),
                                                                std::ref(buffer_bbox_blob32),
                                                                std::ref(buffer_score_blob16),
                                                                std::ref(buffer_bbox_blob16));
    	
    thread landmark_ali_thread(landmark_ali_func,   std::ref(win),
                                                    std::ref(sp),
                                                    std::ref(ultraface),
                                                    std::ref(buffer_img),
                                                    std::ref(buffer_score_blob32),
                                                    std::ref(buffer_bbox_blob32),
                                                    std::ref(buffer_score_blob16),
                                                    std::ref(buffer_bbox_blob16),
                                                    std::ref(buffer_faces));

    thread face_embedding_thread(face_embedding_func,std::ref(ultraface),
                                                    std::ref(buffer_faces),
                                                    std::ref(data_faces));
    
    capture_and_detect_thread.join();
    landmark_ali_thread.join();
    face_embedding_thread.join();
    cap.release();
    return 0;
}
