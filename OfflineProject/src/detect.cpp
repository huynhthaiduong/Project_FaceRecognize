#include "../header/UltraFace.hpp"
#include "../header/student.hpp"
#include <time.h>
#include <thread>
#include "libxl.h"

//Mutex for thread synchronization
static pthread_mutex_t foo_mutex = PTHREAD_MUTEX_INITIALIZER;

struct thread_data 
{
  std::string gst_pipeline1;
  int gst_pipeline;
  std::vector<student>* temp_lst;
  int  thread_id;
  string window_title; //Unique window title for each thread
};

struct Match_object
{
    std::string Name_detected;
    double Avg_value;
    double m_checkAvgValue_d = 0.15f;
    cv::Scalar Bound_style;
    int check_index;
};

/**************************************************************************************************************************************************************************/
/*                                                                        Capture And Detec MULTITHREAD                                                                   */
/**************************************************************************************************************************************************************************/
void *CaptureAndDetec(void *threadarg)
{
    struct thread_data *data;
    data = (struct thread_data *) threadarg;

    //Safely open video stream
    pthread_mutex_lock(&foo_mutex);
    cv::VideoCapture cap(data->gst_pipeline);//, cv::CAP_GSTREAMER);
    pthread_mutex_unlock(&foo_mutex);
    
    if( !cap.isOpened())
    {
      std::cout<<"Not good, open camera failed"<<std::endl;
      return 0;
    }
    std::cout<< "Opened camera successfully!"<<std::endl;

    //Create window with unique title
    cv::namedWindow(data->window_title, cv::WINDOW_AUTOSIZE);

    shape_predictor sp;
    deserialize("../Model/shape_predictor_68_face_landmarks.dat") >> sp;
    anet_type net;
    deserialize("../Model/dlib_face_recognition_resnet_model_v1.dat") >> net;
    student temp_std;
    UltraFace ultraface("../Model/RFB-320.bin", "../Model/RFB-320.param", 432, 240, 64, 0.82);
    std::vector<matrix<rgb_pixel>> faces;
    cv::Mat img;
    auto m_StartTime = std::chrono::system_clock::now();
    double FPS = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Capture  " << FPS << " FPS " <<std::endl;
    while (true)
    {
        if (!cap.read(img))
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
            for (int j = 0; j < (*(data->temp_lst)).size(); j++)
            {
		temp_obj.m_checkAvgValue_d = ultraface.SubVector(face_descriptors[0], (*(data->temp_lst))[j].student_features);
                if (temp_obj.m_checkAvgValue_d < temp_obj.Avg_value)
                {
                    temp_obj.Avg_value = temp_obj.m_checkAvgValue_d;
                    temp_obj.Name_detected = (*(data->temp_lst))[j].student_name;
                    temp_obj.Bound_style = cv::Scalar(0, 255, 0);
                    temp_obj.check_index = j;
                }
            }
            if (temp_obj.check_index != -1)
            {
                (*(data->temp_lst))[temp_obj.check_index].checked = 1;
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
        cv::imshow(data->window_title, img);
    }
    for (int i = 0; i < (*(data->temp_lst)).size(); i++)
    {
        std::cout << "Sinh vien: " << (*(data->temp_lst))[i].student_name << std::endl;
        std::cout << "      MSSV: " << (*(data->temp_lst))[i].student_id << std::endl;
        if ((*(data->temp_lst))[i].checked == 1)
        {
            std::cout << "      Co" << std::endl;
        }
        else
        {
            std::cout << "      Vang" << std::endl;
        }
    }
    //Release VideoCapture object
    cap.release();
    //Destroy previously created window
    cv::destroyWindow(data->window_title);

  //Exit thread
  pthread_exit(NULL);
}


/**************************************************************************************************************************************************************************/
/*                                                                                   MAIN                                                                                 */
/**************************************************************************************************************************************************************************/
int main(void)
{
    const int thread_count = 2;

    pthread_t threads[thread_count];
    struct thread_data td[thread_count];

    //Read list Student
    student temp_student;
    std::vector<student> m_temp_lst;
    temp_student.ReadListStu(temp_student, m_temp_lst);

    //Initialize thread data beforehand
    td[0].gst_pipeline = 1;
    //td[0].gst_pipeline = "v4l2src device=/dev/video1 ! image/jpeg, width=(int)640, height=(int)480, framerate=30/1 ! jpegdec ! videoconvert ! appsink";
    td[0].window_title = "CAM 1 ";
    td[0].temp_lst = &m_temp_lst;

    td[1].gst_pipeline = 0;
    //td[1].gst_pipeline = "v4l2src device=/dev/video0 ! video/x-raw, format=YUY2, width=640, height=480, framerate=25/1 ! videoconvert ! video/x-raw, format=BGR ! appsink";
    td[1].window_title = "CAM 2 ";
    td[1].temp_lst = &m_temp_lst;


    int rc = 0;
    for( int i = 0; i < thread_count; i++ ) 
    {
        cout <<"main() : creating thread, " << i << endl;
        td[i].thread_id = i;

        rc = pthread_create(&(threads[i]), NULL, CaptureAndDetec, (void *)& (td[i]));

        if (rc) 
        {
            cout << "Error:unable to create thread," << rc << endl;
            exit(-1);
        }
    }

    //Wait for the previously spawned threads to complete execution
    for( int i = 0; i < thread_count; i++ )
        pthread_join(threads[i], NULL);

    pthread_exit(NULL);

    return 0;
}

