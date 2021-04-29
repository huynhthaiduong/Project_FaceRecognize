#include "../header/student.hpp"
#include "../header/UltraFace.hpp"
#include <cmath>

int main()
{
    student temp_student;
    //std::map<std::string, dlib::matrix<float,0,1>> data_faces;
    //temp_student.CreateListStu(temp_student, data_faces);
    int imW = 640;
    int imH = 480;

    string ID;
    cout << "enter ID: ";
    cin >> ID;

    std::map<std::string, dlib::matrix<float, 0, 1>> data_faces;
    if (!fs::exists("../dataface/data_faces.dat"))
    {
        serialize("../dataface/data_faces.dat") << data_faces;
    }
    else
    {
        deserialize("../dataface/data_faces.dat") >> data_faces;
    }

    std::ostringstream gst;
    gst << "v4l2src device=/dev/video1 ! image/jpeg, width=(int)" << imW << ", height=(int)" << imH << ", framerate=30/1 ! jpegdec ! videoconvert ! appsink"; 
    const std::string gst_pipeline = gst.str(); 
    cv::VideoCapture cap(gst_pipeline, cv::CAP_GSTREAMER);
    if(!cap.isOpened()) {
	    std::cout<<"Failed to open camera."<<std::endl;
	    return (-1);
    }
    cv::Mat img;
    cv::Mat grayscale_img;
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("../Model/shape_predictor_68_face_landmarks.dat") >> sp;
    anet_type net;
    deserialize("../Model/dlib_face_recognition_resnet_model_v1.dat") >> net;
    std::vector<matrix<rgb_pixel>> faces;
    int cout_img = 0;
    int filenumber = 0;
    std::vector<matrix<rgb_pixel>> array_face;
    auto m_StartTime = std::chrono::system_clock::now();
    double FPS = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Capture " << FPS << " FPS " <<std::endl;
    UltraFace ultraface("../Model/RFB-320.bin", "../Model/RFB-320.param", 432, 240, 128, 0.82);
    while(true)
    {
    	if (!cap.read(img)) {
		std::cout<<"Capture read error"<<std::endl;
		break;
	}
	double fps = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count();
	m_StartTime = std::chrono::system_clock::now();
	cv::putText(img, to_string(static_cast<int>(1000/fps)) + " FPS", cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255), 1, false);
        ncnn::Mat inmat = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);
        /************************************************/
        /*		      DETECT		                    */
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
		cv::rectangle(img, cv::Point(face.x1, face.y1), cv::Point(face.x2, face.y2), cv::Scalar(0, 255, 0), 1);
	        //cv::Mat ROI(img, cv::Rect(cv::Point((imW/2)-75, (imH/2)-75), cv::Point((imW/2)+75, (imH/2)+75)));
	        //cv::Mat croppedImage;
	        // Copy the data into new matrix
	        //ROI.copyTo(croppedImage);
	        //stringstream ssfn;
	        //string filename = "/home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/dataFacesImg/";
	        //ssfn << filename.c_str() << temp_student.student_id << "_" << filenumber << ".jpg";
	        //filename = ssfn.str();
	        //imwrite(filename, croppedImage);
	        //filenumber++;
        }
	face_info.clear();
        cout_img++;
        cout << cout_img <<endl;
        if (cout_img == 100)
            break;
        array_face.push_back(faces[0]);
        if (cv::waitKey(1) == 27)
        {
            break;
        }
        cv::imshow("Detect", img);
    }
    cap.release();
    cv::destroyAllWindows() ;
    std::cout<<"  ***************************************************************"<<std::endl;
    std::cout<<" *                                                               *"<<std::endl;
    std::cout<<"*                          JUST A MOMENT.../                      *"<<std::endl;
    std::cout<<" *                                                               *"<<std::endl;
    std::cout<<"  ***************************************************************"<<std::endl;
    data_faces.erase(ID);
    data_faces.insert(std::pair<std::string, dlib::matrix<float,0,1>>(ID,mean(mat(net(array_face)))));
    cout << data_faces[ID];
    serialize("../dataface/data_faces.dat") << data_faces;
    return 0;
}













