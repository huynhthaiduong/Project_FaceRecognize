#include "../header/student.hpp"
#include "../header/TrainModel.hpp"
#include <cmath>

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
    std::map<std::string, dlib::matrix<float,0,1>> data_faces;
    temp_student.CreateListStu(temp_student, data_faces);
    int imW = 640;
    int imH = 480;
    /*v4l2-ctl -d /dev/video0 --list-formats-ext*/
    //const std::string gst_pipeline = "v4l2src device=/dev/video0 ! video/x-raw, format=YUY2, width=640, height=480, framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink";
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
    deserialize("../Model/shape_predictor_5_face_landmarks.dat") >> sp;
    anet_type net;
    deserialize("../Model/dlib_face_recognition_resnet_model_v1.dat") >> net;
    image_window win;
    std::vector<matrix<rgb_pixel>> faces;
    int cout_img = 0;
    int cout_color = 0;
    int cout_percent = 1;
    int filenumber = 0;
    std::vector<matrix<rgb_pixel>> array_face;
    auto m_StartTime = std::chrono::system_clock::now();
    double FPS = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Capture " << FPS << " FPS " <<std::endl;
    TrainModel trainmodel("../Model/RFB-320.bin", "../Model/RFB-320.param", 432, 240, 128, 0.82);
    while(true)
    {
    	if (!cap.read(img)) {
		std::cout<<"Capture read error"<<std::endl;
		break;
	    }
        double fps = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_StartTime).count();
	    m_StartTime = std::chrono::system_clock::now();
	    cv::circle(img, cv::Point(imW/2, imH/2), 80, cv::Scalar(0, 255, 0), 5);
	    cv::putText(img,to_string(cout_percent) + "%", cv::Point((imW/2)-20, (imH/2)-215), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, cout_color, 0), 2);
	    cv::putText(img, to_string(static_cast<int>(1000/fps)) + " FPS", cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255), 1, false);
        cv::Mat image_clone = img.clone();
        ncnn::Mat inmat = ncnn::Mat::from_pixels(image_clone.data, ncnn::Mat::PIXEL_BGR2RGB, image_clone.cols, image_clone.rows);
        /************************************************/
        /*		      DETECT		                    */
        /************************************************/
        std::vector<TrainInfo> face_info;
        trainmodel.detect(inmat, face_info);

        cv_image<bgr_pixel> cimg(img);
        matrix<rgb_pixel> matrix;
	win.clear_overlay();
        assign_image(matrix, cimg);
        faces.clear();
        for (int i = 0; i < face_info.size(); i++)
        {
            auto face = face_info[i];
	    //if((face.x1>=imW/2-75)&&(face.x1<=imW/2)&&(face.x2>=imW/2)&&(face.x2<=imW/2+75)&&(face.y1>=imH/2-75)&&(face.y1<=imH/2)&&(face.y2>=imH/2)&&(face.y2<=imH/2+75))
            //{
//if((face.x1>=imW/2-75)&&(face.x1<=imW/2))
//{
                //std::cout<<"X"<<face.x2 - face.x1<<std::endl;
                rectangle rect(point(face.x1, face.y1), point(face.x2, face.y2));
                auto shape = sp(matrix, rect);
                dlib::matrix<rgb_pixel> face_chip;
                extract_image_chip(matrix, get_face_chip_details(shape, 150, 0.25), face_chip);
                faces.push_back(move(face_chip));

	        cv::Mat ROI(img, cv::Rect(cv::Point((imW/2)-75, (imH/2)-75), cv::Point((imW/2)+75, (imH/2)+75)));
	        cv::Mat croppedImage;
	        // Copy the data into new matrix
	        ROI.copyTo(croppedImage);
	        stringstream ssfn;
	        string filename = "/home/kyo/Desktop/KLTN/Project_FaceRecognize/OfflineProject/dataFacesImg/";
	        ssfn << filename.c_str() << temp_student.student_id << "_" << filenumber << ".jpg";
	        filename = ssfn.str();
	        imwrite(filename, croppedImage);
	        filenumber++;

	        win.add_overlay(rect);


            //}
        }
        if (faces.size() == 0)
        {
            /*cout << "No faces found in image!" << endl;*/
	        cv::circle(img, cv::Point(imW/2, imH/2), 80, cv::Scalar(0, 0, 255), 5);
	        cv::putText(img,to_string(cout_percent) + "%", cv::Point((imW/2)-20, (imH/2)-215), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255), 2);
            continue;
        }
        cout_img++;
	    cout_color+=2;
	    //if ((cout_img % 2) ==0)
	    cout_percent++;
        cout << cout_img <<endl;
        if (cout_img == 100)
            break;
        array_face.push_back(faces[0]);
        if (cv::waitKey(1) == 27)
        {
            cv::destroyAllWindows();
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
    data_faces.erase(temp_student.student_id);
    data_faces.insert(std::pair<std::string, dlib::matrix<float,0,1>>(temp_student.student_id,mean(mat(net(array_face)))));
    cout << data_faces[temp_student.student_id];
    serialize(temp_student.dat_path) << data_faces;
    ncnn::destroy_gpu_instance();
    return 0;
}
