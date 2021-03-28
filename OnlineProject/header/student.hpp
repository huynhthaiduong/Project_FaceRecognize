#include <dlib/dnn.h>
class student
{
    public:
    student(std::string stdname,std::string stdid,int stdcheck,dlib::matrix<float, 0, 1> sample);
    student();
    std::string id;
    std::string student_name;
    std::string student_id;
    std::string dat_path;
    dlib::matrix<float, 0, 1> student_features;
    int checked;
};

