#include "../header/student.hpp"
student::student()
{}
student::student(std::string stdname,std::string stdid,int stdcheck,dlib::matrix<float, 0, 1> sample)
{
    this->student_name=stdname;
    this->student_id=stdid;
    this->checked=stdcheck;
    this->student_features=sample;
}

