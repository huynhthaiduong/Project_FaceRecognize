//#include "student.hpp"
#include <string>
#include <ctime>
#include <ratio>
#include <chrono>
class id_class
{
    public:
        id_class();
        //std::vector<student> lst_student;
        std::string date;
        int start_time;
        int end_time;
        std::string class_name;
        int room_id;
        void set_date(std::string date);
        void set_time(std::string temp_start_time,std::string temp_end_time);
        void set_class_name(std::string class_name);

};

