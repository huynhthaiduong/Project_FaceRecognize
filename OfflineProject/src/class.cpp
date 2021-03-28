#include "../header/class.hpp"
id_class::id_class()
{
}

void id_class::set_date(std::string date)
{
    this->date=date;
}
void id_class::set_time(std::string temp_start_time, std::string temp_end_time)
{
    std::string temp_time;
    std::string clear_item=":";
    for(int i=0;i<temp_start_time.size();i++)
    {
        if(temp_start_time[i]!=clear_item[0])
        {
            temp_time.push_back(temp_start_time[i]);
        }
        else
        {
            continue;
        }
    }
    this->start_time=stoi(temp_time);
    temp_time.clear();
    for(int i=0;i<temp_end_time.size();i++)
    {
        if(temp_end_time[i]!=clear_item[0])
        {
            temp_time.push_back(temp_end_time[i]);
        }
        else
        {
            continue;
        }
    }
    this->end_time=stoi(temp_time);
}
void id_class::set_class_name(std::string class_name)
{
    
    this->class_name=class_name;
}
