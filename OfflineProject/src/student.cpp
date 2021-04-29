#include "../header/student.hpp"
#include "../header/TrainModel.hpp"
#include "../header/UltraFace.hpp"
#include <sstream>
#include <map>

struct Match_object
{
    std::string Name_detected;
    double Avg_value;
    double m_checkAvgValue_d;
    cv::Scalar Bound_style;
    int check_index;
};

student::student(){}

student::student(std::string stdname,std::string stdid,int stdcheck,dlib::matrix<float, 0, 1> sample)
{
    this->student_name=stdname;
    this->student_id=stdid;
    this->checked=stdcheck;
    this->student_features=sample;
}

void student::CreateListStu(student &temp_student, std::map<std::string, dlib::matrix<float,0,1>> &data_faces)
{
    fstream file;
    int status = 0;
    cout << "Enter Student Name: " << std::endl;
    getline(cin, temp_student.student_name);
    cout << "Enter Student ID: " << std::endl;
    getline(cin, temp_student.student_id);
    temp_student.dat_path = "../dat/" + temp_student.student_id + ".dat";
    std::string std_lst_path = "../list/studentList.csv";
    std::vector<std::string> std_list;
    std::string line;
    if (fs::exists(std_lst_path))
    {
        file.open(std_lst_path, ios::in);
        while (getline( file, line,'\n'))
        {
	  istringstream templine(line); 
	  string data;
	  while (getline( templine, data,','))
	  {
	    std_list.push_back(data.c_str());
	  }
        }
        file.close();
        for (int i = 0; i < std_list.size(); i += 2)
        {
            if (temp_student.student_id == std_list[i+1])
            {
                std_list[i] = temp_student.student_name;
                std_list[i + 1] = temp_student.student_id;
                std_list.pop_back();
                status = 1;
            }
        }
        if (status == 0)
        {
            std_list.push_back(temp_student.student_name);
            std_list.push_back(temp_student.student_id);
        }
    }
    else
    {
        std_list.push_back(temp_student.student_name);
        std_list.push_back(temp_student.student_id);
    }
    file.open(std_lst_path, ios::out);
    for (int i = 0; i < std_list.size(); i+=2)
    {
        file << std_list[i] + ",";
        file << std_list[i+1] + "\n";
    }
    file.close();
    if (!fs::exists(temp_student.dat_path))
    {
        serialize(temp_student.dat_path) << data_faces;
    }
    else
    {
        deserialize(temp_student.dat_path) >> data_faces;
    }
}

int student::ReadListStu(student &temp_student, std::vector<student> &temp_lst, std::map<std::string, std::vector<float>> &data_faces)
{
    //std::map<std::string, dlib::matrix<float, 0, 1>> data_faces;
    std::vector<std::string> std_list;
    fstream file;
    std::string line;
    
    if (fs::exists("../dataface/data_faces.dat"))
    {
        /*file.open("../list/studentList.csv", ios::in);
        while (getline( file, line,'\n'))
        {
	  istringstream templine(line); 
	  string data;
	  while (getline( templine, data,','))
	  {
	    std_list.push_back(data.c_str());
	  }
        }
        file.close();
        for (int i = 0; i < std_list.size(); i += 2)
        {
            temp_student.student_name = std_list[i];
            temp_student.student_id = std_list[i + 1];
            temp_student.dat_path = "../dat/" + temp_student.student_id + ".dat";
            temp_student.checked = 0;
            if (fs::exists(temp_student.dat_path))
            {
                deserialize(temp_student.dat_path) >> data_faces;
            }
            else
            {
                std::cout << "Data couldn't be found. Please check list and dat folder" << std::endl;
                return 0;
            }
            temp_student.student_features = data_faces[temp_student.student_id];
            temp_lst.push_back(temp_student);
        }*/

	deserialize("../dataface/data_faces.dat") >> data_faces;
    }
    else
    {
	    std::cout << "Data couldn't be found. Please check list and dat folder" << std::endl;
        return 0;
    }
    return 1;
}
