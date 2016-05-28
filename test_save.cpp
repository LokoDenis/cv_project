#include <cmath>
#include <iostream>
#include <list>
#include <map>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <fstream>

using namespace cv;
using namespace cv::xfeatures2d;


//! Write cv::Mat as binary
/*!
\param[out] ofs output file stream
\param[in] out_mat mat to save
*/
bool write_binary_descriptor(std::ofstream& ofs, const Mat& output)
{
    if(!ofs.is_open()){
        return false;
    }

    int type = out_mat.type();
    ofs.write((const char*)(&output.flags), sizeof(int));
    ofs.write((const char*)(&output.rows), sizeof(int));
    ofs.write((const char*)(&output.cols), sizeof(int));
    ofs.write((const char*)(&type), sizeof(int));
    ofs.write((uchar*)(out_mat.data), out_mat.elemSize() * out_mat.total());

    return true;
}


//! Save cv::Mat as binary
/*!
\param[in] filename filaname to save
\param[in] output cvmat to save
*/
bool SaveMatBinary(const std::string& filename, const cv::Mat& output){
    std::ofstream ofs(filename, std::ios::binary);
    return WriteMatBinary(ofs, output);
}


//! Read cv::Mat from binary
/*!
\param[in] ifs input file stream
\param[out] in_mat mat to load
*/
bool readMatBinary(std::ifstream& ifs, cv::Mat& in_mat)
{
    if(!ifs.is_open()){
        return false;
    }

    int rows, cols, type;
    ifs.read((char*)(&rows), sizeof(int));
    if(rows==0){
        return true;
    }
    ifs.read((char*)(&cols), sizeof(int));
    ifs.read((char*)(&type), sizeof(int));

    in_mat.release();
    in_mat.create(rows, cols, type);
    ifs.read((char*)(in_mat.data), in_mat.elemSize() * in_mat.total());

    return true;
}


//! Load cv::Mat as binary
/*!
\param[in] filename filaname to load
\param[out] output loaded cv::Mat
*/
bool LoadMatBinary(const std::string& filename, cv::Mat& output){
    std::ifstream ifs(filename, std::ios::binary);
    return ReadMatBinary(ifs, output);
}

int main() {
    std::string path = "/home/oracle/Project/test/";  // folder where YAML data is saved
    std::string storage = "/home/oracle/Project/small_kinopoisk/";  // folder with pictures
    Mat src = imread(storage + "1.jpg", CV_LOAD_IMAGE_UNCHANGED);
    std::vector<KeyPoint> keys;
    Mat descriptor;
    Ptr<Feature2D> f2d = SIFT::create(0, 3, 0.07, 5, 1.6);
    f2d -> detectAndCompute(src, Mat(), keys, descriptor);
    FileStorage fs_descriptor (path + "descr.yml.gz", FileStorage::WRITE);
    FileStorage fs_keys (path + "keys.yml.gz", FileStorage::WRITE);
    fs_descriptor << "descr" << descriptor;
    fs_keys << "keys" << keys;
//    FileStorage fs_descriptor (path + "descr.yml.gz", FileStorage::READ);
//    FileStorage fs_keys (path + "keys.yml.gz", FileStorage::READ);
//    fs_descriptor["descr"] >> descriptor;
//    fs_keys["keys"] >> keys;
//    return 0;
}