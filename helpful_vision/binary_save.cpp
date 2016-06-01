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

void writeBinaryMat(std::string& filename, const Mat& output) {
    std::ofstream ofs(filename, std::ios::binary);
    int type = output.type();
    ofs.write((const char*)(&type), sizeof(type));
    ofs.write((const char*)(&output.rows), sizeof(int));
    ofs.write((const char*)(&output.cols), sizeof(int));
    ofs.write((const char*)(output.data), output.elemSize() * output.total());
    ofs.close();
}

void readBinaryMat(std::string& filename, cv::Mat& input)
{
    std::ifstream ifs(filename, std::ios::binary);
    int rows, cols, type;
    ifs.read((char*)(&type), sizeof(int));
    ifs.read((char*)(&rows), sizeof(int));
    ifs.read((char*)(&cols), sizeof(int));
    input.release();
    input.create(rows, cols, type);
    ifs.read((char*)(input.data), input.elemSize() * input.total());
    ifs.close();
}

int main() {
    std::string path = "/home/oracle/Project/test/disc.bin";  // folder where YAML data is saved
    std::string storage = "/home/oracle/Project/small_kinopoisk/";  // folder with pictures
    Mat src = imread(storage + "1.jpg", CV_LOAD_IMAGE_UNCHANGED);
    std::vector<KeyPoint> keys;
    Mat descriptor;
    Ptr<Feature2D> f2d = SIFT::create(0, 3, 0.07, 5, 1.6);
    f2d -> detectAndCompute(src, Mat(), keys, descriptor);
    descriptor.resize(300);
    writeBinaryMat(path, descriptor);
//    FileStorage fs_descriptor (path + "descr.yml.gz", FileStorage::WRITE);
//    FileStorage fs_keys (path + "keys.yml.gz", FileStorage::WRITE);
//    fs_descriptor << "descr" << descriptor;
//    fs_keys << "keys" << keys;
//    FileStorage fs_descriptor (path + "descr.yml.gz", FileStorage::READ);
//    FileStorage fs_keys (path + "keys.yml.gz", FileStorage::READ);
//    fs_descriptor["descr"] >> descriptor;
//    fs_keys["keys"] >> keys;
//    Mat readMat;
//    readBinaryMat(path, readMat);
//    std::cout << readMat.cols << std::endl;
    return 0;
}