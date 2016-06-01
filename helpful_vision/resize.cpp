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

std::string storage = "/home/oracle/Project/kinopoisk/";  // folder with pictures
std::string destination = "/home/oracle/Project/small_kinopoisk/";
std::string path = "/home/oracle/Project/data/";

using namespace cv;
using namespace cv::xfeatures2d;

int main() {
    int n = 1000000;
    int j = 0;
    FileStorage fs_map(path + "map.yml", FileStorage::WRITE);
    for (int i = 0; i != n; ++i) {
        Mat src = imread(storage + std::to_string(i) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);  // +1 'cause of the images names
        if (src.empty()) {
            continue;
        }
        resize(src, src, Size(480, 640), 0, 0, INTER_LINEAR);
        imwrite(destination + std::to_string(j) + ".jpg", src);
        fs_map << "image_" + std::to_string(j) << i;
        ++j;
    }
    fs_map.release();
    return 0;
}


