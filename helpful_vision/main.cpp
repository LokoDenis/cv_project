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
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;
using namespace cv;
using namespace cv::xfeatures2d;


int main() {
    std::string storage = "/home/oracle/Project/telegrambot/";  // folder with pictures
    std::vector<int> fails;
    for (fs::recursive_directory_iterator it("/home/oracle/Project/telegrambot/"), end; it != end; ++it) {
        if (it->path().extension() == ".jpg") {
            Mat src = imread(it->path().generic_string(), CV_LOAD_IMAGE_UNCHANGED);
            if (src.empty()) {
                std::cout << *it << std::endl;
                fs::remove(it -> path().generic_string());
            }
        }
    }
}