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

using namespace cv;
using namespace cv::xfeatures2d;

int main() {
    int n = 6869;
for (size_t i = 0; i != n; ++i) {
    Mat src = imread(storage + std::to_string(i + 1) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);  // +1 'cause of the images names
    if (src.empty()) {
        throw std::invalid_argument("Check " + std::to_string(i + 1) + " picture");
    }
    resize(src, src, Size(480, 640), 0, 0, INTER_LINEAR);
    imwrite(destination + std::to_string(i + 1) + ".jpg", src);
}
    return 0;
}


