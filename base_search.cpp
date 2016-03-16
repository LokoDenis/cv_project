#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <vector>
#include <opencv2/imgproc.hpp>
#include <string>

using namespace cv;

struct image {
    Mat source;
    std::vector<KeyPoint> keys;
};

int main() {
    std::string path = "/home/oracle/Project/kinopoisk/";
    int n = 1;  // choose how much do we want
    std::vector<image> data;
    for (size_t i = 298; i != n; ++i) {
        Mat src = imread(path + std::to_string(i) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);
        if (src.empty()) continue;

        image new_elem;
        resize(src, new_elem.source, Size(600, 700), 0, 0, INTER_LINEAR);
        Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(0, 3, 0.18, 5, 1.6);
        Mat descriptor;
        std::vector<KeyPoint> k;
        f2d -> detectAndCompute(src, Mat(), k, descriptor);

        new_elem.source = src;
        new_elem.keys = k;
        resize(src, new_elem.source, Size(600, 700), 0, 0, INTER_LINEAR);

        namedWindow("elem", CV_WINDOW_AUTOSIZE);
        imshow("elem", new_elem.source);
        waitKey(0);
        destroyWindow("elem");

        data.push_back(new_elem);
    }

    return 0;
}