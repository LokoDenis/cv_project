#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <stdio.h>
#include <vector>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

int main() {
    Mat src = imread("/home/oracle/Project/small_kinopoisk/12547.jpg", CV_LOAD_IMAGE_UNCHANGED);
    Ptr<Feature2D> f2d = SIFT::create(0, 3, 0.04, 5, 1.6);
    GaussianBlur(src, src, Size(5, 5), 2, 2, BORDER_DEFAULT);
    //medianBlur(third_src, third_src, 3);

    Mat descriptor_one;
    std::vector<KeyPoint> points_sift_one;
    f2d -> detect(src, points_sift_one);
    std::sort(points_sift_one.rbegin(), points_sift_one.rend(),[](KeyPoint a, KeyPoint b){ return a.response * a.size < b.response * b.size;});
    points_sift_one.resize(300);
    f2d -> compute(src, points_sift_one, descriptor_one);

    //drawing keypoints

    Mat keys_one, keys_two, blobe_keys_one, blobe_keys_two, sift_keys_one, sift_keys_two;

    drawKeypoints(src, points_sift_one, keys_one, Scalar(0, 0, 255), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    namedWindow("SIFT", CV_WINDOW_AUTOSIZE);
    imshow ("SIFT", keys_one);
    waitKey(0);
    destroyWindow("SIFT");
    return 0;
}