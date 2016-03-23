#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <vector>
#include <opencv2/imgproc.hpp>
#include <string>

using namespace cv;
using namespace cv::xfeatures2d;

int main() {
    Mat src = imread("/home/oracle/Project/Images/6th2.jpg", CV_LOAD_IMAGE_UNCHANGED);

    Ptr<Feature2D> f2d = SIFT::create(0, 3, 0.08, 10, 1.6);
    //SIFT(int nfeatures=0, int nOctaveLayers=3,
    // double contrastThreshold=0.04, double edgeThreshold=10, double sigma=1.6)

    std::vector<KeyPoint> keys;
    Mat descriptor;
    resize(src, src, Size(600, 800), 0, 0, INTER_LINEAR);

    f2d->detectAndCompute(src, Mat(), keys, descriptor);

    Mat drawing;
    std::string name;
    name = "keys";
    drawKeypoints(src, keys, drawing, Scalar(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    namedWindow(name, CV_WINDOW_AUTOSIZE);
    imshow (name, drawing);
    waitKey(0);

    int K = descriptor.rows / 80;
    std::vector<int> labels(descriptor.rows);
    kmeans(descriptor, K, labels,  cvTermCriteria(CV_TERMCRIT_NUMBER + CV_TERMCRIT_EPS, 10000, 0.001), 10, KMEANS_PP_CENTERS);
    // kmeans(InputArray data, int K, InputOutputArray bestLabels,
    // TermCriteria criteria, int attempts,
    // int flags, OutputArray centers=noArray())
    //show dotes of claster m;
    for (size_t i = 0; i != keys.size(); ++i) {
        for (int m = 0; m != K; ++m) {
            if (labels[i] == m) {
                circle(src, keys[i].pt, 3, m * 15, CV_FILLED, 8, 0);
            }
        }
    }

    // creating visual words
    size_t i = 0;
    while (i != labels.size()) {
        std::vector<int> word (K, 0);
        ++word[labels[i]];
    }

    name = "cluster";
    namedWindow(name);
    imshow(name, src);
    waitKey(0);
    destroyWindow(name);

    imwrite("/home/oracle/ClionProjects/comvision/clusters/example.jpg", src);

return 0;
}