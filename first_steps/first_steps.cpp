#include <iostream>
#include <opencv2/core.hpp> // Mat structure
#include <opencv2/highgui/highgui.hpp> //imread and etc.
#include <opencv2/imgproc/imgproc.hpp>


int main() {
    cv::Mat img = cv::imread("/home/oracle/ClionProjects/comvision/Abbey_Road.jpeg", CV_LOAD_IMAGE_COLOR);

    if (img.empty()) {
        std::cerr << "Image was not loaded. Shutting down.";
        return -1;
    }

    cv::namedWindow("Beatles", CV_WINDOW_AUTOSIZE);
    cv::imshow("Beatles", img);
    cv::waitKey(0);
    cv::destroyWindow("Beatles");

    // inverting colours through changing pixels' values
    cv::Mat negative = img.clone(); // creating a full copy as if "negative = img" means a pointer to img (if I'm right)

    for (size_t i = 0; i != negative.rows; ++i) {
        for (size_t j = 0; j != negative.cols; ++ j) {
                negative.at<cv::Vec3b>(i, j)[0] = static_cast<uchar> (255 - negative.at<cv::Vec3b>(i, j)[0]);
                negative.at<cv::Vec3b>(i, j)[1] = static_cast<uchar> (255 - negative.at<cv::Vec3b>(i, j)[1]);
                negative.at<cv::Vec3b>(i, j)[2] = static_cast<uchar> (255 - negative.at<cv::Vec3b>(i, j)[2]);
            }

    }

    cv::namedWindow("Negative", CV_WINDOW_NORMAL);
    cv::imshow("Negative", negative);
    cv::imwrite("/home/oracle/ClionProjects/comvision/first_steps/negative.jpeg", negative);
    cv::waitKey(0);
    cv::destroyWindow("Negative");


    //resizing the negative picture
    cv::Mat resized;
    cv::resize(negative, resized, resized.size(), 0.7, 0.7, cv::INTER_LINEAR);
    cv::namedWindow("Resized", CV_WINDOW_NORMAL);
    cv::imshow("Resized", resized);
    cv::imwrite("/home/oracle/ClionProjects/comvision/first_steps/resized.jpeg", resized);
    cv::waitKey(0);
    cv::destroyWindow("Resized");

    //rotating the negative picture
    cv::Mat rotated;
    cv::Point2d src_center(resized.cols * 0.5, resized.rows * 0.5); // defining a center of the source picture
    cv::Mat rot_mat = cv::getRotationMatrix2D(src_center, 30 , 1);  // creating a rotation matrix
    warpAffine(resized, rotated, rot_mat, resized.size());
    cv::namedWindow("Rotated", CV_WINDOW_NORMAL);
    cv::imshow("Rotated", rotated);
    cv::imwrite("/home/oracle/ClionProjects/comvision/first_steps/rotated.jpeg", rotated);
    cv::waitKey(0);
    cv::destroyWindow("Rotated");

    return 0;
}


