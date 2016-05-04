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
    Mat src = imread("/home/oracle/Project/Images/lena.jpg", CV_LOAD_IMAGE_UNCHANGED);
    Mat sec_src = imread("/home/oracle/Project/Images/exam.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat third_src = imread("/home/oracle/Project/kinopoisk/56.jpg", CV_LOAD_IMAGE_UNCHANGED);
    Ptr<Feature2D> f2d = SURF::create(1000, 1, 2, 1, 0);
    Ptr<Feature2D> f2d_sift = SIFT::create(0, 3, 0.15, 5, 1.6); //  trying cycles
    // Setup SimpleBlobDetector parameters.
    SimpleBlobDetector::Params params;

    //rotating pictures
    cv::Mat src_rotated;
    cv::Point2d src_center(src.cols * 0.5, src.rows * 0.5); // defining a center of the source picture
    cv::Mat rot_mat = cv::getRotationMatrix2D(src_center, 20 , 1);  // creating a rotation matrix
    warpAffine(src, src_rotated, rot_mat, src.size());

    cv::Mat sec_src_rotated;
    cv::Point2d sec_src_center(sec_src.cols * 0.5, sec_src.rows * 0.5); // defining a center of the source picture
    cv::Mat sec_rot_mat = cv::getRotationMatrix2D(sec_src_center, 50 , 1);  // creating a rotation matrix
    warpAffine(sec_src, sec_src_rotated, sec_rot_mat, sec_src.size());

    cv::Mat third_src_rotated;
    int angle = 20;
    cv::Point2d third_src_center(third_src.cols * 0.5, third_src.rows * 0.5); // defining a center of the source picture
    cv::Mat third_rot_mat = cv::getRotationMatrix2D(third_src_center, angle , 1);  // creating a rotation matrix
    warpAffine(third_src, third_src_rotated, third_rot_mat, third_src.size());

    // Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 250;

    // Filter by Area.
    params.filterByArea = true;
    params.minArea = 50;

    // Filter by Circularity
    params.filterByCircularity = true;
    params.minCircularity = 0.7;

    // Filter by Convexity
    params.filterByConvexity = true;
    params.minConvexity = 0.9;

    // Filter by Inertia
    params.filterByInertia = true;
    params.minInertiaRatio = 0.6;
    params.maxInertiaRatio = 1;

    Ptr<SimpleBlobDetector> blobe_detector = SimpleBlobDetector::create(params);

     //inverting colors
//    for (size_t i = 0; i != third_src_rotated.rows; ++i) {
//        for (size_t j = 0; j != third_src_rotated.cols; ++ j) {
//            third_src_rotated.at<cv::Vec3b>(i, j)[0] = static_cast<uchar> (255 - third_src_rotated.at<cv::Vec3b>(i, j)[0]);
//            third_src_rotated.at<cv::Vec3b>(i, j)[1] = static_cast<uchar> (255 - third_src_rotated.at<cv::Vec3b>(i, j)[1]);
//            third_src_rotated.at<cv::Vec3b>(i, j)[2] = static_cast<uchar>  (255 - third_src_rotated.at<cv::Vec3b>(i, j)[2]);
//        }
//    }

    //resize in case image too large to fill the screen
    Mat sized_src_one;
    Mat sized_src_two;
    Mat sized_src_blobe_one;
    Mat sized_src_blobe_two;
    Mat sized_src_sift_one = third_src;
    Mat sized_src_sift_two = third_src_rotated;

    resize(src, sized_src_one, Size(600, 800), 0, 0, INTER_LINEAR);
    resize(src_rotated, sized_src_two, Size(600, 800), 0, 0, INTER_LINEAR);
    resize(sec_src, sized_src_blobe_one, Size(600, 600), 0, 0, INTER_LINEAR);
    resize(sec_src_rotated, sized_src_blobe_two, Size(600, 600), 0, 0, INTER_LINEAR);
//    resize(third_src, sized_src_sift_one, Size(600, 800), 0, 0, INTER_LINEAR);
//    resize(third_src_rotated, sized_src_sift_two, Size(600, 800), 0, 0, INTER_LINEAR);


    // descriptors and detectors

    Mat descriptor_one, descriptor_two, descriptor_sift_one, descriptor_sift_two;
    std::vector<KeyPoint> points_one, points_two, blobe_points_one, blobe_points_two;
    std::vector<KeyPoint> points_sift_one, points_sift_two;
    f2d -> detectAndCompute(sized_src_one, Mat(), points_one, descriptor_one); //surf
    f2d -> detectAndCompute(sized_src_two, Mat(), points_two, descriptor_two);
    f2d_sift -> detectAndCompute (sized_src_sift_one, Mat(), points_sift_one, descriptor_sift_one);
    f2d_sift -> detectAndCompute(sized_src_sift_two, Mat(), points_sift_two, descriptor_sift_two);
    blobe_detector -> detect(sized_src_blobe_one, blobe_points_one);
    blobe_detector -> detect(sized_src_blobe_two, blobe_points_two);

    //drawing keypoints

    Mat keys_one, keys_two, blobe_keys_one, blobe_keys_two, sift_keys_one, sift_keys_two;

    drawKeypoints(sized_src_one, points_one, keys_one, Scalar(0, 0, 255), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    drawKeypoints(sized_src_two, points_two, keys_two, Scalar(0, 0, 255), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    drawKeypoints(sized_src_blobe_one, blobe_points_one, blobe_keys_one, Scalar(0,0,255), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    drawKeypoints(sized_src_blobe_two, blobe_points_two, blobe_keys_two, Scalar(0,0,255), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    drawKeypoints(sized_src_sift_one, points_sift_one, sift_keys_one, Scalar(0,0,255), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    drawKeypoints(sized_src_sift_two, points_sift_two, sift_keys_two, Scalar(0,0,255), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    namedWindow("SURF", CV_WINDOW_AUTOSIZE);
    namedWindow("SURF2", CV_WINDOW_AUTOSIZE);
    imshow ("SURF", keys_one);
    imshow ("SURF2", keys_two);
    waitKey(0);

//    destroyWindow("SURF");
//    destroyWindow("SURF2");

    namedWindow("BLOBE", CV_WINDOW_AUTOSIZE);
    namedWindow("BLOBE2", CV_WINDOW_AUTOSIZE);
    imshow ("BLOBE", blobe_keys_one);
    imshow ("BLOBE2", blobe_keys_two);
    waitKey(0);
    destroyWindow("BLOBE");
    destroyWindow("BLOBE2");

    namedWindow("SIFT", CV_WINDOW_AUTOSIZE);
    namedWindow("SIFT2", CV_WINDOW_AUTOSIZE);
    imshow ("SIFT", sift_keys_one);
    imshow ("SIFT2", sift_keys_two);
    waitKey(0);
    destroyWindow("SIFT");
    destroyWindow("SIFT2");

    //writing keypoints to files

    imwrite("/home/oracle/ClionProjects/comvision/Detected_dotes/surf1.jpg", keys_one);
    imwrite("/home/oracle/ClionProjects/comvision/Detected_dotes/surf2.jpg", keys_two);
    imwrite("/home/oracle/ClionProjects/comvision/Detected_dotes/blobe1.jpg", blobe_keys_one);
    imwrite("/home/oracle/ClionProjects/comvision/Detected_dotes/blobe2.jpg", blobe_keys_two);
    imwrite("/home/oracle/ClionProjects/comvision/Detected_dotes/sift1.jpg", sift_keys_one);
    imwrite("/home/oracle/ClionProjects/comvision/Detected_dotes/sift1.jpg", sift_keys_two);

    //matching
    BFMatcher matcher (NORM_L2, true);
    BFMatcher matcher_sift (NORM_L2, true);
    std::vector<DMatch> surf_matches_vector;
    std::vector<DMatch> sift_matches_vector;
    matcher.match(descriptor_one, descriptor_two, surf_matches_vector);
    matcher_sift.match(descriptor_sift_one, descriptor_sift_two, sift_matches_vector);

    //showing results of matching
    namedWindow("SUFR_matches", 1);
    Mat surf_matches;
    drawMatches(keys_one, points_one, keys_two, points_two, surf_matches_vector, surf_matches);
    imshow ("SURF_matches", surf_matches);

    waitKey(0);

    namedWindow("SIFT_matches", 1);
    Mat sift_matches;
    drawMatches(sift_keys_one, points_sift_one, sift_keys_two, points_sift_two, sift_matches_vector, sift_matches);
    imshow ("SIFT_matches", sift_matches);

    waitKey(0);
    
    imwrite("/home/oracle/ClionProjects/comvision/Detected_dotes/surf_matches.jpg", surf_matches);
    imwrite("/home/oracle/ClionProjects/comvision/Detected_dotes/siftf_matches.jpg", sift_matches);

    return 0;
}