#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string>

using namespace cv;  //Is it ok to use namespace?

int main() {
    Mat source = cv::imread("/home/oracle/Project/Images/image.jpg", CV_LOAD_IMAGE_GRAYSCALE);


    std::string name = "Sobel Demo";
    namedWindow(name, CV_WINDOW_AUTOSIZE);
    int depth = CV_16S;

    // Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    GaussianBlur(source, source, Size(3,3), 0, 0, BORDER_DEFAULT);

    // Gradient X
//    Scharr(source, grad_x, depth, 1, 0, 1, 0, BORDER_DEFAULT );  // scale = one cause we're in gray, delta = 0 - default
    Sobel(source, grad_x, depth, 1, 0, 3, 1, 0, BORDER_DEFAULT);

    // Gradient Y
//    Scharr(source, grad_y, depth, 0, 1, 1, 0, BORDER_DEFAULT);
    Sobel(source, grad_y, depth, 1, 0, 3, 1, 0, BORDER_DEFAULT);

    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    Mat grad = abs_grad_x.clone();

    for (size_t i = 0; i != grad.rows; ++i) {
        for (size_t j = 0; j != grad.cols; ++j) {
            grad.at<uchar>(i, j) = sqrt(pow(abs_grad_x.at<uchar>(i, j), 2) + pow(abs_grad_y.at<uchar>(i, j), 2));
        }
    }

    Mat sized_grad; // image too large to fill the screen
    resize(grad, sized_grad, sized_grad.size(), 1, 1, INTER_LINEAR);

    imshow(name, sized_grad);
    waitKey(0);
    imwrite("/home/oracle/ClionProjects/comvision/Gradients/Image.jpg", sized_grad);
    return 0;
}