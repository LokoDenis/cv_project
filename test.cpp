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

struct Image {
    Mat description;  // mat of descriptor for each Image
    std::vector<double> word;  // visual words
};

void createABase(std::string& path, int n, std::vector<Image>& data, Mat& collection) {
    for (size_t i = 0; i != n; ++i) {
        Mat src = imread(path + std::to_string(i + 1) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);  // +1 'cause of the images names
        if (src.empty()) continue;
        Image newElement;
        resize(src, src, Size(600, 700), 0, 0, INTER_LINEAR);
        Ptr<Feature2D> f2d = SIFT::create(0, 3, 0.18, 5, 1.6);
        Mat descriptor;
        std::vector<KeyPoint> keypoints;
        f2d->detectAndCompute(src, Mat(), keypoints, descriptor);
        newElement.description = descriptor;
        data.push_back(newElement);
        collection.push_back(descriptor);
    }
}

void computeVisualWords(std::string& path, std::vector<Image>& data, Mat& clusterCenters, Mat& collection) {
    int K = collection.rows / 80;
    std::vector<int> labels(collection.rows);
    kmeans(collection, K, labels,
           cvTermCriteria(CV_TERMCRIT_EPS, 500, 0.001), 1, KMEANS_PP_CENTERS, clusterCenters);
    // kmeans(InputArray data, int K, InputOutputArray bestLabels,
    // TermCriteria criteria, int attempts,
    // int flags, OutputArray clusterCenters=noArray())
    // creating vectors of clusters' frequencies
    unsigned int i = 0;  // creating vectors of clusters' frequencies
    while (i < labels.size()) {
        for (auto &elem : data) {
            std::vector<double> visualWord(K, 0);
            for (size_t j = 0; j != elem.description.rows; ++j) {
                ++visualWord[labels[i]];
                ++i;
            }
            elem.word = visualWord;
        }
    }
}

size_t findMinIndex(std::vector<double>& data) {
    size_t min = 0;
    for (size_t i = 1; i != data.size(); ++i) {
        if (data[min] > data[i]) {
            min = i;
        }
    }
    return min;
}

int main() {
    std::string path = "/home/oracle/Project/data/";
    std::string storage = "/home/oracle/Project/kinopoisk/";  // folder with pictures
    int k;  // approximate number of pictures in the base
    std::cout << "How many pictures: ";
    std::cin >> k;
    std::vector<Image> data;  // vector of computed pictures
    Mat clusterCenters;
    Mat collection;  // matrix with all descriptors for kmeans
    createABase(storage, k, data, collection);  // creating straight and inverted index dictionaries
    computeVisualWords(path, data, clusterCenters, collection);

    std::cout << "Which word do you want to check? ";
    std::cin >> k;
    Image newElement;
    Mat src = imread(storage + std::to_string(k) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);  // reading Image and calculating its descriptor
    resize(src, src, Size(600, 700), 0, 0, INTER_LINEAR);
    Ptr<Feature2D> f2d = SIFT::create(0, 3, 0.18, 5, 1.6);
    Mat descriptor;
    std::vector<KeyPoint> keys;
    f2d->detectAndCompute(src, Mat(), keys, descriptor);
    newElement.description = descriptor;

    int K = clusterCenters.rows;  // quantity of clusters
    std::vector<int> labels(descriptor.rows);
    std::vector<std::vector<double>> distances(descriptor.rows, std::vector<double>(clusterCenters.rows, 0));
    double currentDistance = 0;
    for (size_t j = 0; j != descriptor.rows; ++j) {
        float *currentDot = descriptor.ptr<float>(j);
        for (size_t i = 0; i != clusterCenters.rows; ++i) {
            float *currentRow = clusterCenters.ptr<float>(i);
            currentDistance = normL2Sqr(currentDot, currentRow, descriptor.cols);
            distances[j][i] = currentDistance;
        }
    }

    //  calculation the cluster number for each descriptor
    for (size_t i = 0; i != labels.size(); ++i) {
        labels[i] = findMinIndex(distances[i]);
    }

    unsigned int i = 0;  // creating vectors of clusters' frequencies
    while (i < labels.size()) {
        std::vector<double> visualWord(K, 0);
        for (size_t j = 0; j != descriptor.rows; ++j) {
            ++visualWord[labels[i]];
            ++i;
        }
        newElement.word = visualWord;
    }

    std::cout << "Manual visual word:";
    for (size_t i = 0; i != newElement.word.size(); ++i) {
        std::cout << " " << newElement.word[i];
    }

    std::cout << "\nKmeans visual word:";
    for (size_t j = 0; j != data[k - 1].word.size(); ++j) {
        std::cout << " " << data[k - 1].word[j];
    }

    return 0;
}
