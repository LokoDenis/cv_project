#include <cmath>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <string>
#include <vector>

using namespace cv;
using namespace cv::xfeatures2d;

struct image {
    Mat description;  // mat of descriptor for each image
    std::vector<double> word;  // visual words
};

void TF_IDF(std::vector<image>& data, int K) {
    std::vector<int> count_no_zeros(K, 0);  // vectors with dotes in particular clusters (for IDF)
    std::vector<int> total_dotes_appear(data.size(), 0);  // total sum of dotes in each image (for TF)

    for (size_t j = 0; j != data.size(); ++j) {
        for (size_t i = 0; i != data[j].word.size(); ++i) {
            total_dotes_appear[j] += data[j].word[i];
            if (data[j].word[i] != 0) {
                ++count_no_zeros[i];
            }
        }
    }

    for (size_t picture = 0; picture != data.size(); ++picture) {
        for (size_t cluster = 0; cluster != data[picture].word.size(); ++cluster) {
            data[picture].word[cluster] = data[picture].word[cluster] / total_dotes_appear[picture] * log(data.size() / count_no_zeros[cluster]);
        }
    }
}

// walking through the base, reading raw data for further clustering
void base_walk(std::string path, int n, std::vector<image>& data, Mat& collection) {
        n += 298;
        std::cout << "base_walk begin";
        for (size_t i = 298; i != n; ++i) {
            std::cout << "\n" << i;
        Mat src = imread(path + std::to_string(i) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);
        if (src.empty()) continue;
        image new_elem;
        resize(src, src, Size(600, 700), 0, 0, INTER_LINEAR);
        Ptr<Feature2D> f2d = SIFT::create(0, 3, 0.18, 5, 1.6);
        Mat descriptor;
        std::vector<KeyPoint> k;
        f2d->detectAndCompute(src, Mat(), k, descriptor);
        new_elem.description = descriptor;
        data.push_back(new_elem);
        collection.push_back(descriptor);
    }
    std::cout << "\nbase_walk end \n";
}

// creating visual words for pictures using clustering and TF-IDF metric
void to_visual_words(std::vector<image>& data, Mat& collection) {
    int K = collection.rows / 80;
    std::vector<int> labels(collection.rows);
    std::cout << "\nclustering begin\n";
    kmeans(collection, K, labels,
           cvTermCriteria(CV_TERMCRIT_NUMBER + CV_TERMCRIT_EPS, 100, 1.0), 2, KMEANS_PP_CENTERS);
    // kmeans(InputArray data, int K, InputOutputArray bestLabels,
    // TermCriteria criteria, int attempts,
    // int flags, OutputArray centers=noArray())
    std::cout << "\nclusterig end'\n";
    std::cout << "\ncreating words begin:";
    unsigned int i = 0;  // creating vectors of clusters' frequencies
    while (i < labels.size()) {
        int j = 0;
    for (auto& elem : data) {
        std::cout << j << " ";
        std::vector<double> word(K, 0);
            for (size_t j = 0; j != elem.description.rows; ++j) {
                ++word[labels[i]];
                ++i;
            }
        elem.word = word;
        ++j;
        }
    }
    std::cout << "\ncreating words end";
    std::cout << "\n TF_IDF begin";
    TF_IDF(data, K);
    std::cout << "\n TF_IDF end";
}

// saving data of descriptors and words (counted with the usage of TF-IDF metric) to the user's path
void save(std::string path, std::vector<image>& data) {
    std::cout << "\nsave begin";
    FileStorage fs_description(path + "descriptors.yml", FileStorage::WRITE);
    FileStorage fs_words(path + "words.yml", FileStorage::WRITE);
    size_t i = 0;
    for (auto& elem : data) {
        fs_description << "data_" + std::to_string(i) + "_description" << elem.description;
        fs_words << "data_" + std::to_string(i) + "_word" << elem.word;
        ++i;
    }
    std::cout << "\n save end";
}

int main() {
    int k = 10;
    std::string path = "/home/oracle/Project/kinopoisk/";
    std::vector<image> data;  // vector of computed pictures
    Mat collection;  // matrix with all descriptors for kmeans
    base_walk(path, k, data, collection);
    to_visual_words(data, collection);
    path = "/home/oracle/Project/data/";
    save(path, data);
    return 0;
}