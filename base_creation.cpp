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
#include <string>
#include <vector>

// Tasks:
// add clusters' centres and save them YEP
// save data for calculating TF and IDF for a new picture YEP

using namespace cv;
using namespace cv::xfeatures2d;

struct image {
    Mat description;  // mat of descriptor for each image
    std::vector<double> word;  // visual words
};

void TF_IDF(std::string& path, std::vector<image>& data, int K) {
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

    std::cout << "\nwrite begin";
    FileStorage fs_idf(path + "idf.yml", FileStorage::WRITE);
    FileStorage fs_tf(path + "tf.yml", FileStorage::WRITE);
    fs_idf << "dots" << count_no_zeros;
    fs_tf << "sums" << total_dotes_appear;
    fs_idf.release();
    fs_tf.release();
    std::cout << "\nwrite end";

    for (size_t picture = 0; picture != data.size(); ++picture) {
        for (size_t cluster = 0; cluster != data[picture].word.size(); ++cluster) {
            data[picture].word[cluster] = data[picture].word[cluster] / total_dotes_appear[picture] * log(data.size() / count_no_zeros[cluster]);
        }
    }
}

// walking through the source, reading raw data for further clustering
void base_walk(std::string& path, int n, std::vector<image>& data, Mat& collection) {
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
void to_visual_words(std::string&path, std::vector<image>& data, Mat& centers, Mat& collection) {
    int K = collection.rows / 80;
    std::vector<int> labels(collection.rows);
    std::cout << "\nclustering begin\n";
    kmeans(collection, K, labels,
           cvTermCriteria(CV_TERMCRIT_EPS, 500, 0.001), 1, KMEANS_PP_CENTERS, centers);
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
    TF_IDF(path, data, K);
    std::cout << "\n TF_IDF end";
}

// saving data of descriptors, clusters' centers, words (counted with the usage of TF-IDF metric) and dicts to the user's path
void save(std::string& path, std::vector<image>& data, Mat& centers, std::map<int, std::vector<double>>& straight_index,
          std::vector<std::vector<int>>& inverted_index) {
    std::cout << "\nsave begin";
    FileStorage fs_description(path + "descriptors.yml", FileStorage::WRITE);
    FileStorage fs_words(path + "words.yml", FileStorage::WRITE);
    FileStorage fs_centers(path + "centers.yml", FileStorage::WRITE);
    FileStorage fs_straight_index(path + "straight_index.yml", FileStorage::WRITE);
    FileStorage fs_inverted_index(path + "inverted_index.yml", FileStorage::WRITE);
    size_t i = 0;
    for (auto &elem : data) {
        fs_description << "data_" + std::to_string(i) + "_description" << elem.description;
        fs_words << "data_" + std::to_string(i) + "_word" << elem.word;
        ++i;
    }
        fs_description.release();
        fs_words.release();

        fs_centers << "centers" << centers;
        fs_centers.release();

        i = 0;
        for (auto &elem : straight_index) {
            fs_straight_index << "straingt_index_" + std::to_string(i++) << elem.second;
        }
        fs_straight_index.release();

        for (size_t i = 0; i != inverted_index.size(); ++i) {
            fs_inverted_index << "inverted_index_ " + std::to_string(i) << inverted_index[i];
        }
        fs_inverted_index.release();
        std::cout << "\n save end";
    }

int main() {
    int choice;
    do {
        std::cout << "\nWhat do you want to do?" << std::endl;
        std::cout << "1) Create a base" << std::endl;
        std::cout << "2) Add a picture" << std::endl;
        std::cout << "3) Search for the picture" << std::endl;
        std::cout << "4) Exit" << std::endl;
        std::cin >> choice;
        switch (choice) {
            case 1: {
                std::string storage = "/home/oracle/Project/kinopoisk/";  // pictures folder
                std::string path = "/home/oracle/Project/data/";  // save data folder
                int k = 150;
                std::vector<image> data;  // vector of computed pictures
                Mat centers;
                Mat collection;  // matrix with all descriptors for kmeans
                base_walk(storage, k, data, collection);
                to_visual_words(path, data, centers, collection);
                // creating straight index dict
                std::map<int, std::vector<double>> straight_index;
                for (size_t i = 0; i != data.size(); ++i) {
                    straight_index.insert(std::pair<int, std::vector<double>>(i, data[i].word));
                }

                //creating inverted index dict with keys - numbers in cluster. Default initialized with -1
                std::vector<std::vector<int>> inverted_index (centers.rows);

                for (size_t i = 0; i != data.size(); ++i) {
                    for (size_t j = 0; j != data[i].word.size(); ++j) {
                        if (data[i].word[j] > 0) {
                            inverted_index[j].push_back(i);
                        }
                    }
                }

                save(path, data, centers, straight_index, inverted_index);
                break;
            }
            case 2:

                // read data
                // calculate tf-idf for the current
                // add to lists
                break;

            case 3: break;

            case 4: break;
            default:
            std::cout << "Incorrect choice. You should enter 1, 2, 3 or 4" << std::endl;
                break;
        }
    } while (choice != 4);

    return 0;
}