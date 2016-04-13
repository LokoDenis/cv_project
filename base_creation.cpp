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

// Current Task: 1) get to know how FileStorage::APPEND works.
//               2) search for an image.

using namespace cv;
using namespace cv::xfeatures2d;

struct Image {
    Mat description;  // mat of descriptor for each Image
    std::vector<double> word;  // visual words
};

void countMetric(std::string& path, std::vector<Image>& data, int K) {
    std::vector<int> countNoZeros(K, 0);  // vectors with dotes in particular clusters (for IDF)
    std::vector<int> quantityOfDotes(data.size(), 0);  // total sum of dotes in each Image (for TF)

    for (size_t j = 0; j != data.size(); ++j) {
        for (size_t i = 0; i != data[j].word.size(); ++i) {
            quantityOfDotes[j] += data[j].word[i];
            if (data[j].word[i] != 0) {
                ++countNoZeros[i];
            }
        }
    }

    std::cout << "\nwrite begin";
    FileStorage fs_idf(path + "idf.yml", FileStorage::WRITE);
    fs_idf << "dots" << countNoZeros;
    fs_idf << "size" << static_cast<int>(data.size());
    fs_idf.release();
    std::cout << "\nwrite end";

    for (size_t picture = 0; picture != data.size(); ++picture) {
        for (size_t cluster = 0; cluster != data[picture].word.size(); ++cluster) {
            data[picture].word[cluster] = data[picture].word[cluster] / quantityOfDotes[picture] * log(data.size() / countNoZeros[cluster]);
        }
    }
}

// walking through the source, reading raw data for further clustering
void createABase(std::string& path, int n, std::vector<Image>& data, Mat& collection) {
        n += 298;
        std::cout << "createABase() begin";
        for (size_t i = 298; i != n; ++i) {
            std::cout << "\n" << i;
        Mat src = imread(path + std::to_string(i) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);
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
    std::cout << "\ncreateABase() end \n";
}

// creating visual words for pictures using clustering and TF-IDF metric
void computeVisualWords(std::string& path, std::vector<Image>& data, Mat& clusterCenters, Mat& collection) {
    int K = collection.rows / 80;
    std::vector<int> labels(collection.rows);
    std::cout << "\nclustering begin\n";
    kmeans(collection, K, labels,
           cvTermCriteria(CV_TERMCRIT_EPS, 500, 0.001), 1, KMEANS_PP_CENTERS, clusterCenters);
    // kmeans(InputArray data, int K, InputOutputArray bestLabels,
    // TermCriteria criteria, int attempts,
    // int flags, OutputArray clusterCenters=noArray())
    std::cout << "\nclusterig end'\n";
    std::cout << "\ncreating words begin:";
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
    std::cout << "\ncreating words end";
    std::cout << "\n countMetric begin";
    countMetric(path, data, K);
    std::cout << "\n countMetric end";
}

// saving data of descriptors, clusters' clusterCenters, words (counted with the usage of TF-IDF metric) and dicts to the user's path
void saveData(std::string& path, std::vector<Image>& data, Mat& clusterCenters, std::map<int, std::vector<double>>& indexStraight,
          std::vector<std::vector<int>>& indexInverted) {
    std::cout << "\nsave begin";
    FileStorage fs_description(path + "descriptors.yml", FileStorage::WRITE);
    FileStorage fs_words(path + "words.yml", FileStorage::WRITE);
    FileStorage fs_clusterCenters(path + "clusterCenters.yml", FileStorage::WRITE);
    FileStorage fs_indexStraight(path + "indexStraight.yml", FileStorage::WRITE);
    FileStorage fs_indexInverted(path + "indexInverted.yml", FileStorage::WRITE);
    size_t i = 0;
    for (auto &elem : data) {
        fs_description << "data_" + std::to_string(i) + "_description" << elem.description;
        fs_words << "data_" + std::to_string(i) + "_word" << elem.word;
        ++i;
    }
        fs_description.release();
        fs_words.release();

        fs_clusterCenters << "clusterCenters" << clusterCenters;
        fs_clusterCenters.release();

        i = 0;
        for (auto &elem : indexStraight) {
            fs_indexStraight << "indexStraight " + std::to_string(i++) << elem.second;
        }
        fs_indexStraight.release();

        for (size_t i = 0; i != indexInverted.size(); ++i) {
            fs_indexInverted << "indexInverted " + std::to_string(i) << indexInverted[i];
        }
        fs_indexInverted.release();
        std::cout << "\n save end";
    }

void calculateInverdedIndex(std::vector<std::vector<int>>& inverted,
                            std::vector<Image>& data) {
    for (size_t i = 0; i != data.size(); ++i) {
        for (size_t j = 0; j != data[i].word.size(); ++j) {
            if (data[i].word[j] > 0) {
                inverted[j].push_back(i);
            }
        }
    }
}

void calculateStraightIndex(std::map<int, std::vector<double>>& straight, std::vector<Image>& data) {
    for (size_t i = 0; i != data.size(); ++i) {
        straight.insert(std::pair<int, std::vector<double>>(i, data[i].word));
    }
}

void calculateIndex(std::map<int, std::vector<double>>& straight, std::vector<std::vector<int>>& inverted,
                       std::vector<Image>& data) {
    calculateStraightIndex(straight, data);
    calculateInverdedIndex(inverted, data);
}

void restoreMetric(std::string& path, Image& element, Mat& clusterCenters) {
    FileStorage fs_readIdf(path + "idf.yml", FileStorage::READ);
    std::vector<int> countNoZeros;
    int quantityOfImages;
    fs_readIdf["dots"] >> countNoZeros;
    fs_readIdf["size"] >> quantityOfImages;
    fs_readIdf.release();

    int totalSum = 0;
    for (auto& elem : element.word) {
        totalSum += elem;
    }
    ++quantityOfImages;
    for (size_t cluster = 0; cluster != element.word.size(); ++cluster) {
        if (element.word[cluster] != 0) {
            countNoZeros[cluster] += 1;
        }
        element.word[cluster] = (element.word[cluster] / totalSum) * log((quantityOfImages) / countNoZeros[cluster]);
    }

    FileStorage fs_writeIdf(path + "idf.yml", FileStorage::WRITE);
    fs_writeIdf << "dots" << countNoZeros;
    fs_writeIdf << "size" << quantityOfImages;
    fs_writeIdf.release();

}

int countEuclidesDistance (int a, int b) {
    return (a - b) * (a - b);
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

void appendSource(std::string& path, Image& newElement) {
    FileStorage fs_read (path + "idf.yml", FileStorage::READ);
    int quantityOfElements = 0;
    fs_read ["size"] >> quantityOfElements;
    fs_read.release();

    FileStorage fs_appendDescriptors (path + "descriptors.yml", FileStorage::APPEND);
    FileStorage fs_appendWords (path + "words.yml", FileStorage::APPEND);

    fs_appendDescriptors << "data_" + std::to_string(quantityOfElements - 1) + "_description" << newElement.description;
    fs_appendWords << "data_" + std::to_string(quantityOfElements - 1) + "_word" << newElement.word;
    fs_appendDescriptors.release();
    fs_appendWords.release();
}

void appendIndex(std::string& path, Image& newElement) {
    FileStorage fs_read (path + "idf.yml", FileStorage::READ);
    int quantityOfElements = 0;
    fs_read ["size"] >> quantityOfElements;
    fs_read.release();

    // adding new element to the existing straight index
    FileStorage fs_indexStraight (path + "indexStraight.yml", FileStorage::APPEND);
    fs_indexStraight << "indexStraight " + std::to_string(quantityOfElements - 1) << newElement.word;
    fs_indexStraight.release();

    Mat clusterCenters;
    FileStorage fs_clusterCenters(path + "clusterCenters.yml", FileStorage::READ);
    fs_clusterCenters["clusterCenters"] >> clusterCenters;
    fs_clusterCenters.release();

    // adding new element to the existing inverted index
    FileStorage fs_readIndexInverted (path + "indexInverted.yml", FileStorage::READ);
    std::vector<std::vector<int>> indexInverted (clusterCenters.rows);
    for (size_t i = 0; i != indexInverted.size(); ++i) {
        fs_readIndexInverted ["indexInverted " + std::to_string(i)] >> indexInverted[i];
    }

    for (size_t j = 0; j != newElement.word.size(); ++j) {
        if (newElement.word[j] > 0) {
            indexInverted[j].push_back(quantityOfElements - 1);
        }
    }

    fs_readIndexInverted.release();

    // writing changes to the existing inverted index
    FileStorage fs_writeIndexInverted (path + "indexInverted.yml", FileStorage::WRITE);
    for (size_t i = 0; i != indexInverted.size(); ++i) {
            fs_writeIndexInverted << "indexInverted " + std::to_string(i) << indexInverted[i];
    }
    fs_writeIndexInverted.release();
}

void restoreAVisualWord(std::string& source, std::string& path, Image& newElement) {
    Mat src = imread(source, CV_LOAD_IMAGE_UNCHANGED);  // reading Image and calculating its descriptor
    resize(src, src, Size(600, 700), 0, 0, INTER_LINEAR);
    Ptr<Feature2D> f2d = SIFT::create(0, 3, 0.18, 5, 1.6);
    Mat descriptor;
    std::vector<KeyPoint> k;
    f2d->detectAndCompute(src, Mat(), k, descriptor);
    newElement.description = descriptor;

    Mat clusterCenters;
    FileStorage fs_clusterCenters(path + "clusterCenters.yml", FileStorage::READ);
    fs_clusterCenters["clusterCenters"] >> clusterCenters;
    fs_clusterCenters.release();

    int K = clusterCenters.rows;  // quantity of clusters
    std::vector<int> labels (descriptor.rows);
    std::vector<std::vector<double>> distances (descriptor.rows, std::vector<double> (clusterCenters.rows, 0));

    for (size_t i = 0; i != descriptor.rows; ++i) {
        for (size_t j = 0; j != clusterCenters.rows; ++j) {
            int currentDistance = 0;
            for (size_t z = 0; z != descriptor.cols; ++z) {  // counting using Euclides metric
//                int temp = static_cast<int>(descriptor.at<uchar>(i, z));
//                int temp2 = static_cast<int>(clusterCenters.at<uchar>(j , z));
                currentDistance += countEuclidesDistance(static_cast<int>(descriptor.at<uchar>(i, z)), static_cast<int>(clusterCenters.at<uchar>(j , z)));
            }
            distances[i][j] = sqrt(currentDistance);
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
    restoreMetric(path, newElement, clusterCenters);
}

int main() {
    std::string storage = "/home/oracle/Project/kinopoisk/";  // folder with pictures
    std::string path = "/home/oracle/Project/data/";  // folder where YAML data is saved
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
                int k = 150;  // approximate number of pictures in the base
                std::vector<Image> data;  // vector of computed pictures
                Mat clusterCenters;
                Mat collection;  // matrix with all descriptors for kmeans
                createABase(storage, k, data, collection);  // creating straight and inverted index dictionaries
                computeVisualWords(path, data, clusterCenters, collection);
                std::map<int, std::vector<double>> indexStraight;
                std::vector<std::vector<int>> indexInverted (clusterCenters.rows);
                calculateIndex(indexStraight, indexInverted, data);
                saveData(path, data, clusterCenters, indexStraight, indexInverted);
                break;
            }
            case 2: {
                std::string source = storage + "632.jpg";  // path to the Image
                Image addingImage;
                restoreAVisualWord(source, path, addingImage);  // calculate tf-idf for the current
                appendSource(path, addingImage);  // append data
                appendIndex(path, addingImage);  // append indexes
                break;
            }
            case 3: {
                

            }
                break;

            case 4: break;
            default:
            std::cout << "Incorrect choice. You should enter 1, 2, 3 or 4" << std::endl;
                break;
        }
    } while (choice != 4);

    return 0;
}