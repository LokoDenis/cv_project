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
#include <set>
#include <stdexcept>

const double delta = 0.002;  // 0.009
const int fileDivide = 1;
std::string path = "/home/oracle/Project/data/";  // folder where YAML data is saved
std::string storage = "/home/oracle/Project/small_kinopoisk/";  // folder with pictures

using namespace cv;
using namespace cv::xfeatures2d;

struct Image {
    Mat description;  // mat of descriptor for each Image
    std::vector<double> word;  // visual words
};

void countMetric(std::vector<Image>& data, int K) {
    std::cout << "countMetric begin\n";
    std::vector<int> countNoZeros(static_cast<unsigned int>(K), 1);  // vectors with Dots in particular clusters (for IDF)
    std::vector<int> quantityOfDots(data.size(), 0);  // total sum of Dots in each Image (for TF)

    for (size_t j = 0; j != data.size(); ++j) {
        for (size_t i = 0; i != data[j].word.size(); ++i) {
            quantityOfDots[j] += data[j].word[i];
            if (data[j].word[i] != 0) {
                ++countNoZeros[i];
            }
        }
    }

    FileStorage fs_idf(path + "idf.yml", FileStorage::WRITE);
    fs_idf << "dots" << countNoZeros;
    fs_idf << "size" << static_cast<int>(data.size());
    fs_idf.release();

    for (size_t picture = 0; picture != data.size(); ++picture) {
        for (size_t cluster = 0; cluster != data[picture].word.size(); ++cluster) {
            double mean = data[picture].word[cluster] / quantityOfDots[picture] * log(data.size() / countNoZeros[cluster]);
            data[picture].word[cluster] = (mean > delta ? mean : 0);
        }
    }
    std::cout << "countMetric end\n";
}

// walking through the source, reading raw data for further clustering
void createABase(int n, std::vector<Image>& data, Mat& collection) {
    int currentNumber = 0;
    FileStorage fs_writeKeys(path + "keys_" + std::to_string(currentNumber) + ".yml.gz", FileStorage::WRITE);
    Ptr<Feature2D> f2d = SIFT::create(0, 3, 0.04, 5, 1.6);

    std::cout << "CreateABase begin\n";
    for (size_t i = 0; i != n; ++i) {

        if (i % 1000 == 0) {
            std::cout << i << std::endl;
        }

        Mat src = imread(storage + std::to_string(i) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);  // +1 'cause of the images names
        if (src.empty()) {
            throw std::invalid_argument("Check " + std::to_string(i) + " picture");
        }
        GaussianBlur(src, src, Size(5,5), 2, 2, BORDER_DEFAULT);
        Image newElement;
        std::vector<KeyPoint> keypoints;
        f2d -> detect(src, keypoints);
        if (keypoints.size() < 10) {
            throw std::invalid_argument("Bad detector for " + std::to_string(i) + " picture");
        } else if (keypoints.size() > 300) {
            std::sort(keypoints.rbegin(), keypoints.rend(), [](KeyPoint a, KeyPoint b){return a.response * a.size < b.response * b.size;});
            keypoints.resize(300);
        };
        Mat descriptor;
        f2d -> compute(src, keypoints, descriptor);
        newElement.description = descriptor;
        data.push_back(newElement);
        collection.push_back(descriptor);
        if (i / fileDivide > currentNumber) {
            currentNumber = i / fileDivide;
            fs_writeKeys.open(path + "keys_" + std::to_string(currentNumber) + ".yml.gz", FileStorage::WRITE);
        }
        fs_writeKeys << "data_" + std::to_string(i) + "_keys" << keypoints;
    }
    fs_writeKeys.release();
    std::cout << "CreateABase end\n";
}

// creating visual words for pictures using clustering and TF-IDF metric
void computeVisualWords(std::vector<Image>& data, Mat& clusterCenters, Mat& collection) {
    std::cout << "computeVisualWords begin\n";
    int K = 2900;  // number of clusters
    std::vector<int> labels(static_cast<unsigned int>(collection.rows));
    std::cout << "  kmeans begin\n";
    kmeans(collection, K, labels,
           cvTermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 1, KMEANS_PP_CENTERS, clusterCenters);
    // kmeans(InputArray data, int K, InputOutputArray bestLabels,
    // TermCriteria criteria, int attempts,
    // int flags, OutputArray clusterCenters=noArray())
    // creating vectors of clusters' frequencies
    std::cout << "  kmeans end\n";
    unsigned int i = 0;  // creating vectors of clusters' frequencies
    while (i < labels.size()) {
        for (auto &elem : data) {
            std::vector<double> visualWord(static_cast<unsigned int>(K), 0);
            for (size_t j = 0; j != elem.description.rows; ++j) {
                ++visualWord[labels[i]];
                ++i;
            }
            elem.word = visualWord;
        }
    }
    countMetric(data, K);
}

// saving data of descriptors, clusters' clusterCenters, words (counted with the usage of TF-IDF metric) and dicts to the user's path
void saveData(std::vector<Image>& data, Mat& clusterCenters,
              std::vector<std::vector<int>>& indexInverted) {
    std::cout << "saveData begin";
    FileStorage fs_clusterCenters(path + "clusterCenters.yml", FileStorage::WRITE);
    FileStorage fs_indexInverted(path + "indexInverted.yml", FileStorage::WRITE);
    int i = 0;
    int fileNumber = 0;
    FileStorage fs_words(path + "words_" + std::to_string(fileNumber) + ".yml.gz", FileStorage::WRITE);
    FileStorage fs_description(path + "descriptors_" + std::to_string(fileNumber) + ".yml.gz", FileStorage::WRITE);
    for (auto &elem : data) {
        if (i % fileDivide == 0 && i != 0) {
            ++fileNumber;
            fs_words.open(path + "words_" + std::to_string(fileNumber) + ".yml.gz", FileStorage::WRITE);
            fs_description.open(path + "descriptors_" + std::to_string(fileNumber) + ".yml.gz", FileStorage::WRITE);
        }
        fs_description << "data_" + std::to_string(i) + "_descriptors" << elem.description;
        fs_words << "data_" + std::to_string(i) + "_word" << elem.word;
        ++i;
    }
    fs_description.release();
    fs_words.release();

    fs_clusterCenters << "clusterCenters" << clusterCenters;
    fs_clusterCenters.release();

    for (size_t j = 0; j != indexInverted.size(); ++j) {
        fs_indexInverted << "indexInverted " + std::to_string(j) << indexInverted[j];
    }
    fs_indexInverted.release();
    std::cout << "saveData end\n";
}

void calculateInvertedIndex(std::vector<std::vector<int>>& inverted,
                            std::vector<Image>& data) {
    std::cout << "calculateInvertedIndex begin\n";
    for (size_t i = 0; i != data.size(); ++i) {
        for (size_t j = 0; j != data[i].word.size(); ++j) {
            if (data[i].word[j] > 0) {
                inverted[j].push_back(static_cast<int>(i));
            }
        }
    }
    std::cout << "calculateInvertedIndex end\n";
}

void restoreMetric(Image& element, bool trigger) {
    FileStorage fs_readIdf(path + "idf.yml", FileStorage::READ);
    std::vector<int> countNoZeros;
    int quantityOfImages;
    fs_readIdf["dots"] >> countNoZeros;
    fs_readIdf["size"] >> quantityOfImages;
    fs_readIdf.release();

    int totalSum = element.description.rows;
    ++quantityOfImages;
    for (size_t cluster = 0; cluster != element.word.size(); ++cluster) {
        if (element.word[cluster] != 0) {
            countNoZeros[cluster] += 1;
        }
        double mean = (element.word[cluster] / totalSum) * log((quantityOfImages) / countNoZeros[cluster]);
        element.word[cluster] = (mean > delta ? mean : 0);
    }

    if (trigger) {
        FileStorage fs_writeIdf(path + "idf.yml", FileStorage::WRITE);
        fs_writeIdf << "dots" << countNoZeros;
        fs_writeIdf << "size" << quantityOfImages;
        fs_writeIdf.release();
    }
}

double countEuclidesDistance (double a, double b) {
    return (a - b) * (a - b);
}

void appendSource(Image& newElement, int name) {
    int fileNumber = (name - 1) / fileDivide;
    FileStorage fs_appendDescriptors (path + "descriptors_" + std::to_string(fileNumber) + ".yml.gz", FileStorage::APPEND);
    FileStorage fs_appendWords (path + "words_" + std::to_string(fileNumber) + ".yml.gz", FileStorage::APPEND);
    fs_appendDescriptors << "data_" + std::to_string(name - 1) + "_descriptors" << newElement.description;
    fs_appendWords << "data_" + std::to_string(name - 1) + "_word" << newElement.word;
    fs_appendDescriptors.release();
    fs_appendWords.release();

    FileStorage fs_appendKeys (path + "keys_" + std::to_string(fileNumber) + ".yml.gz", FileStorage::APPEND);
    Mat img = imread(storage + std::to_string(name) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);
    std::vector<KeyPoint> keys;
    Ptr<Feature2D> f2d = SIFT::create(0, 3, 0.07, 5, 1.6);
    f2d -> detect(img, keys);
    if (keys.size() > 350) {
        std::sort(keys.rbegin(), keys.rend(), [](KeyPoint a, KeyPoint b){return a.response * a.size < b.response * b.size;});
        keys.resize(350);
    }
    fs_appendKeys << "data_" + std::to_string(name - 1) + "_word" << keys;
    fs_appendKeys.release();
}

void appendIndex(Image& newElement, int name) {
    // Reading the existing inverted index
    FileStorage fs_readIndexInverted (path + "indexInverted.yml", FileStorage::READ);
    FileStorage fs_writeIndexInverted (path + "indexInverted.yml", FileStorage::WRITE);

    std::vector<int> curr;  // the most effective way I can
    for (size_t i = 0; i != newElement.word.size(); ++ i) {
        fs_readIndexInverted ["indexInverted " + std::to_string(i)] >> curr;
        if (newElement.word[i] > 0) {
            curr.push_back(name - 1);
        }
        fs_writeIndexInverted << "indexInverted " + std::to_string(i) << curr;
    }
    fs_readIndexInverted.release();
    fs_writeIndexInverted.release();
}

void restoreAVisualWord(std::string& source, std::string& path, Image& newElement, std::vector<KeyPoint>& k) {
    Mat src = imread(source, CV_LOAD_IMAGE_UNCHANGED);  // reading Image and calculating its descriptor
    resize(src, src, Size(480, 640), 0, 0, INTER_LINEAR);
    GaussianBlur(src, src, Size(3,3), 2, 2, BORDER_DEFAULT);
    Ptr<Feature2D> f2d = SIFT::create(0, 3, 0.07, 5, 1.6);
    f2d->detect(src, k);
    if (k.size() > 350) {
        std::sort(k.rbegin(), k.rend(), [](KeyPoint a, KeyPoint b){return a.response * a.size < b.response * b.size;});
        k.resize(350);
    }
    Mat descriptor;
    f2d -> compute (src, k, descriptor);
    newElement.description = descriptor;

    Mat clusterCenters;
    FileStorage fs_clusterCenters(path + "clusterCenters.yml", FileStorage::READ);
    fs_clusterCenters["clusterCenters"] >> clusterCenters;
    fs_clusterCenters.release();

    int K = clusterCenters.rows;  // quantity of clusters
    std::vector<int> labels(static_cast<unsigned int>(descriptor.rows));
    std::vector<std::vector<double>> distances(static_cast<unsigned int>(descriptor.rows),
                                               std::vector<double>(static_cast<unsigned int>(clusterCenters.rows), 0));

    double currentDistance = 0;
    for (int j = 0; j != descriptor.rows; ++j) {
        float *currentDot = descriptor.ptr<float>(j);
        for (int i = 0; i != clusterCenters.rows; ++i) {
            float *currentRow = clusterCenters.ptr<float>(i);
            currentDistance = normL2Sqr(currentDot, currentRow, descriptor.cols);
            distances[j][i] = currentDistance;
        }
    }

    //  calculation the cluster number for each descriptor
    for (size_t i = 0; i != labels.size(); ++i) {
        labels[i] = static_cast<int>(std::distance(distances[i].begin(),
                                                   std::min_element(distances[i].begin(), distances[i].end())));  // min index
    }

    unsigned int i = 0;  // creating vectors of clusters' frequencies
    while (i < labels.size()) {
        std::vector<double> visualWord(static_cast<unsigned int>(K), 0);
        for (size_t j = 0; j != descriptor.rows; ++j) {
            ++visualWord[labels[i]];
            ++i;
        }
        newElement.word = visualWord;
    }
}

int main() {
    int choice;
    do {
        std::cout << "\nWhat do you want to do?" << std::endl;
        std::cout << "1) Create a base" << std::endl;
        std::cout << "2) Add a picture" << std::endl;
        std::cout << "3) Exit" << std::endl;
        std::cin >> choice;
        switch (choice) {
            case 1: {
                int k;  // approximate number of pictures in the base
                std::cout << "\nHow many pictures: ";
                std::cin >> k;
                std::vector<Image> data;  // vector of computed pictures
                Mat clusterCenters;
                Mat collection;  // matrix with all descriptors for kmeans
                createABase(k, data, collection);  // creating straight and inverted index dictionaries
                computeVisualWords(data, clusterCenters, collection);
                std::vector<std::vector<int>> indexInverted (static_cast<unsigned int>(clusterCenters.rows));
                calculateInvertedIndex(indexInverted, data);
                saveData(data, clusterCenters, indexInverted);
                std::cout << "\nDone!";
                break;
            }
            case 2: {
                int num;
                std::cout << "\nPicture's number: ";
                std::cin >> num;
                std::string source = storage + std::to_string(num) + ".jpg";  // path to the Image
                Image addingImage;
                std::vector<KeyPoint> k;
                restoreAVisualWord(source, path, addingImage, k);  // calculate tf-idf for the current
                restoreMetric(addingImage, true);
                appendSource(addingImage, num);  // append data
                appendIndex(addingImage, num);  // append indexes
                std::cout << "\nDone! ";
                break;
            }
            case 3: break;
            default:
                std::cout << "Incorrect choice. You should enter 1, 2, or 3" << std::endl;
                break;
        }
    } while (choice != 3);

    return 0;
}

