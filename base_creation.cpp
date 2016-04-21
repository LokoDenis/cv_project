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

// Current Task: 1) get to know how FileStorage::APPEND works.
//               2) search for an image.
//               3) manual clustering CHECK AGAIN
//

using namespace cv;
using namespace cv::xfeatures2d;

struct Image {
    Mat description;  // mat of descriptor for each Image
    std::vector<double> word;  // visual words
};

void countMetric(std::string& path, std::vector<Image>& data, int K) {
    std::vector<int> countNoZeros(static_cast<unsigned int>(K), 1);  // vectors with dotes in particular clusters (for IDF)
    std::vector<int> quantityOfDotes(data.size(), 0);  // total sum of dotes in each Image (for TF)

    for (size_t j = 0; j != data.size(); ++j) {
        for (size_t i = 0; i != data[j].word.size(); ++i) {
            quantityOfDotes[j] += data[j].word[i];
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
            data[picture].word[cluster] = data[picture].word[cluster] / quantityOfDotes[picture] * log(data.size() / countNoZeros[cluster]);
        }
    }
}

// walking through the source, reading raw data for further clustering
void createABase(std::string& path, int n, std::vector<Image>& data, Mat& collection) {
    for (size_t i = 0; i != n; ++i) {
        Mat src = imread(path + std::to_string(i + 1) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);  // +1 'cause of the images names
        if (src.empty()) continue;
        Image newElement;
        Ptr<Feature2D> f2d = SIFT::create(0, 3, 0.15, 5, 1.6);
        Mat descriptor;
        std::vector<KeyPoint> keypoints;
        f2d->detectAndCompute(src, Mat(), keypoints, descriptor);
        newElement.description = descriptor;
        data.push_back(newElement);
        collection.push_back(descriptor);
    }
}

// creating visual words for pictures using clustering and TF-IDF metric
void computeVisualWords(std::string& path, std::vector<Image>& data, Mat& clusterCenters, Mat& collection) {
    int K = collection.rows / 80;
    std::vector<int> labels(static_cast<unsigned int>(collection.rows));
    kmeans(collection, K, labels,
           cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 1, KMEANS_PP_CENTERS, clusterCenters);
    // kmeans(InputArray data, int K, InputOutputArray bestLabels,
    // TermCriteria criteria, int attempts,
    // int flags, OutputArray clusterCenters=noArray())
    // creating vectors of clusters' frequencies
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
    countMetric(path, data, K);
}

// saving data of descriptors, clusters' clusterCenters, words (counted with the usage of TF-IDF metric) and dicts to the user's path
void saveData(std::string& path, std::vector<Image>& data, Mat& clusterCenters,
          std::vector<std::vector<int>>& indexInverted) {
    FileStorage fs_words(path + "words.yml", FileStorage::WRITE);
    FileStorage fs_clusterCenters(path + "clusterCenters.yml", FileStorage::WRITE);
    FileStorage fs_indexInverted(path + "indexInverted.yml", FileStorage::WRITE);
    size_t i = 0;
    int descriptorFile = 0;
    FileStorage fs_description(path + "descriptors_" + std::to_string(descriptorFile) + ".yml", FileStorage::WRITE);
    for (auto &elem : data) {
        if (i % 2000 == 0 && i != 0) {
            ++descriptorFile;
            fs_description.open(path + "descriptors_" + std::to_string(descriptorFile) + ".yml", FileStorage::WRITE);
            }
        fs_description << "data_" + std::to_string(i) + "_description" << elem.description;
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
    }

void calculateInvertedIndex(std::vector<std::vector<int>>& inverted,
                            std::vector<Image>& data) {
    for (size_t i = 0; i != data.size(); ++i) {
        for (size_t j = 0; j != data[i].word.size(); ++j) {
            if (data[i].word[j] > 0) {
                inverted[j].push_back(static_cast<int>(i));
            }
        }
    }
}

void restoreMetric(std::string& path, Image& element, bool trigger) {
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
        element.word[cluster] = (element.word[cluster] / totalSum) * log((quantityOfImages) / countNoZeros[cluster]);
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

void appendSource(std::string& path, Image& newElement, int name) {
    int descriptorFile = (name - 1) / 2000;
    FileStorage fs_appendDescriptors (path + "descriptors_" + std::to_string(descriptorFile) + ".yml", FileStorage::APPEND);
    FileStorage fs_appendWords (path + "words.yml", FileStorage::APPEND);

    fs_appendDescriptors << "data_" + std::to_string(name - 1) + "_description" << newElement.description;
    fs_appendWords << "data_" + std::to_string(name - 1) + "_word" << newElement.word;
    fs_appendDescriptors.release();
    fs_appendWords.release();
}

void appendIndex(std::string& path, Image& newElement, int name) {
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

void restoreAVisualWord(std::string& source, std::string& path, Image& newElement) {
    Mat src = imread(source, CV_LOAD_IMAGE_UNCHANGED);  // reading Image and calculating its descriptor
    Ptr<Feature2D> f2d = SIFT::create(0, 3, 0.15, 5, 1.6);
    Mat descriptor;
    std::vector<KeyPoint> k;
    f2d->detectAndCompute(src, Mat(), k, descriptor);
    newElement.description = descriptor;

//    cv::Point2d src_center(src.cols * 0.5, src.rows * 0.5); // defining a center of the source picture
//    cv::Mat rot_mat = cv::getRotationMatrix2D(src_center, 50 , 1);  // creating a rotation matrix
//    warpAffine(src, src, rot_mat, src.size());
//    GaussianBlur(src, src, Size(5,5), 2, 2, BORDER_DEFAULT);
//    medianBlur(src, src, 5);
//
//    namedWindow("initial", CV_WINDOW_AUTOSIZE);
//    imshow("initial", src);
//    waitKey(0);
//    destroyWindow("initial");

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

int searchInBase(std::string& path, Image& newElement) {
    FileStorage fs_readInvertedIndex (path + "indexInverted.yml", FileStorage::READ);
    std::set<int> candidates;  // we will find out during iterating through the base
    std::set<int> best;
    std::set<int> intersect;

    bool firstTime = true;
    for (size_t i = 0; i != newElement.word.size(); ++i) {
        if (newElement.word[i] != 0) {
            std::vector<int> curr;
            fs_readInvertedIndex["indexInverted " + std::to_string(i)] >> curr;
            for (auto& elem : curr) {
                candidates.insert(elem);
            }
            if (firstTime) {
                best = candidates;
                firstTime = false;
            } else {
                std::set_intersection(curr.begin(), curr.end(), best.begin(), best.end(), std::inserter(intersect, intersect.begin()));
                best = intersect;
                intersect.clear();
            }
        }
    }
    fs_readInvertedIndex.release();
    FileStorage fs_readWords (path + "words.yml", FileStorage::READ);
    int minIndex;
    double minValue;
    bool first = true;
    if (best.empty()) {
        best = candidates;
    }
    for (auto elem : best) {
        std::vector<double> word;
        double currentDistance = 0;
        fs_readWords["data_" + std::to_string(elem) + "_word"] >> word;
        for (size_t j = 0; j != word.size(); ++j) {
            currentDistance += countEuclidesDistance(newElement.word[j], word[j]);
        }
        word.empty();
        if (first) {
            minIndex = elem;
            minValue = currentDistance;
            first = false;
        } else {
            if (currentDistance < minValue) {
                minValue = currentDistance;
                minIndex = elem;
            }
        }
    }
    fs_readWords.release();
    return minIndex;
}

void visualize (std::string storage, int number, std::string source) {
    Mat best = imread(storage + std::to_string(number + 1) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);  // +1 'cause of the storage
    Mat img = imread(source, CV_LOAD_IMAGE_UNCHANGED);
    resize(img, img, Size(600, 700), 0, 0, INTER_LINEAR);
    resize(best, best, Size(600, 700), 0, 0, INTER_LINEAR);
    namedWindow("initial", CV_WINDOW_AUTOSIZE);
    imshow("initial", img);
    waitKey(0);
    destroyWindow("initial");
    namedWindow("best", CV_WINDOW_AUTOSIZE);
    imshow("best", best);
    waitKey(0);
    destroyWindow("best");
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
                int k;  // approximate number of pictures in the base
                std::cout << "\nHow many pictures: ";
                std::cin >> k;
                std::vector<Image> data;  // vector of computed pictures
                Mat clusterCenters;
                Mat collection;  // matrix with all descriptors for kmeans
                createABase(storage, k, data, collection);  // creating straight and inverted index dictionaries
                computeVisualWords(path, data, clusterCenters, collection);
                std::vector<std::vector<int>> indexInverted (static_cast<unsigned int>(clusterCenters.rows));
                calculateInvertedIndex(indexInverted, data);
                saveData(path, data, clusterCenters, indexInverted);
                std::cout << "\nDone!";
                break;
            }
            case 2: {
                int num;
                std::cout << "\nPicture's number: ";
                std::cin >> num;
                std::string source = storage + std::to_string(num) + ".jpg";  // path to the Image
                Image addingImage;
                restoreAVisualWord(source, path, addingImage);  // calculate tf-idf for the current
                restoreMetric(path, addingImage, true);
                appendSource(path, addingImage, num);  // append data
                appendIndex(path, addingImage, num);  // append indexes
                std::cout << "\nDone! ";
                break;
            }
            case 3: {
                int num;
                std::cout << "\nPicture's number: ";
                std::cin >> num;
                std::string source = storage + std::to_string(num) + ".jpg";  // path to the image
                Image currentPicture;
                restoreAVisualWord(source, path, currentPicture);
                restoreMetric(path, currentPicture, false);
                int nearest = searchInBase(path, currentPicture);
                visualize(storage, nearest, source);
                break;
            }
            case 4: break;
            default:
            std::cout << "Incorrect choice. You should enter 1, 2, 3 or 4" << std::endl;
                break;
        }
    } while (choice != 4);

    return 0;
}