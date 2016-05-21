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

// add EuclidesCount in the 1st half of the searchInBase function. Create a structure, which consists of a number
// and a Euclides distance and after that just use std::sort with lambda function

const double delta = 0.001;
const int fileDivide = 1;
std::string path = "/home/oracle/Project/data/";  // folder where YAML data is saved
std::string storage = "/home/oracle/Project/small_kinopoisk/";  // folder with pictures

using namespace cv;
using namespace cv::xfeatures2d;

struct Candidate {
    int number;
    double distance;
};

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
    FileStorage fs_writeKeys(path + "keys_" + std::to_string(currentNumber) + ".yml", FileStorage::WRITE);
    std::cout << "CreateABase begin\n";
    for (size_t i = 0; i != n; ++i) {
        Mat src = imread(storage + std::to_string(i + 1) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);  // +1 'cause of the images names
        if (src.empty()) {
            throw std::invalid_argument("Check " + std::to_string(i + 1) + " picture");
        }
        GaussianBlur(src, src, Size(5,5), 2, 2, BORDER_DEFAULT);
        Image newElement;
        Ptr<Feature2D> f2d = SIFT::create(0, 3, 0.07, 5, 1.6);
        std::vector<KeyPoint> keypoints;
        f2d->detect(src, keypoints);
        if (keypoints.size() < 10) {
            throw std::invalid_argument("Bad detector for " + std::to_string(i + 1) + " picture");
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
            currentNumber = i/fileDivide;
            fs_writeKeys.open(path + "keys_" + std::to_string(currentNumber) + ".yml", FileStorage::WRITE);
        }
        fs_writeKeys << "data_" + std::to_string(i) + "_keys" << keypoints;
    }
    fs_writeKeys.release();
    std::cout << "CreateABase end\n";
}

// creating visual words for pictures using clustering and TF-IDF metric
void computeVisualWords(std::vector<Image>& data, Mat& clusterCenters, Mat& collection) {
    std::cout << "computeVisualWords begin\n";
    int K = 1100;  // number of clusters
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
    FileStorage fs_words(path + "words_" + std::to_string(fileNumber) + ".yml", FileStorage::WRITE);
    FileStorage fs_description(path + "descriptors_" + std::to_string(fileNumber) + ".yml", FileStorage::WRITE);
    for (auto &elem : data) {
        if (i % fileDivide == 0 && i != 0) {
            ++fileNumber;
            fs_words.open(path + "words_" + std::to_string(fileNumber) + ".yml", FileStorage::WRITE);
            fs_description.open(path + "descriptors_" + std::to_string(fileNumber) + ".yml", FileStorage::WRITE);
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
    FileStorage fs_appendDescriptors (path + "descriptors_" + std::to_string(fileNumber) + ".yml", FileStorage::APPEND);
    FileStorage fs_appendWords (path + "words_" + std::to_string(fileNumber) + ".yml", FileStorage::APPEND);
    fs_appendDescriptors << "data_" + std::to_string(name - 1) + "_descriptors" << newElement.description;
    fs_appendWords << "data_" + std::to_string(name - 1) + "_word" << newElement.word;
    fs_appendDescriptors.release();
    fs_appendWords.release();

    FileStorage fs_appendKeys (path + "keys_" + std::to_string(fileNumber) + ".yml", FileStorage::APPEND);
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

    // simulate the noise
//    cv::Point2d src_center(src.cols * 0.5, src.rows * 0.5); // defining a center of the source picture
//    cv::Mat rot_mat = cv::getRotationMatrix2D(src_center, 50 , 1);  // creating a rotation matrix
//    warpAffine(src, src, rot_mat, src.size());
//    GaussianBlur(src, src, Size(5,5), 2, 2, BORDER_DEFAULT);
//    medianBlur(src, src, 3);

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

std::vector<Candidate> searchInBase(Image& newElement) {  //saving top 30 elements for further work
    FileStorage fs_readInvertedIndex (path + "indexInverted.yml", FileStorage::READ);
    std::set<int> best;

    for (size_t i = 0; i != newElement.word.size(); ++i) {
        if (newElement.word[i] != 0) {
            std::vector<int> curr;
            fs_readInvertedIndex["indexInverted " + std::to_string(i)] >> curr;
                best.insert(curr.begin(), curr.end());
            }
        }

    fs_readInvertedIndex.release();

    int currentNumber = *best.begin() / fileDivide;
    FileStorage fs_readWord(path + "words_" + std::to_string(currentNumber) + ".yml", FileStorage::READ);
    std::vector<Candidate> matches;
    for (auto elem : best) {
        if (elem / fileDivide != currentNumber) {
            currentNumber = elem / fileDivide;
            fs_readWord.open(path + "words_" + std::to_string(currentNumber) + ".yml", FileStorage::READ);
        }
        std::vector<double> word;
        fs_readWord["data_" + std::to_string(elem) + "_word"] >> word;
        double difference = 0;
        for (size_t i = 0; i != newElement.word.size(); ++i) {
            difference += countEuclidesDistance(newElement.word[i], word[i]);
        }
        Candidate add;
        add.number = elem;
        add.distance = difference;
        matches.push_back(add);
    }

    fs_readWord.release();
    std::sort(matches.begin(), matches.end(), [](Candidate a, Candidate b) {return a.distance < b.distance;});

//    int num = -1;
//    for (int i = 0; i != matches.size(); ++i) {
//        if (matches[i].number == 9) {
//            num = i;
//            break;
//        }
//    }

    if (matches.size() > 100) {  // can't figure out, what size should it be
        matches.resize(100);
    }

    return matches;
}

int findMatch (std::string address, std::vector<Candidate>& data, Image& scenElement, std::vector<KeyPoint>& scene_keys) {

    int bestMatch = data[0].number;  //the best according to the bag-of-words algo
    // std::sort(data.begin(), data.end(),[](Candidate a, Candidate b){return a.number < b.number;});
    double maxPercent = 0;

    int currentNumber = data[0].number / fileDivide;
    FileStorage fs_readDescriptor(path + "descriptors_" + std::to_string(currentNumber) + ".yml", FileStorage::READ);
    FileStorage fs_readKeys(path + "keys_" + std::to_string(currentNumber) + ".yml", FileStorage::READ);

    for (auto elem : data) {
        int inlierCount = 0;
        std::vector<KeyPoint> img_keys;
        Mat currentDescriptor;
//        if (elem.number == 9) {
//            std::cout << "HERE\n";
//        }
        if (elem.number / fileDivide != currentNumber) {
            currentNumber = elem.number / fileDivide;
            fs_readDescriptor.open(path + "descriptors_" + std::to_string(currentNumber) + ".yml", FileStorage::READ);
            fs_readKeys.open(path + "keys_" + std::to_string(currentNumber) + ".yml", FileStorage::READ);
        }
        fs_readDescriptor["data_" + std::to_string(elem.number) + "_descriptors"] >> currentDescriptor;
        fs_readKeys["data_" + std::to_string(elem.number) + "_keys"] >> img_keys;

        FlannBasedMatcher matcher;
        std::vector<DMatch> matches;
        matcher.match(currentDescriptor, scenElement.description, matches);  // current -> object from our base
                                                                             // scene -> photo we have
        double min_dist = 100;  // cutting "bad" matches due to the distance
        double max_dist = 0;
        for (size_t i = 0; i != currentDescriptor.rows; ++i) {
            if (matches[i].distance < min_dist) {
                min_dist = matches[i].distance;
            }
            if (matches[i].distance > max_dist) {
                max_dist = matches[i].distance;
            }
        }

        std::vector<DMatch> betterMatches;
        for (size_t i = 0; i != currentDescriptor.rows; ++i) {
            if (matches[i].distance <= 3 * min_dist) {
                betterMatches.push_back(matches[i]);
            }
        }

        std::vector<Point2f> img_points, scene_points;  // best matched points
        //item.trainIdx: This attribute gives us the index of the descriptor in the list of train descriptors
        // (in our case, it’s the list of descriptors in the scene).
        //item.queryIdx: This attribute gives us the index of the descriptor in the list of query descriptors
        // (in our case, it’s the list of descriptors in the base).

        for (size_t i = 0; i != betterMatches.size(); ++i) {
            img_points.push_back(img_keys[betterMatches[i].queryIdx].pt);
            scene_points.push_back(scene_keys[betterMatches[i].trainIdx].pt);
        }

        Mat mask;  // from that we can get the info of inliers
        Mat H = findHomography(img_points, scene_points, CV_RANSAC, 7, mask);  // another function

        for (size_t i = 0; i != mask.rows; ++i) {  // if the value is 0, that means the dot is an outlier
            if (static_cast<int>(mask.at<uchar>(i))) {
                ++inlierCount;
            }
        }

        double percent = static_cast<double>(inlierCount)/betterMatches.size();

        if (percent > maxPercent) {
            maxPercent = percent;
            bestMatch = elem.number;
        }
    }
    fs_readKeys.release();
    fs_readDescriptor.release();
    return bestMatch;
}

void visualize (int number, std::string source) {
    Mat best = imread(storage + std::to_string(number + 1) + ".jpg", CV_LOAD_IMAGE_UNCHANGED);  // +1 'cause of the storage
    Mat img = imread(source, CV_LOAD_IMAGE_UNCHANGED);
    resize(img, img, Size(480, 640), 0, 0, INTER_LINEAR);
    namedWindow("initial", CV_WINDOW_AUTOSIZE);
    imshow("initial", img);
    namedWindow("best", CV_WINDOW_AUTOSIZE);
    imshow("best", best);
    waitKey(2500);
    destroyWindow("initial");
    destroyWindow("best");
    std::cout << number + 1 << std::endl;
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
            case 3: {
                int num;
                std::string dir = "/home/oracle/Project/test/";
                std::cout << "\nReading pictures from " << dir;
                std::cout << "\nEnter 0 to exit.";
                while (true) {
                    std::cout << "\nPicture's number: ";
                    std::cin >> num;
                    if (num == 0) {
                        break;
                    }
                    std::string source = dir + std::to_string(num) + ".jpg";  // path to the image
                    Image currentPicture;
                    std::vector<KeyPoint> k;
                    restoreAVisualWord(source, path, currentPicture, k);
                    restoreMetric(currentPicture, false);
                    std::vector<Candidate> top = searchInBase(currentPicture);
                    int nearest = findMatch(source, top, currentPicture, k);
                    visualize(nearest, source);
                }
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