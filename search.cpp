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
std::string path = "/home/oracle/Project/data/";  // path to lib files

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

double countEuclidesDistance (double a, double b) {
    return (a - b) * (a - b);
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

void restoreMetric(Image& element) {
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
    FileStorage fs_readWord(path + "words_" + std::to_string(currentNumber) + ".yml.gz", FileStorage::READ);
    std::vector<Candidate> matches;
    for (auto elem : best) {
        if (elem / fileDivide != currentNumber) {
            currentNumber = elem / fileDivide;
            fs_readWord.open(path + "words_" + std::to_string(currentNumber) + ".yml.gz", FileStorage::READ);
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

    if (matches.size() > 40) {
        matches.resize(40);
    }

    return matches;
}

int findMatch (std::string address, std::vector<Candidate>& data, Image& scenElement, std::vector<KeyPoint>& scene_keys) {

    int bestMatch = data[0].number;  //the best according to the bag-of-words algo
    // std::sort(data.begin(), data.end(),[](Candidate a, Candidate b){return a.number < b.number;});
    double maxPercent = 0;

    int currentNumber = data[0].number / fileDivide;
    FileStorage fs_readDescriptor(path + "descriptors_" + std::to_string(currentNumber) + ".yml.gz", FileStorage::READ);
    FileStorage fs_readKeys(path + "keys_" + std::to_string(currentNumber) + ".yml.gz", FileStorage::READ);

    for (auto elem : data) {
        int inlierCount = 0;
        std::vector<KeyPoint> img_keys;
        Mat currentDescriptor;

        if (elem.number / fileDivide != currentNumber) {
            currentNumber = elem.number / fileDivide;
            fs_readDescriptor.open(path + "descriptors_" + std::to_string(currentNumber) + ".yml.gz", FileStorage::READ);
            fs_readKeys.open(path + "keys_" + std::to_string(currentNumber) + ".yml.gz", FileStorage::READ);
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
            if (matches[i].distance <= 4 * min_dist) {
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

        double percent = static_cast<double>(inlierCount) / betterMatches.size();

        if (percent > maxPercent) {
            maxPercent = percent;
            bestMatch = elem.number;
        }
    }
    fs_readKeys.release();
    fs_readDescriptor.release();

    FileStorage fs_map(path + "map.yml", FileStorage::READ);
    int realNumber;
    fs_map["image_" + std::to_string(bestMatch)] >> realNumber;
    fs_map.release();
    return realNumber;
}

int main(int argc, char* argv[]) {
    std::string source(argv[1]);
    std::string dir = "/home/oracle/Project/bot/";
    Image currentPicture;
    std::vector<KeyPoint> k;
    restoreAVisualWord(source, path, currentPicture, k);
    restoreMetric(currentPicture);
    std::vector<Candidate> top = searchInBase(currentPicture);
    int nearest = findMatch(source, top, currentPicture, k);
    std::cout << "http://www.kinopoisk.ru/film/" + std::to_string(nearest) << "\n";
}
