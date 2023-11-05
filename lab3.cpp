#include <iostream>
#include <opencv2\opencv.hpp>
#include <algorithm>

using namespace std;
using namespace cv;

int main()
{
    int colorType = 1;

    Mat img1 = imread("i3.jpg", colorType);
    if (img1.empty()) {
        return 0;
    }
    Mat img2 = imread("i4.jpg", colorType);
    if (img2.empty()) {
        return 0;
    }

    Ptr<ORB> detector = ORB::create();
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptor1, descriptor2;

    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);
    detector->compute(img1, keypoints1, descriptor1);
    detector->compute(img2, keypoints2, descriptor2);

    Mat keypoints1draw, keypoints2draw;
    drawKeypoints(img1, keypoints1, keypoints1draw);
    drawKeypoints(img2, keypoints2, keypoints2draw);
    imshow("keypoints1draw", keypoints1draw);
    imshow("keypoints2draw", keypoints2draw);

    //Сопоставление дескрипторов точек с помощью BFMatcher 
    BFMatcher matcher(NORM_HAMMING);
    std::vector<std::vector<DMatch>> matches;
    matcher.knnMatch(descriptor1, descriptor2, matches, 2);

    std::vector<KeyPoint> matched1, matched2;
    std::vector<DMatch> good_matches;

    float match_ratio = 0.5f;

    for (size_t i = 0; i < matches.size(); i++) {
        DMatch first = matches[i][0];
        float dist1 = matches[i][0].distance;
        float dist2 = matches[i][1].distance;

        if (dist1 < match_ratio * dist2) {
            int new_i = static_cast<int>(matched1.size());
            matched1.push_back(keypoints1[first.queryIdx]);
            matched2.push_back(keypoints2[first.trainIdx]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }

    Mat dMatches;
    drawMatches(img1, matched1, img2, matched2, good_matches, dMatches);
    imshow("Matches", dMatches);


    waitKey();
    return 0;
}