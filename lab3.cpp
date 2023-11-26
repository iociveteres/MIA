#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/core/cvstd.hpp"
#include <iostream>
#include "opencv2/core.hpp"
//#include "opencv2/xfeatures2d.hpp"
#include <math.h>
#include <iostream>
using namespace cv;
using namespace std;

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

    //Инициализация детектора, подготовка контейнеров для характерных точек и их дескрипторов
    Ptr<ORB> detector = ORB::create();
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptor1, descriptor2;

    //Поиск характерных точек на сшиваемых изображениях и вычисление их дескрипторов
    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);
    detector->compute(img1, keypoints1, descriptor1);
    detector->compute(img2, keypoints2, descriptor2);

    //Отрисовка найденных характерных точек
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

    //--------------------------------------------------------------------------------------------------------------------------

    // Оценка матрицы гомографии
    std::vector<Point2f> matched1_pts, matched2_pts;
    for (int i = 0; i < good_matches.size(); i++) {
        matched1_pts.push_back(matched1[good_matches[i].queryIdx].pt);
        matched2_pts.push_back(matched2[good_matches[i].trainIdx].pt);
    }

    Mat H = findHomography(matched1_pts, matched2_pts, RANSAC);

    Mat alignedImg2;
    warpPerspective(img2, alignedImg2, H, img1.size());

    Size sz = img1.size();
    sz.width += alignedImg2.size().width;
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
    Mat pano1 = Mat::zeros(sz, alignedImg2.type());
    Mat pano2 = Mat::zeros(sz, alignedImg2.type());
    std::vector<Mat> imgs;
    imgs.push_back(img1);
    imgs.push_back(alignedImg2);
    stitcher->stitch(imgs, pano1);
    stitcher->composePanorama(imgs, pano2);

    imshow("alignedImg2", alignedImg2);
    imshow("stitch PANORAMA", pano1);
    imshow("composePanorama PANORAMA", pano2);





    //Mat H = findHomography(Mat(matched1_pts), Mat(matched2_pts), RANSAC);

    //// Трансформация второго изображения
    //Mat transformed_img2, img2_2;
    //resize(img2, img2_2, Size(img2.cols * 2, img2.rows * 2)); // уменьшение изображения в два раза
    //warpPerspective(img2_2, transformed_img2, homography, Size(img2.cols * 2, img2.rows * 2), INTER_CUBIC);

    ////Point a cv::Mat header at it (no allocation is done)
    //Mat final(Size(img2.cols + img1.cols, img2.rows * 2), CV_8UC3);
    //resize(transformed_img2, transformed_img2, Size(transformed_img2.cols / 2, transformed_img2.rows / 2)); // уменьшение изображения в два раза

    //Mat roi1(final, Rect(0, 0, img1.cols, img1.rows));
    //Mat roi2(final, Rect(img1.cols, 0, transformed_img2.cols, transformed_img2.rows));
    //transformed_img2.copyTo(roi2);
    //img1.copyTo(roi1);
    //imshow("final", final);

    //imshow("Transformed Image 2", transformed_img2);

    waitKey();
    return 0;
}