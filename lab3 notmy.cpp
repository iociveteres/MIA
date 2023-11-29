
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

int mainnot()
{
    //Загрузка изображений с диска
    Mat img1 = imread("img3.jpeg");
    Mat img2 = imread("img4.jpeg");
    
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
    drawKeypoints(img1, keypoints1, keypoints1draw, Scalar(0, 255, 255));
    drawKeypoints(img2, keypoints2, keypoints2draw, Scalar(0, 255, 255));

    Mat drawDisp1, drawDisp2;

    //Сопоставление дескрипторов точек с помощью BFMatcher
    BFMatcher matcher(NORM_HAMMING);
    std::vector<std::vector<DMatch>> matches;
    matcher.knnMatch(descriptor1, descriptor2, matches, 2);
    std::vector<KeyPoint> matched1, matched2;
    std::vector<DMatch> good_matches;
    float match_ratio = 0.5f;
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for (size_t i = 0; i < matches.size(); i++) {
        DMatch first = matches[i][0];


        float dist1 = matches[i][0].distance;
        float dist2 = matches[i][1].distance;
        if (dist1 < match_ratio * dist2) {
            int new_i = static_cast<int>(matched1.size());
            matched1.push_back(keypoints1[first.queryIdx]);
            matched2.push_back(keypoints2[first.trainIdx]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
            obj.push_back(keypoints1[first.queryIdx].pt);
            scene.push_back(keypoints2[first.trainIdx].pt);
        }
    }
    Mat alignedImg1, alignedImg2;
    Mat H = findHomography(obj, scene);

    Mat Hi = H.inv();

    Size sz = img1.size();
    warpPerspective(img1, alignedImg1, Hi, sz);
    warpPerspective(img2, alignedImg2, H, sz);
    sz.width *= 2;

    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
    Mat pano = Mat::zeros(sz, alignedImg1.type());
    vector<Mat> imgs;
    imgs.push_back(alignedImg1);
    imgs.push_back(alignedImg2);
    cout << "A image: " << alignedImg1.size() << "\n";
    cout << "B image: " << alignedImg2.size() << "\n";
    cout << "Size array: " << imgs.size() << "\n";
    Stitcher::Status status = stitcher->stitch(imgs, pano);

    cout << "Status: " << status << "\n";
    cout << "P image: " << pano.size() << "\n";
    Mat dMatches;
    drawMatches(img1, matched1, img2, matched2, good_matches, dMatches, Scalar(0, 255, 255), Scalar(0, 0, 255));

    imwrite("output/keypoints.png", dMatches);

    resize(keypoints1draw, drawDisp1, Size(1008, 1344));
    resize(keypoints2draw, drawDisp2, Size(1008, 1344));
    imwrite("output/aligned1.png", alignedImg1);
    imwrite("output/aligned2.png", alignedImg2);
    imwrite("output/panorama.png", pano);
    return 0;
}