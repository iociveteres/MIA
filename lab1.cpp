﻿#include <iostream>
#include <opencv2\opencv.hpp>
#include <algorithm>

using namespace std;
using namespace cv;

void gauss3(const Mat& input_img, Mat& output_img)
{
    output_img = Mat::zeros(input_img.size(), CV_8U);
    float k = 36; // коэффициент нормировки 
    float Fk[3][3] = {{1,  4, 1}, 
                      {4, 16, 4},  
                      {1,  4, 1} }; // маска фильтра 
    for (int i = 1; i < input_img.cols - 1; i++)
        for (int j = 1; j < input_img.rows - 1; j++) {
            uchar pix_value = input_img.at<uchar>(j, i);
            // далее производим свертку 
            float Rez = 0;
            for (int ii = -1; ii <= 1; ii++)
                for (int jj = -1; jj <= 1; jj++) {
                    uchar blurred = input_img.at<uchar>(j + jj, i + ii);
                    Rez += Fk[ii + 1][jj + 1] * blurred;
                }
            uchar blurred = Rez / k; // осуществляем нормировку 
            output_img.at<uchar>(j, i) = blurred;
        }
}

void gauss5(const Mat& input_img, Mat& output_img)
{
    output_img = Mat::zeros(input_img.size(), CV_8U);
    float k = 324; // коэффициент нормировки 
    float Fk[5][5] = { {1,  4,  8,  4,  1}, 
                       {4, 16, 32, 16,  4}, 
                       {8, 32, 64, 32,  8}, 
                       {4, 16, 32, 16,  4}, 
                       {1,  4,  8,  4,  1}}; // маска фильтра 
    for (int i = 2; i < input_img.cols - 2; i++)
        for (int j = 2; j < input_img.rows - 2; j++) {
            uchar pix_value = input_img.at<uchar>(j, i);
            // далее производим свертку 
            float Rez = 0;
            for (int ii = -2; ii <= 2; ii++)
                for (int jj = -2; jj <= 2; jj++) {
                    uchar blurred = input_img.at<uchar>(j + jj,
                        i + ii);
                    Rez += Fk[ii + 2][jj + 2] * blurred;
                }
            uchar blurred = Rez / k; // осуществляем нормировку 
            output_img.at<uchar>(j, i) = blurred;
        }
}

void mosaic3(const Mat& input_img, Mat& output_img)
{
    output_img = Mat::zeros(input_img.size(), CV_8U);
    float k = 9; // коэффициент нормировки 
    float Fk[3][3] = {{1,  1, 1},
                      {1,  1, 1},
                      {1,  1, 1} }; // маска фильтра
    for (int i = 1; i < input_img.cols - 1; i += 3)
        for (int j = 1; j < input_img.rows - 1; j += 3) {
            uchar pix_value = input_img.at<uchar>(j, i);
            // далее производим свертку 
            float Rez = 0;
            for (int ii = -1; ii <= 1; ii++)
                for (int jj = -1; jj <= 1; jj++) {
                    uchar blurred = input_img.at<uchar>(j + jj, i + ii);
                    Rez += Fk[ii + 1][jj + 1] * blurred;
                }
            uchar blurred = Rez / k; // осуществляем нормировку
            for (int ii = -1; ii <= 1; ii++)
                for (int jj = -1; jj <= 1; jj++) {
                    output_img.at<uchar>(j + jj, i + ii) = blurred;
                }
        }
}

void mosaic5(const Mat& input_img, Mat& output_img)
{
    output_img = Mat::zeros(input_img.size(), CV_8U);
    float k = 25; // коэффициент нормировки 
    float Fk[5][5] = {{1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1},
                      {1, 1, 1, 1, 1} }; // маска фильтра
    for (int i = 2; i < input_img.cols - 2; i += 5)
        for (int j = 2; j < input_img.rows - 2; j += 5) {
            uchar pix_value = input_img.at<uchar>(j, i);
            // далее производим свертку 
            float Rez = 0;
            for (int ii = -2; ii <= 2; ii++)
                for (int jj = -2; jj <= 2; jj++) {
                    uchar blurred = input_img.at<uchar>(j + jj, i + ii);
                    Rez += Fk[ii + 2][jj + 2] * blurred;
                }
            uchar blurred = Rez / k; // осуществляем нормировку
            for (int ii = -2; ii <= 2; ii++)
                for (int jj = -2; jj <= 2; jj++) {
                    output_img.at<uchar>(j + jj, i + ii) = blurred;
                }
        }
}

void apertur3(const Mat& input_img, Mat& output_img)
{
    output_img = Mat::zeros(input_img.size(), CV_8U);
    float k = 7; // коэффициент нормировки 
    float Fk[3][3] = {{-1,  -1, -1},
                      {-1,  15, -1},
                      {-1,  -1, -1} }; // маска фильтра
    for (int i = 1; i < input_img.cols - 1; i += 1)
        for (int j = 1; j < input_img.rows - 1; j += 1) {
            uchar pix_value = input_img.at<uchar>(j, i);
            // далее производим свертку 
            float Rez = 0;
            for (int ii = -1; ii <= 1; ii++)
                for (int jj = -1; jj <= 1; jj++) {
                    uchar blurred = input_img.at<uchar>(j + jj, i + ii);
                    Rez += Fk[ii + 1][jj + 1] * blurred;
                }
            uchar blurred = Rez / k; // осуществляем нормировку
            output_img.at<uchar>(j, i) = blurred;
        }
}

void mediana3(const Mat& input_img, Mat& output_img)
{
    output_img = Mat::zeros(input_img.size(), CV_8U);
    vector <uchar> vec;
    vec.resize(9);
    for (int i = 1; i < input_img.cols - 1; i += 1)
        for (int j = 1; j < input_img.rows - 1; j += 1) {
            int index = 0;
            for (int ii = -1; ii <= 1; ii++)
                for (int jj = -1; jj <= 1; jj++) {
                    uchar pix_value = input_img.at<uchar>(j + jj, i + ii);
                    vec[index] = pix_value;
                    index++;
                }
            sort(vec.begin(), vec.end()); 
            output_img.at<uchar>(j, i) = vec[4];
        }
}

void DoG(const Mat& gauss3_img, const Mat& gauss5_img, Mat& output_img)
{
    output_img = Mat::zeros(gauss3_img.size(), CV_8U);
    absdiff(gauss3_img, gauss5_img, output_img);
    output_img *= 5; //для наглядности
}

void sobelVert(const Mat& input_img, Mat& output_img)
{
    output_img = Mat::zeros(input_img.size(), CV_8U);
    float Fk[3][3] = {{-1, 0, 1},
                      {-2, 0, 2},
                      {-1, 0, 1} }; // маска фильтра 
    for (int i = 1; i < input_img.cols - 1; i++)
        for (int j = 1; j < input_img.rows - 1; j++) {
            // далее производим свертку 
            float Rez = 0;
            for (int ii = -1; ii <= 1; ii++)
                for (int jj = -1; jj <= 1; jj++) {
                    uchar pix_value = input_img.at<uchar>(j + jj, i + ii);
                    Rez += Fk[ii + 1][jj + 1] * pix_value;
                }
            output_img.at<uchar>(j, i) = Rez;
        }
}

int main()
{
    string name = "";
    int colorType = 0;
    cout << "File name: ";
    cin >> name;
    //cout << "Color type (0/1): ";
    //cin >> colorType;
    
    Mat input_img = imread(name, colorType);
    if (input_img.empty()) {
        return 0;
    }

    Mat gauss3_img, gauss5_img, mosaic3_img, apertur3_img, mediana3_img, DoG_img, canny_img, sobelVert_img;
    gauss3(input_img, gauss3_img);
    gauss5(input_img, gauss5_img);
    mosaic3(input_img, mosaic3_img);
    apertur3(gauss3_img, apertur3_img);
    mediana3(apertur3_img, mediana3_img);
    DoG(gauss3_img, gauss5_img, DoG_img);
    Canny(input_img, canny_img, 50, 200);
    sobelVert(input_img, sobelVert_img);

    namedWindow("input_img", WINDOW_AUTOSIZE);
    imshow("input_img", input_img);
    namedWindow("gauss3_img", WINDOW_AUTOSIZE);
    imshow("gauss3_img", gauss3_img);
    namedWindow("gauss5_img", WINDOW_AUTOSIZE);
    imshow("gauss5_img", gauss5_img);
    namedWindow("mosaic3_img", WINDOW_AUTOSIZE);
    imshow("mosaic3_img", mosaic3_img);
    namedWindow("apertur3_img", WINDOW_AUTOSIZE);
    imshow("apertur3_img", apertur3_img);
    namedWindow("mediana3_img", WINDOW_AUTOSIZE);
    imshow("mediana3_img", mediana3_img);
    namedWindow("DoG_img", WINDOW_AUTOSIZE);
    imshow("DoG_img", DoG_img);
    namedWindow("canny_img", WINDOW_AUTOSIZE);
    imshow("canny_img", canny_img);
    namedWindow("sobelVert_img", WINDOW_AUTOSIZE);
    imshow("sobelVert_img", sobelVert_img);

    waitKey(0);
    destroyWindow("input_img");
    destroyWindow("gauss3_img");
    destroyWindow("gauss5_img");
    destroyWindow("mosaic3_img");
    destroyWindow("apertur3_img");
    destroyWindow("mediana3_img");
    destroyWindow("DoG_img");
    destroyWindow("canny_img");
    destroyWindow("sobelVert_img");
    return 0;
}