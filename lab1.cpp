#include <iostream>
#include <opencv2\opencv.hpp>

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
    float k = 249; // коэффициент нормировки 
    float Fk[5][5] = { {1,  0,  0,  0,  1}, 
                       {0, 46,  0, 46,  0}, 
                       {0,  0, 61,  0,  0}, 
                       {0, 46,  0, 46,  0}, 
                       {1,  0,  0,  0,  1}}; // маска фильтра 
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

    Mat gauss3_img, gauss5_img, mosaic3_img, apertur3_img;
    gauss3(input_img, gauss3_img);
    gauss5(input_img, gauss5_img);
    mosaic3(input_img, mosaic3_img);
    apertur3(gauss3_img, apertur3_img);

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

    waitKey(0);
    destroyWindow("input_img");
    destroyWindow("gauss3_img");
    destroyWindow("gauss5_img");
    destroyWindow("mosaic3_img");
    destroyWindow("apertur3_img");
    return 0;
}