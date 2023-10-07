#include <iostream>
#include <opencv2\opencv.hpp>
#include <algorithm>

using namespace std;
using namespace cv;

void erosion(const Mat& input_img, Mat& output_img, int n = 1)
{
    output_img = Mat::zeros(input_img.size(), CV_8U);
    for (int i = n; i < input_img.cols - n; i++)
        for (int j = n; j < input_img.rows - n; j++) {
            uchar pix_value = input_img.at<uchar>(j, i);
            float min = 255;
            for (int ii = -n; ii <= n; ii++)
                for (int jj = -n; jj <= n; jj++) {
                    uchar Y = input_img.at<uchar>(j + jj, i + ii);
                    if (Y < min)
                        min = Y;
                }
            output_img.at<uchar>(j, i) = min;
        }
}

void dilation(const Mat& input_img, Mat& output_img, int n = 1)
{
    output_img = Mat::zeros(input_img.size(), CV_8U);
    for (int i = n; i < input_img.cols - n; i++)
        for (int j = n; j < input_img.rows - n; j++) {
            uchar pix_value = input_img.at<uchar>(j, i);
            float max = 0;
            for (int ii = -n; ii <= n; ii++)
                for (int jj = -n; jj <= n; jj++) {
                    uchar Y = input_img.at<uchar>(j + jj, i + ii);
                    if (Y > max)
                        max = Y;
                }
            output_img.at<uchar>(j, i) = max;
        }
}

void opening(const Mat& input_img, Mat& output_img, int n = 1)
{
    Mat buf;
    erosion(input_img, buf, n);
    dilation(buf, output_img, n);
}

void closing(const Mat& input_img, Mat& output_img, int n = 1)
{
    Mat buf;
    dilation(input_img, buf, n);
    erosion(buf, output_img, n);
}

void contour(const Mat& input_img, Mat& output_img, int n = 1)
{
    Mat buf1, buf2;
    dilation(input_img, buf1, n);
    erosion(input_img, buf2, n);
    output_img = buf1 - buf2;

    //erosion(input_img, buf2, n);
    //output_img = input_img - buf2;
}

void MG(const Mat& input_img, Mat& output_img)
{
    output_img = Mat::zeros(input_img.size(), CV_8U);
    Mat bufs[3];
    for (int i = 0; i < 3; i += 1) {
        Mat buf1, buf2;
        dilation(input_img, buf1, i + 1);
        erosion(input_img, buf2, i + 1);
        erosion(buf1 - buf2, bufs[i], i);
    }
    for (int i = 0; i < input_img.cols; i += 1)
        for (int j = 0; j < input_img.rows; j += 1) {
            float result = 0;
            for (int q = 0; q < 3; q += 1) {
                result += bufs[q].at<uchar>(j, i);
            }
            output_img.at<uchar>(j, i) = (uchar)(result / 3);
        }
}

int main()
{
    string name = "";
    int colorType = 0;
    cout << "File name: ";
    cin >> name;

    Mat gray_img = imread(name, colorType);
    if (gray_img.empty()) {
        return 0;
    }

    Mat erosion_img, dilation_img, open_img, close_img, contour_img, MG_img;
    erosion(gray_img, erosion_img);
    dilation(gray_img, dilation_img);
    opening(gray_img, open_img);
    closing(gray_img, close_img);
    contour(gray_img, contour_img);
    MG(gray_img, MG_img);

    Mat bin_img, bin_erosion_img, bin_dilation_img, bin_open_img, bin_close_img, bin_contour_img, bin_MG_img;
    threshold(gray_img, bin_img, 150, 255, THRESH_BINARY); //THRESH_BINARY_INV
    erosion(bin_img, bin_erosion_img);
    dilation(bin_img, bin_dilation_img);
    opening(bin_img, bin_open_img);
    closing(bin_img, bin_close_img);
    contour(bin_img, bin_contour_img);
    MG(bin_img, bin_MG_img);

    namedWindow("gray_img", WINDOW_AUTOSIZE);
    imshow("gray_img", gray_img);
    namedWindow("erosion_img", WINDOW_AUTOSIZE);
    imshow("erosion_img", erosion_img);
    namedWindow("dilation_img", WINDOW_AUTOSIZE);
    imshow("dilation_img", dilation_img);
    namedWindow("open_img", WINDOW_AUTOSIZE);
    imshow("open_img", open_img);
    namedWindow("close_img", WINDOW_AUTOSIZE);
    imshow("close_img", close_img);
    namedWindow("contour_img", WINDOW_AUTOSIZE);
    imshow("contour_img", contour_img);
    namedWindow("MG_img", WINDOW_AUTOSIZE);
    imshow("MG_img", MG_img);

    namedWindow("bin_img", WINDOW_AUTOSIZE);
    imshow("bin_img", bin_img);
    namedWindow("bin_erosion_img", WINDOW_AUTOSIZE);
    imshow("bin_erosion_img", bin_erosion_img);
    namedWindow("bin_dilation_img", WINDOW_AUTOSIZE);
    imshow("bin_dilation_img", bin_dilation_img);
    namedWindow("bin_open_img", WINDOW_AUTOSIZE);
    imshow("bin_open_img", bin_open_img);
    namedWindow("bin_close_img", WINDOW_AUTOSIZE);
    imshow("bin_close_img", bin_close_img);
    namedWindow("bin_contour_img", WINDOW_AUTOSIZE);
    imshow("bin_contour_img", bin_contour_img);
    namedWindow("bin_MG_img", WINDOW_AUTOSIZE);
    imshow("bin_MG_img", bin_MG_img);

    waitKey(0);
    destroyWindow("gray_img");
    destroyWindow("erosion_img");
    destroyWindow("dilation_img");
    destroyWindow("open_img");
    destroyWindow("close_img");
    destroyWindow("contour_img");
    destroyWindow("MG_img");

    destroyWindow("bin_img");
    destroyWindow("bin_erosion_img");
    destroyWindow("bin_dilation_img");
    destroyWindow("bin_open_img");
    destroyWindow("bin_close_img");
    destroyWindow("bin_contour_img");
    destroyWindow("bin_MG_img");
    return 0;
}