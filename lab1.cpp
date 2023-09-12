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
    float k = 228; // коэффициент нормировки 
    float Fk[5][5] = { {1,  4,  8,  4,  1}, 
                       {4,  8, 16,  8,  4}, 
                       {8, 16, 64, 16,  8}, 
                       {4,  8, 16,  8,  4}, 
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

int main()
{
    string name = "";
    int colorType = 0;
    cout << "File name: ";
    cin >> name;
    cout << "Color type (0/1): ";
    cin >> colorType;
    
    Mat input_img = imread(name, colorType);
    if (input_img.empty()) {
        return 0;
    }

    Mat output1_img, output2_img;
    gauss3(input_img, output1_img);
    gauss5(input_img, output2_img);

    namedWindow("input_img", WINDOW_AUTOSIZE);
    imshow("input_img", input_img);
    namedWindow("output1_img", WINDOW_AUTOSIZE);
    imshow("output1_img", output1_img);
    namedWindow("output2_img", WINDOW_AUTOSIZE);
    imshow("output2_img", output2_img);

    waitKey(0);
    destroyWindow("input_img");
    destroyWindow("output1_img");
    destroyWindow("output2_img");
    return 0;
}