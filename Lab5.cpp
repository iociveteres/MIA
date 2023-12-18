#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


Mat getHist(const Mat& hist_array, int scaleX = 1, int scaleY = 1)
{
    Mat hist_img = Mat::zeros(100, 256, CV_8U);
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 100; j++) {
            if (hist_array.at<double>(0, i) * 100 > j) {
                hist_img.at<unsigned char>(99 - j, i) = 255;
            }
        }
    bitwise_not(hist_img, hist_img); // инвертируем изображение 
    resize(hist_img, hist_img, Size(hist_img.cols * scaleX, hist_img.rows * scaleY), scaleX, scaleY, INTER_NEAREST);
    return hist_img;
}

vector<float> zigzag(Mat block, int size) {
    std::vector<int> zigzagOrder = {
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63
    };

    std::vector<float> result;
    int count = 0;
    float prev = block.at<float>(0, 0);
    for (int i = 0; i < zigzagOrder.size(); i++) {
        int row = zigzagOrder[i] / size;
        int col = zigzagOrder[i] % size;
        float val = block.at<float>(row, col);

        if (val == prev) {
            count++;
        }else {
            result.push_back(count);
            result.push_back(prev);
            count = 1;
            prev = val;
        }
    }
    result.push_back(count);
    result.push_back(prev);
    result.push_back(-1);
    return result;
}

Mat zigzagDecoder(const vector<float>& codeRLE, int size)
{
    Mat result(size, size, CV_32F);
    const int zigzagOrder[64] = {
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63
    };

    int count = 0;
    for (int j = 0; j < codeRLE.size() - 1; j += 2)
    {
        int N = codeRLE[j];
        int V = codeRLE[j + 1];
        int n_i = 0;

        for (int i = count; i < size * size and n_i != N; i++, n_i ++)
        {
            result.at<float>(zigzagOrder[i] / size, zigzagOrder[i] % size) = V;
        }

        count += N;
    }
    return result;
}

void Task1(const Mat& img) {
    // Создаем заполненный нулями Mat-контейнер размером 1 x 256 
    Mat hist = Mat::zeros(1, 256, CV_64FC1);
    for (int i = 0; i < img.cols; i++)
        for (int j = 0; j < img.rows; j++) {
            int r = img.at<unsigned char>(j, i);
            hist.at<double>(0, r) = hist.at<double>(0, r) + 1.0;
        }
    double m = 0, M = 0;
    minMaxLoc(hist, &m, &M); // ищем глобальный минимум и максимум 
    Mat histNorm = hist / M; // используем максимум для нормировки по высоте 

    int totalPixels = img.rows * img.cols;
    Mat histProbs = Mat::zeros(1, 256, CV_64FC1);
    for (int i = 0; i < 256; i++)
        histProbs.at<double>(0, i) = hist.at<double>(0, i) / totalPixels;

    double entropy = 0;
    for (int i = 0; i < 256; i++)
        if (histProbs.at<double>(0, i) > 0)
            entropy -= histProbs.at<double>(0, i) * log2(histProbs.at<double>(0, i));
    double redundancy = 1 - (entropy / 8);

    imshow("orig img", img);
    imshow("hist minmax normalize", getHist(histNorm, 2, 2));
    imshow("hist probabilities", getHist(histProbs, 2, 2));
    cout << "Entropy: " << entropy << endl;
    cout << "Redundancy: " << redundancy << endl;
}

void Task2(const Mat& img) {
    Mat output_img = Mat::zeros(img.size(), CV_8U);
    Mat img_f;
    img.convertTo(img_f, CV_32F, 1.0 / 255);
    Mat histDctKef = Mat::zeros(1, 256, CV_64FC1);

    // Разделение изображения на блоки 8x8 и выполнение ДКП преобразования для каждого блока 
    for (int y = 0; y < img_f.rows; y += 8) {
        for (int x = 0; x < img_f.cols; x += 8) {
            Mat block = img_f(Rect(x, y, 8, 8)) * 255 - 128; //to [-127,127]

            dct(block, block);
            block = (block / 8) + 128; //to [0,255]

            block.convertTo(block, CV_8U);
            Mat outBlockRect(output_img, Rect(x, y, 8, 8));
            block.copyTo(outBlockRect);

            // Вычисление гистограммы для блока
            for (int i = 0; i < block.rows; i++)
                for (int j = 0; j < block.cols; j++)
                    histDctKef.at<double>(0, block.at<uchar>(i, j))++;
        }
    }

    double histDctKefMin = 0, histDctKefMax = 0;
    minMaxLoc(histDctKef, &histDctKefMin, &histDctKefMax); // ищем глобальный минимум и максимум 
    histDctKef = histDctKef / histDctKefMax;

    int totalPixels = img.rows * img.cols;
    Mat histDctKefProbs = Mat::zeros(1, 256, CV_64FC1);
    for (int i = 0; i < 256; i++)
        histDctKefProbs.at<double>(0, i) = histDctKef.at<double>(0, i) / totalPixels;

    double entropyDctKef = 0;
    for (int i = 0; i < 256; i++)
        if (histDctKefProbs.at<double>(0, i) > 0)
            entropyDctKef -= histDctKefProbs.at<double>(0, i) * log2(histDctKefProbs.at<double>(0, i));
    double redundancyDctKef = 1 - (entropyDctKef / 8);

    imshow("DCT img", output_img);
    imshow("hist dct kef normalize", getHist(histDctKef, 2, 2));
    imshow("hist dct kef probabilities", getHist(histDctKefProbs, 2, 2));
    cout << "Entropy dct kef: " << entropyDctKef << endl;
    cout << "Redundancy dct kef: " << redundancyDctKef << endl;
}

int main()
{
    // Загрузка изображения
    int colorType = 0;
    Mat img = imread("shrek.png", colorType);
    if (img.empty()) {
        return 0;
    }
    resize(img, img, Size(img.cols - img.cols % 8, img.rows - img.rows % 8));


    //-------------------- 1 ------------------
    Task1(img);



    //-------------------- 2 ------------------
    Task2(img);



    //-------------------- 3 и 4------------------
    
    Mat gamma = Mat::zeros(8, 8, CV_32S);
    int quality = 1; // Ввод коэффициента масштаба квантования [1;31]
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            gamma.at<int>(i, j) = 8 + (i + j) * quality;

    Mat img_f;
    img.convertTo(img_f, CV_32F, 1.0 / 255);
    vector<vector<float>> resultCodeRLE;
    int resultCounter = 0;

    // Разделение изображения на блоки 8x8 и выполнение ДКП преобразования для каждого блока 
    for (int y = 0; y < img_f.rows; y += 8) {
        for (int x = 0; x < img_f.cols; x += 8) {
            Mat block = img_f(Rect(x, y, 8, 8)) * 255 - 128; //to [-127,127]

            dct(block, block);
            //block = (block) / 8;
            
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    block.at<float>(i, j) = round(block.at<float>(i, j) / gamma.at<int>(i, j));
                }
            }

            vector<float> blockRLE = zigzag(block, 8);
            resultCodeRLE.push_back(blockRLE);
            resultCounter++;
        }
    }



    //-------------------- 5 ------------------

    float bitImg = img.rows * img.cols * 8;
    float bitRLE = 0;
    for (int i = 0; i < resultCodeRLE.size(); i++)
    {
        bitRLE += resultCodeRLE[i].size() * 8;
    }
    cout << "Img bits count = " << (int)bitImg << endl;
    cout << "Img RLE bits count = " << (int)bitRLE << endl;
    cout << "Profit RLE = " << (1 - bitRLE / bitImg) * 100 << "%" << endl;



    //-------------------- 6 ------------------

    Mat decode_img = Mat::zeros(img.size(), CV_8U);
    int imgBlocksWidth = img.cols / 8;

    for (int i = 0; i < resultCodeRLE.size(); i++)
    {
        Mat block = zigzagDecoder(resultCodeRLE[i], 8);

        for (int z = 0; z < 8; z++) {
            for (int c = 0; c < 8; c++) {
                block.at<float>(z, c) = round(block.at<float>(z, c) * gamma.at<int>(z, c));
            }
        }
        
        block.convertTo(block, CV_32F);
        //varible_img = (varible_img) * 8;

        idct(block, block);

        block = (block + 128);

        block.convertTo(block, CV_8U);
        int block_row = i / imgBlocksWidth;
        int block_col = i % imgBlocksWidth;
        Mat outBlockRect(decode_img, Rect(block_col * 8, block_row * 8, 8, 8));
        block.copyTo(outBlockRect);
    }


    Mat subtract_img;
    subtract(img, decode_img, subtract_img);

    imshow("decode img", decode_img);
    imshow("subtract img", subtract_img);

    waitKey(0);
    return 0;
}

