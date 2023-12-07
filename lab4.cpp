#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;

//Mat MSICE(const Mat& img) {
//
//    Mat result = img.clone();
//    Mat averages;
//
//    for (int i = 0; i < 3; i++) {
//        int scale = pow(2, i);
//        GaussianBlur(img, averages, Size(0, 0), scale);
//        subtract(img, averages, averages);
//        add(result, averages, result);
//    }
//
//    normalize(result, result, 0, 255, NORM_MINMAX);
//    return result;
//}

Mat MSICE(const Mat& Y, int M) {

    Mat result = Mat::zeros(Y.size(), Y.type());

    int dk[3];
    dk[0] = Y.cols * Y.rows / 7085;
    dk[1] = dk[0] / 2;
    dk[2] = dk[1] / 2;
    Mat Ss[3];

    for (int dk_i = 0; dk_i < 3; dk_i++) {
        blur(Y, Ss[dk_i], Size(dk[dk_i], dk[dk_i]));
    }

    for (int i = 0; i < Y.rows; i++) {
        for (int j = 0; j < Y.cols; j++) {
            float out = 0;
            uchar y = Y.at<uchar>(i, j);
            for (int dk_i = 0; dk_i < 3; dk_i++) {
                uchar s = Ss[dk_i].at<uchar>(i, j);
                float A = (y - s) == 0 ? M : M / (y - s);
                float out_dki;
                if (y >= s) {
                    out_dki = (255 + A) * y / (A + y);
                }
                else {
                    //float A2 = (s - y) == 0 ? M : M / (s - y);
                    out_dki = A * y / (255 + A - y);
                }
                out += out_dki;
            }
            out /= 3;
            result.at<uchar>(i, j) = out <= 0 ? 0 : (out >= 255 ? 255 : out);
            //result.at<uchar>(i, j) = out;
        }
    }

    normalize(result, result, 0, 255, NORM_MINMAX);

    return result;
}

int main()
{
    int colorType = 1;

    Mat img1 = imread("endo.jpg", colorType);
    if (img1.empty()) {
        return 0;
    }

	vector<Mat> rgb;
	split(img1, rgb);
    Mat gAfterMSICE;
    vector<Mat> rgbAfterCLAHE = vector<Mat>(rgb.size());

    gAfterMSICE = MSICE(rgb[1], 50000); //менять параметр, 500, 5000(метода), 50000, 500000

    Ptr<CLAHE> clahe = createCLAHE(2, Size(4, 4)); //2, Size(4, 4)       40, Size(8, 8)
    clahe->apply(rgb[2], rgbAfterCLAHE[2]);
    clahe->apply(gAfterMSICE, rgbAfterCLAHE[1]);
    clahe->apply(rgb[0], rgbAfterCLAHE[0]);

    Mat final;
    merge(rgbAfterCLAHE, final);


    imshow("img1", img1);
    imshow("R", rgb[2]);
    imshow("G", rgb[1]);
    imshow("B", rgb[0]);
    imshow("G After MSICE", gAfterMSICE);
    imshow("R After CLAHE", rgbAfterCLAHE[2]);
    imshow("G After CLAHE", rgbAfterCLAHE[1]);
    imshow("B After CLAHE", rgbAfterCLAHE[0]);
    imshow("final", final);

    //Uncommented this for save images with real size
    //imwrite("/keypoints1draw.jpg", keypoints1draw);
    //imwrite("/keypoints2draw.jpg", keypoints2draw);
    //imwrite("/dMatches.jpg", dMatches);
    //imwrite("/alignedImg2.jpg", alignedImg2);
    //imwrite("/final.jpg", final);

	waitKey();
    return 0;
}
