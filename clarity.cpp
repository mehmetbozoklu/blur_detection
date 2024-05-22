/*!
   \file "blur_detection"
   \brief "test clarity on pics which are given by path"
   \author "Mehmet BOZOKLU"
   \date "21"/"May"/"2024"
*/

#include <iostream>
#include <string>
#include <filesystem>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
namespace fs = filesystem;


/*!
   \brief "OpenCV port of 'LAPM' algorithm"
   \param "src" image source
   \return "measurement of focus"
*/
double lapm(const Mat& src) {
    Mat M = (Mat_<double>(3, 1) << -1, 2, -1);
    Mat G = getGaussianKernel(3, -1, CV_64F);

    Mat Lx;
    sepFilter2D(src, Lx, CV_64F, M, G);

    Mat Ly;
    sepFilter2D(src, Ly, CV_64F, G, M);

    Mat FM = abs(Lx) + abs(Ly);

    double focusMeasure = mean(FM).val[0];
    return focusMeasure;
}


/*!
   \brief "OpenCV port of 'LAPV' algorithm"
   \param "src" image source
   \return "measurement of focus"
*/
double lapv(const Mat& src) {
    Mat lap;
    Laplacian(src, lap, CV_64F);

    Scalar mu, sigma;
    meanStdDev(lap, mu, sigma);

    double focusMeasure = sigma.val[0]*sigma.val[0];
    return focusMeasure;
}


/*!
   \brief "OpenCV port of 'TENG' algorithm"
   \param "src" image source
   \return "measurement of focus"
*/
double teng(const Mat& src, int ksize) {
    Mat Gx, Gy;
    Sobel(src, Gx, CV_64F, 1, 0, ksize);
    Sobel(src, Gy, CV_64F, 0, 1, ksize);

    Mat FM = Gx.mul(Gx) + Gy.mul(Gy);

    double focusMeasure = mean(FM).val[0];
    return focusMeasure;
}


/*!
   \brief "OpenCV port of 'GLVN' algorithm"
   \param "src" image source
   \return "measurement of focus"
*/
double glvn(const Mat& src) {
    Scalar mu, sigma;
    meanStdDev(src, mu, sigma);

    double focusMeasure = (sigma.val[0]*sigma.val[0]) / mu.val[0];
    return focusMeasure;
}

/*!
   \brief "sort by second of pair"
   \param "a" image path
   \param "b" image focus value
   \return "sorting bool"
*/
bool sortValue(const pair<string, double> &a, const pair<string, double> &b) {
    return (a.second < b.second);
}


/*!
   \brief "path can be given via command line" Usage:
   ./clarity /your/image/files/path/
   or
   ./clarity
*/
int main(int argc, char* argv[])
{
    //modify this line for your files
    string path = "../dataset/";

    if(argc>1){
      cout << argc << " " << argv[1] << endl;
      path = argv[1];
    }

    vector<pair<string, double>> table_ml;
    vector<pair<string, double>> table_vl;
    vector<pair<string, double>> table_t;
    vector<pair<string, double>> table_nv;
    double ml, vl, t, nv;

    for (const auto & entry : fs::directory_iterator(path)) {
        cout << entry.path() << endl;

        Mat img = imread(entry.path(), IMREAD_COLOR);
        if(img.empty()) {
            cout << "Could not read the image: " << entry.path() << endl;
            return 1;
        }

        imshow(entry.path(), img);

        ml = lapm(img);
        vl = lapv(img);
        t  = teng(img, 3);
        nv = glvn(img);

        cout << "Lapm :" << ml << endl;
        cout << "Lapv :" << vl << endl;
        cout << "Teng :" << t  << endl;
        cout << "Glvn :" << nv << endl;
        cout << endl;

        table_ml.push_back(make_pair(entry.path(), ml));
        table_vl.push_back(make_pair(entry.path(), vl));
        table_t.push_back(make_pair(entry.path(),   t));
        table_nv.push_back(make_pair(entry.path(), nv));

        int k = waitKey(0); // Wait for a key in the window for next pic
    }

    sort(table_ml.begin(), table_ml.end(), sortValue);
    sort(table_vl.begin(), table_vl.end(), sortValue);
    sort(table_t.begin(),  table_t.end(),  sortValue);
    sort(table_nv.begin(), table_nv.end(), sortValue);

    cout << "Sorting pics from blur to clarity:" << endl;

    for (int i=0; i<size(table_ml); i++) {
        cout << table_ml[i].first << " (lapm): " << table_ml[i].second << endl;
        cout << table_vl[i].first << " (lapv): " << table_vl[i].second << endl;
        cout << table_t[i].first  << " (teng): " << table_t[i].second  << endl;
        cout << table_nv[i].first << " (glnv): " << table_nv[i].second << endl;
        cout << endl;
    }

    destroyAllWindows();

    return 0;
}
