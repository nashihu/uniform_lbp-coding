#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include "cv.h"
#include "highgui.h"
#include <opencv2/ml/ml.hpp>
#include <math.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

//#define num_ 458;
int getHopCount(uchar i)
{
	int a[8] = { 0 };
	int k = 7;
	int cnt = 0;
	while (i)
	{
		a[k] = i & 1;
		i >>= 1;
		--k;
	}
	for (int k = 0; k<8; ++k)
	{
		if (a[k] != a[k + 1 == 8 ? 0 : k + 1])
		{
			++cnt;
		}
	}
	return cnt;
}

void lbp59table(uchar* table)
{
	memset(table, 0, 256);
	uchar temp = 1;
	for (int i = 0; i<256; ++i)
	{
		if (getHopCount(i) <= 2)
		{
			table[i] = temp;
			temp++;
		}
		//printf("%d\n",table[i]);
	}
}

void LBP(IplImage* src, IplImage* dst)
{
	int width = src->width;
	int height = src->height;
	uchar table[256];
	lbp59table(table);
	for (int j = 1; j<width - 1; j++)
	{
		for (int i = 1; i<height - 1; i++)
		{
			uchar neighborhood[8] = { 0 };
			neighborhood[7] = CV_IMAGE_ELEM(src, uchar, i - 1, j - 1);
			neighborhood[6] = CV_IMAGE_ELEM(src, uchar, i - 1, j);
			neighborhood[5] = CV_IMAGE_ELEM(src, uchar, i - 1, j + 1);
			neighborhood[4] = CV_IMAGE_ELEM(src, uchar, i, j + 1);
			neighborhood[3] = CV_IMAGE_ELEM(src, uchar, i + 1, j + 1);
			neighborhood[2] = CV_IMAGE_ELEM(src, uchar, i + 1, j);
			neighborhood[1] = CV_IMAGE_ELEM(src, uchar, i + 1, j - 1);
			neighborhood[0] = CV_IMAGE_ELEM(src, uchar, i, j - 1);
			uchar center = CV_IMAGE_ELEM(src, uchar, i, j);
			uchar temp = 0;

			for (int k = 0; k<8; k++)
			{
				temp += (neighborhood[k] >= center)*(1 << k);
			}
			//CV_IMAGE_ELEM( dst, uchar, i, j)=temp;
			CV_IMAGE_ELEM(dst, uchar, i, j) = table[temp];
		}
	}
}
//#define aa 5;
int main()
{
	const char *name_r;
	int i = 1;
	//int ii = aa;
	//cout << aa << endl;
	//ofstream fin;
	//fin.open("E:\\blink.txt");
	//float feature[ 458 ][59] = { 0 };
	//Mat::ones()
	int row = 458;
	int col = 59;
	Mat feature = Mat::zeros(row, col, CV_32FC1);
	Mat feture_cha = Mat::zeros(row - 1, col, CV_32FC1);
	//float feature_cha[457][59] = { 0 };
	Mat sample = Mat::zeros(1, col * 2,CV_32FC1);
	//float sample[1][118] = { 0 };
	Mat feature_blink = Mat::zeros(row - 1, col * 2, CV_32FC1);
	//float feature_blink[457][118] = { 0 };
	float response[1][457] = {0};
	for (int i = 1; i <= 458; i++)
	{
		int temp = 0;
		int sum = 0;
		string str = to_string(i)+".bmp";
		string name = "E:\\data\\SVM\\SVM\\"+str;
		name_r = name.c_str();
		IplImage* img = cvLoadImage(name_r, 0);
		//IplImage img_ = IplImage(image);
		//IplImage* img = &img_;
		IplImage* dst = cvCreateImage(cvGetSize(img), 8, 1);
		LBP(img, dst);
		Mat result = cvarrToMat(dst);
		for (int j = 1; j < result.cols - 1; j++)
		{
			for (int k = 1; k < result.rows - 1; k++)
			{
				temp = int(result.at<uchar>(k, j));
				//cout << temp << " ";
				//cout << sum << endl;
				//system("pause");
				//cout << temp << endl;
				if (temp <= 58 && temp > 0)
				{
					feature[i - 1][temp]++;
				}
				if (temp == 0)
				{
					feature[i - 1][temp]++;
				}
			}
		}
		for (int j = 0; j < 59; j++)
		{
			sum = sum + feature[i - 1][j] * feature[i - 1][j];
		}
		//fin << sum << ' ';

		for (int j = 0; j < 59; j++)
		{
			feature[i - 1][j] = feature[i - 1][j] / sqrt(sum);

		}
		cvReleaseImage(&img);
		cvReleaseImage(&dst);
	}
	for (int i = 1; i <= 457; i++)
	{
		for (int j = 0; j < 59; j++)
		{
			feature_cha[i - 1][j] = feature[i][j] - feature[i - 1][j];
			feature_blink[i - 1][j + 59] = feature_cha[i - 1][j];
			feature_blink[i - 1][j] = feature[i][j];
		}
	}
	Mat trainingDataMat = Mat(457, 118, CV_32FC1, feature_blink);
	//read the label and the data
	ifstream infile;
	infile.open("E:\\data\\SVM\\SVM\\label.txt", ios::in);
	char str[1000];
	int label[457] = { -1 };
	while (!infile.eof())
	{
		infile >> str;
	}
	int length = strlen(str);
	//bool a = (str[13] == '1');
	//cout << a << endl;
	for (int m = 0; m < length; m++)
	{
		label[m] = -1;
	}
	for (int m = 0; m < length; m++)
	{
		if (str[m] == '1')
			label[m - 1] = 1;
	}
	for (int m = 0; m < length; m++)
	{
		cout << label[m] << " ";
	}
	cout << endl;
	//train the svm classifier
	Mat labelsMat = Mat(457, 1, CV_32SC1, label);
	//set the svm parameter
	Ptr<ml::SVM> svm = ml::SVM::create();
	//svm->GAMMA = 0.1;
	//svm->
	svm->setType(ml::SVM::Types::C_SVC);
	svm->setKernel(ml::SVM::KernelTypes::RBF);
	//svm->setGamma( 5);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
	svm->train(trainingDataMat, ml::SampleTypes::ROW_SAMPLE, labelsMat);

	//svm->train(trainingDataMat, cv::ml::SampleTypes::ROW_SAMPLE ,labelsMat);
	//svm->get
	//int c = svm->get_support_vector_count();
	//cout << c<<endl;
	Mat sv = svm->getSupportVectors();
	for (int i = 0; i<sv.rows; i++)
	{
		const float* v = sv.ptr<float>(i);
		cout << *v << " ";
	}
	svm->save("svmalive2.3.2-version4.xml");
	infile.close();
	//Ptr<ml::SVM> SVM;
	Ptr<ml::SVM> SVM = ml::StatModel::load<ml::SVM>("svmalive2.3.2-version4.xml");
	for (int i = 1; i <= 457; i++)
	{
		for (int j = 0; j < 118; j++)
		{
			sample[0][j] = feature_blink[i - 1][j];
			//feature_blink[i - 1][j + 59] = feature_cha[i - 1][j];
			//feature_blink[i - 1][j] = feature[i][j];

		}
		Mat testDataMat = Mat(1, 118, CV_32FC1, sample);
		response[0][i-1] = SVM->predict(testDataMat);
	}
	
	//cout << response;
	//if (response == 1)
	//{
	//	cout << "ceshishuju:" << i << endl;
	//}
	//imshow("ÊµÊ±¶ÔÆë",photo);
	//waitKey(10);

	system("pause");
	return 0;
}