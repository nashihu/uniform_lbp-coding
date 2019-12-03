#include <cstdint>
#include <fstream>
#include <iostream>
#include <direct.h>
#include <string>
#include <stdlib.h>
#include <stdio.h>
//#include <vector.h>
#include <io.h>
#include "cv.h"
#include "highgui.h"
#include <opencv2/ml/ml.hpp>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <time.h>
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
struct fea
{
	Mat traindata;
	Mat trainlabel;
};
//#define aa 5;
int extract_fea(int row, int col, string file_name,fea *temp);
int sum_row(int row[], int num)
{
	int temp=0;
	for (int i = 0; i < num; i++)
	{
		temp = temp + row[i];
	}
	return temp;
}
int combine_file(string file_name,int dirnumber)
{
	string dir_name = "E:/data/训练样本/photo/movie_photo/";
	int count = 1;
	for (int i = 1; i <= dirnumber; i++)
	{
		_finddata_t fileinfo;
		long handle;
		string dirname = dir_name + to_string(i) + "/*.bmp";
		handle = _findfirst(dirname.c_str(), &fileinfo);
		if (-1 == handle)return -1;
		printf("%s/n", fileinfo.name);
		std::cout << endl;
		string ima_read_name = dir_name + to_string(i) + "/" + fileinfo.name;
		Mat image_temp = imread(ima_read_name.c_str());
		string ima_write_name = "./" + file_name + "./" + to_string(count) + ".bmp";
		imwrite(ima_write_name.c_str(), image_temp);
		count++;
		while (!_findnext(handle, &fileinfo))
		{
			printf("%s/n", fileinfo.name);
			std::cout << endl;
			ima_read_name = dir_name + to_string(i) + "/" + fileinfo.name;
			image_temp = imread(ima_read_name.c_str());
			ima_write_name = "./" + file_name + "./" + to_string(count) + ".bmp";
			imwrite(ima_write_name.c_str(), image_temp);
			count++;
		}
		_findclose(handle);		 //别忘了关闭句柄 
		//system("pause");
	}
	return 0;
}
int main()
{
	int data[94] = { 0 };
	std::ifstream infile;
	//提取file_id
	clock_t  clockBegin, clockEnd;
	string file_path = "E:\\项目\\PD\\PD-冯志连\\";
	for (int file_id = 1; file_id <2; file_id++)
	{	
		//string txt_name = "E:\\data\\blink\\training\\blink\\feature_single.txt";//+ to_string(file_id) + "\\10\\zuo\\feature_single.txt";
		string txt_name = file_path +"feature_you.txt";
		ofstream iofile(txt_name);
		Mat feature = Mat::zeros(5600, 59, CV_32FC1);
		Mat feature_cha = Mat::zeros(9, 59, CV_32FC1);
		Mat feature_blink = Mat::zeros(9, 118, CV_32FC1);
		int time_no = 0;
		int count = 0;
		for (int i =1; i <= 5600; i++)
		{
			int temp = 0;
			int sum = 0;
			string file_name = file_path  + "\\photo_you\\" + to_string(i) + "you.bmp";
			//string file_name = "E:\\vsfile\\jinglun\\eyeunblink\\FaceAlignment\\examples\\examples\\test\\unblink\\unblink\\zuoyan\\" + to_string(data[file_id]) + "\\" + to_string(i) + "eye_zuo.bmp";
			Mat src;
			Mat src_ =  imread(file_name.c_str(), 0);
			if (!src_.empty())
			{
				//std::cout << file_name << endl;
				resize(src_, src, Size(32, 32), 0, 0, CV_INTER_LINEAR);
				Mat img_;
				//clockBegin = clock();
				cv::equalizeHist(src, img_);
				IplImage* image_LBP = &(IplImage(img_));
				IplImage* dst = cvCreateImage(cvGetSize(image_LBP), 8, 1);
				LBP(image_LBP, dst);
				Mat result = cvarrToMat(dst);
				for (int j = 1; j < result.cols - 1; j++)
				{
					for (int k = 1; k < result.rows - 1; k++)
					{
						temp = int(result.at<uchar>(k, j));
						if (temp <= 58 && temp > 0)
						{
							feature.at<float>(i - 1, temp)++;
						}
						if (temp == 0)
						{
							feature.at<float>(i - 1, temp)++;
						}
					}
				}
				for (int j = 0; j < 59; j++)
				{
					sum = sum + feature.at<float>(i - 1, j) * feature.at<float>(i - 1, j);
				}
				for (int j = 0; j < 59; j++)
				{
					feature.at<float>(i - 1, j) = feature.at<float>(i - 1, j) / sqrt(sum);
					iofile << feature.at<float>(i - 1, j) << " ";
				}
				iofile << endl;
			}
			else
			{
				cout << file_name << endl;
			}
		}
		iofile.close();
	}
	system("pause");
}
int extract_fea(int dirnumber, int col, string file_name, fea* temp)
{
	time_t start,finish;
	int sequen_number = 10;
	ofstream iofile("feature.txt");
	int i = 1;
	Mat feature = Mat::zeros(dirnumber*sequen_number, col, CV_32FC1);
	Mat feature_cha = Mat::zeros((sequen_number-1)*dirnumber, col, CV_32FC1);
	Mat feature_blink = Mat::zeros((sequen_number - 1)*dirnumber, col * 2, CV_32FC1);
	double sam=0;
	for (int i = 1; i <= dirnumber*sequen_number; i++)
	{
		int temp = 0;
		int sum = 0;
		string name = "./"+ file_name + "./"+to_string(i)+".bmp";
		//IplImage* img = cvLoadImage(name.c_str(), 0);
		//Mat src=cvarrToMat(img);
		Mat src = imread(name.c_str(), 0);
		Mat img_;
		equalizeHist(src, img_);
		IplImage* image_LBP = &(IplImage(img_));
		IplImage* dst = cvCreateImage(cvGetSize(image_LBP), 8, 1);	
		start = clock();
		LBP(image_LBP, dst);
		Mat result = cvarrToMat(dst);
		
		for (int j = 1; j < result.cols - 1; j++)
		{
			for (int k = 1; k < result.rows - 1; k++)
			{
				temp = int(result.at<uchar>(k, j));
				if (temp <= 58 && temp > 0)
				{
					feature.at<float>(i-1,temp)++;
				}
				if (temp == 0)
				{
					feature.at<float>(i - 1, temp)++;
				}
			}
		}
		for (int j = 0; j < 59; j++)
		{
			sum = sum + feature.at<float>(i-1,j) * feature.at<float>(i-1,j);
		}
		for (int j = 0; j < 59; j++)
		{
			feature.at<float>(i-1,j) = feature.at<float>(i-1,j) / sqrt(sum);
		}	
		finish = clock();
		double tim = double(finish - start)*1000 / CLOCKS_PER_SEC;
		system("pause");
		cout << "time cost:" << tim;
		//cvReleaseImage(&image_LBP);
		sam = sam + tim;
		cvReleaseImage(&dst);
	}
	std::cout << "sam:" << sam << endl;
	system("pause");
	for (int i = 1; i <= dirnumber; i++)
	{
		for (int k = 0; k <= (sequen_number - 2); k++)
		{
			for (int j = 0; j < 59; j++)
			{
				feature_cha.at<float>((i - 1) * (sequen_number - 1) + k, j) = feature.at<float>((i - 1) * sequen_number + k + 1, j) - feature.at<float>((i - 1) * sequen_number + k, j);
				feature_blink.at<float>((i - 1) * (sequen_number - 1) + k, j + 59) = feature_cha.at<float>((i - 1) * (sequen_number - 1) + k, j);
				feature_blink.at<float>((i - 1) * (sequen_number - 1) + k, j) = feature.at<float>((i - 1) * sequen_number + k + 1, j);
			}
			for (int j = 0; j < 118; j++)
			{
				iofile << feature_blink.at<float>((i - 1) * (sequen_number - 1) + k, j) << " ";
			}
			iofile << endl;
		}
		
	}
	iofile.close();
	return 0;
}