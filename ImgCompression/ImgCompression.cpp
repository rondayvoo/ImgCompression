#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

constexpr auto LOWRENDER = 0; //For rendering less
constexpr auto BASERANK = 2;
constexpr auto BASEKVAL = 10; //max 14
constexpr auto IMGRANK = 50;
constexpr auto IMGKVAL = 2; //max 14

//Function to find isolated SVD rank matrix.
Mat SVDrank(Mat inp, int rank)
{
	Mat S, U, VT, output;
	SVD::compute(inp, S, U, VT);
	output = U.col(rank) * VT.row(rank) * S.at<float>(rank);

	return output;
}

//Function to find the sum of the first maxRank SVD matrices.
Mat SVDsum(Mat inp, int maxRank)
{
	Mat temp, S, U, VT, output;

	SVD::compute(inp, S, U, VT);
	output = U.col(0) * VT.row(0) * S.at<float>(0);

	for (int i = 1; i < maxRank + 1; i++)
	{
		temp = U.col(i) * VT.row(i) * S.at<float>(i);
		add(temp, output, output);
	}

	return output;
}

//Function to find the sum of the first kValue diagonals of the DCT.
Mat DCTsum(Mat inp, int kValue)
{
	Mat output = inp;
	Mat DCTtemp(8, 8, CV_32F);

	for (int i = 0; i < output.cols >> 3; i++)
	{
		for (int j = 0; j < output.rows >> 3; j++)
		{
			for (int x = 0; x < 8; x++)
			{
				for (int y = 0; y < 8; y++)
				{
					DCTtemp.at<float>(Point(x, y)) = output.at<float>(Point(8 * i + x, 8 * j + y));
				}
			}

			dct(DCTtemp, DCTtemp);

			for (int k = 0; k < 8; k++)
			{
				for (int m = 0; m < 8; m++)
				{
					if (k + m >= kValue)
					{
						DCTtemp.at<float>(Point(k, m)) = 0.0;
					}
				}
			}

			idct(DCTtemp, DCTtemp);

			for (int x = 0; x < 8; x++)
			{
				for (int y = 0; y < 8; y++)
				{
					output.at<float>(Point(8 * i + x, 8 * j + y)) = DCTtemp.at<float>(Point(x, y));
				}
			}
		}
	}

	return output;
}

//Function for printing compression efficiency and accuracy of SVD to console.
void printStatsSVD(Mat* inp, Mat* out, int rank)
{
	Mat temp[3];
	float efficiency = ((inp[0].rows + inp[0].cols + 1.0) * ((float)rank + 1)) / (inp[0].rows * inp[0].cols) * 100.0;
	float accuracy = 0.0;

	for (int i = 0; i < 3; i++)
	{
		compare(inp[i], out[i], temp[i], CMP_LE);
		accuracy += countNonZero(temp[i]);
	}

	accuracy /= 0.01 * 3.0 * inp[0].rows * inp[0].cols;

	cout << "SVD Rank: " << rank << endl;
	cout << "Space required: " << efficiency << "%" << endl;
	cout << "Information stored: " << accuracy << "%" << endl << endl;
}

//Function for printing compression efficiency and accuracy of DCT to console.
void printStatsDCT(Mat* inp, Mat* out, int kval)
{
	Mat temp[3];
	float efficiency = 0.0;
	float accuracy = 0.0;

	if (kval < 9)
	{
		efficiency = (float)((kval + 1) * (kval + 2)) / 128 * 100;
	}

	else
	{
		efficiency = (float)(64 - ((14 - kval) * (14 - kval + 1) / 2)) / 64 * 100;
	}

	for (int i = 0; i < 3; i++)
	{
		compare(inp[i], out[i], temp[i], CMP_LE);
		accuracy += countNonZero(temp[i]);
	}

	accuracy /= 0.01 * 3.0 * inp[0].rows * inp[0].cols;

	cout << "DCT K-Value: " << kval << endl;
	cout << "Space required: " << efficiency << "%" << endl;
	cout << "Information stored: " << accuracy << "%" << endl << endl;
}

int main(int argv, char** argc)
{
	Mat base(8, 8, CV_8UC3, Scalar(255, 255, 255));
	Mat expandedSVD(512, 512, CV_8UC3);
	Mat expandedDCT(512, 512, CV_8UC3);
	Mat img = imread("credit.png", IMREAD_COLOR);
	Mat baseSplit[3], baseSVD[3], baseDCT[3], expandSplitSVD[3], expandSplitDCT[3], imgSplit[3], imgSVD[3], imgDCT[3];
	Mat imgSVDout, imgDCTout;

	for (int i = 0; i < 3; i++)
	{
		expandSplitSVD[i] = Mat::zeros(512, 512, CV_8UC1);
		expandSplitDCT[i] = Mat::zeros(512, 512, CV_8UC1);
	}

	Vec3b color = { 0, 0, 0 };
	base.at<Vec3b>(Point(2, 1)) = color;
	base.at<Vec3b>(Point(2, 2)) = color;
	base.at<Vec3b>(Point(5, 1)) = color;
	base.at<Vec3b>(Point(5, 2)) = color;
	base.at<Vec3b>(Point(2, 5)) = color;
	base.at<Vec3b>(Point(3, 5)) = color;
	base.at<Vec3b>(Point(4, 5)) = color;
	base.at<Vec3b>(Point(5, 5)) = color;
	base.at<Vec3b>(Point(1, 4)) = color;
	base.at<Vec3b>(Point(6, 4)) = color;

	//Splits BGR representation into individual matrices
	split(base, baseSplit);

	//Performs BGR compression for 8 * 8 sample by a factor of BASERANK/BASEKVAL
	for (int i = 0; i < 3; i++)
	{
		baseSVD[i] = baseSplit[i];
		baseSVD[i].convertTo(baseSVD[i], CV_32F);
		baseSVD[i] = SVDsum(baseSVD[i], BASERANK);
		baseSVD[i].convertTo(baseSVD[i], CV_8UC1);

		baseDCT[i] = baseSplit[i];
		baseDCT[i].convertTo(baseDCT[i], CV_32F);
		baseDCT[i] = DCTsum(baseDCT[i], BASEKVAL);
		baseDCT[i].convertTo(baseDCT[i], CV_8UC1);
	}

	//Prints storage efficiency and information accuracy stats of 8 * 8 sample to console
	printStatsSVD(baseSplit, baseSVD, BASERANK);
	printStatsDCT(baseSplit, baseDCT, BASEKVAL);

	//Inflates 8 * 8 image to 512 * 512
	for (int i = 0; i < 512; i++)
	{
		for (int j = 0; j < 512; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				expandSplitSVD[k].at<uchar>(Point(i, j)) = baseSVD[k].at<uchar>(Point(i >> 6, j >> 6));
				expandSplitDCT[k].at<uchar>(Point(i, j)) = baseDCT[k].at<uchar>(Point(i >> 6, j >> 6));
			}
		}
	}

	//Merges BGR components to form color image
	merge(expandSplitSVD, 3, expandedSVD);
	merge(expandSplitDCT, 3, expandedDCT);

	if (LOWRENDER == 0)
	{
		//Splits img to BGR matrices
		split(img, imgSplit);

		//Performs BGR compression for user input image by a factor of IMGRANK/IMGKVAL
		for (int i = 0; i < 3; i++)
		{
			imgSVD[i] = imgSplit[i];
			imgSVD[i].convertTo(imgSVD[i], CV_32F);
			imgSVD[i] = SVDsum(imgSVD[i], IMGRANK);
			imgSVD[i].convertTo(imgSVD[i], CV_8UC1);

			imgDCT[i] = imgSplit[i];
			imgDCT[i].convertTo(imgDCT[i], CV_32F);
			imgDCT[i] = DCTsum(imgDCT[i], IMGKVAL);
			imgDCT[i].convertTo(imgDCT[i], CV_8U);
		}

		//Prints storage efficiency and information accuracy stats of user input image to console
		printStatsSVD(imgSplit, imgSVD, IMGRANK);
		printStatsDCT(imgSplit, imgDCT, IMGKVAL);

		//Merges BGR arays to form color image
		merge(imgSVD, 3, imgSVDout);
		merge(imgDCT, 3, imgDCTout);

		//Displays image
		imshow("img_SVD", imgSVDout);
		imshow("img_DCT", imgDCTout);
	}

	imshow("base_SVD", expandedSVD);
	imshow("base_DCT", expandedDCT);

	waitKey(0);
}