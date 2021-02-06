#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

constexpr auto BASERANK = 2;
constexpr auto IMGRANK = 16;

//Function to find isolated SVD rank matrix.
Mat SVDrank(Mat inp, int rank)
{
	Mat S, U, VT;
	SVD::compute(inp, S, U, VT);
	inp = U.col(rank) * VT.row(rank) * S.at<float>(rank);

	return inp;
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

//Function for printing compression efficiency and accuracy to console.
void printStats(Mat* inp, Mat* out, int rank)
{
	Mat temp[3];
	float efficiency = ((inp[0].rows + inp[0].cols + 1.0) *  ((float) rank + 1)) / (inp[0].rows * inp[0].cols) * 100.0;
	float accuracy = 0.0;

	for (int i = 0; i < 3; i++)
	{
		compare(inp[i], out[i], temp[i], CMP_LE); //Experiment with this parameter
		accuracy += countNonZero(temp[i]);
	}

	accuracy /= 0.01 * 3.0 * inp[0].rows * inp[0].cols;
	
	cout << "Rank: " << rank << endl;
	cout << "Space required: " << efficiency << "%" << endl;
	cout << "Information stored: " << accuracy << "%" << endl;
}

int main(int argv, char** argc)
{
	Mat base(8, 8, CV_8UC3, Scalar(255, 255, 255));
	Mat expanded(512, 512, CV_8UC3);
	Mat img = imread("ipsum.png", IMREAD_COLOR);
	Mat baseSplit[3], expandSplit[3], imgSplit[3], calcSplit[3], imgOutput;
	expandSplit[0] = Mat::zeros(512, 512, CV_8UC1);
	expandSplit[1] = Mat::zeros(512, 512, CV_8UC1);
	expandSplit[2] = Mat::zeros(512, 512, CV_8UC1);

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
	split(img, imgSplit);

	//Performs BGR SVD compression for 8 * 8 sample by a factor of BASERANK
	for (int i = 0; i < 3; i++)
	{
		calcSplit[i] = baseSplit[i];
		baseSplit[i].convertTo(baseSplit[i], CV_32FC1);
		baseSplit[i] = SVDsum(baseSplit[i], BASERANK);
		baseSplit[i].convertTo(baseSplit[i], CV_8UC1);
	}

	//Prints storage efficiency and information accuracy stats of 8 * 8 sample to console
	printStats(calcSplit, baseSplit, BASERANK);

	//Performs BGR SVD compression for user input image by a factor of IMGRANK
	for (int i = 0; i < 3; i++)
	{
		calcSplit[i] = imgSplit[i];
		imgSplit[i].convertTo(imgSplit[i], CV_32FC1);
		imgSplit[i] = SVDsum(imgSplit[i], IMGRANK);
		imgSplit[i].convertTo(imgSplit[i], CV_8UC1);
	}

	//Prints storage efficiency and information accuracy stats of user input image to console
	printStats(calcSplit, imgSplit, IMGRANK);
	
	//Inflates 8 * 8 image to 512 * 512
	for (int i = 0; i < 512; i++)
	{
		for (int j = 0; j < 512; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				expandSplit[k].at<uchar>(Point(i, j)) = baseSplit[k].at<uchar>(Point(i >> 6, j >> 6));
			}
		}
	}

	//Merges BGR components to form color image
	merge(expandSplit, 3, expanded);
	merge(imgSplit, 3, imgOutput);

	//Displays image
	imshow("channel_1", expanded);
	imshow("channel_2", imgOutput);
	waitKey(0);
}
