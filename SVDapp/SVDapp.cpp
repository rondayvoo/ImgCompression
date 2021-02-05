#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#define MAXRANK 2

using namespace cv;
using namespace std;

Mat SVDrank(Mat inp, int rank)
{
	Mat S, U, VT;
	SVD::compute(inp, S, U, VT);
	inp = U.col(rank) * VT.row(rank) * S.at<float>(rank);
	cout << format(S, Formatter::FMT_DEFAULT) << endl;

	return inp;
}

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

void printStats(Mat* inp, Mat* out, int rank)
{
	Mat temp[3];
	float accuracy = 0;
	float efficiency = ((inp[0].rows + inp[0].cols + 1.0) * (float) rank) / (inp[0].rows * inp[0].cols) * 100.0;

	for (int i = 0; i < 3; i++)
	{
		compare(inp[i], out[i], temp[i], CMP_EQ);
		accuracy += (float) countNonZero(temp[i]);
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
	Mat baseSplit[3], expandSplit[3], calcSplit[3];
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

	//Performs BGR SVD compression by a factor of MAXRANK
	for (int i = 0; i < 3; i++)
	{
		calcSplit[i] = baseSplit[i];
		baseSplit[i].convertTo(baseSplit[i], CV_32FC1);
		baseSplit[i] = SVDsum(baseSplit[i], MAXRANK);
		baseSplit[i].convertTo(baseSplit[i], CV_8UC1);
	}

	//Prints storage efficiency and information accuracy stats to console
	printStats(calcSplit, baseSplit, MAXRANK);
	
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

	//Displays image
	cout << format(baseSplit[0], Formatter::FMT_DEFAULT) << endl;
	namedWindow("channel_1", WINDOW_AUTOSIZE);
	imshow("channel_1", expanded);
	waitKey(0);
}
