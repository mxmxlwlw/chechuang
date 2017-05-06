#include <iostream>
#include <fstream>
#include <vector>
#include "opencv2/opencv.hpp"


class OutOfWindows
{
	cv::Ptr<cv::BackgroundSubtractorMOG2> bkgSubTrackor_;
	cv::Point2f x1_;
	cv::Point2f x2_;
	cv::Point2f x3_;
	float dist1_;
	float dist2_;
	float dist3_;
	float dist4_;
	float thresh_;
	float scaleFactor_;

	int history_;
	double varParam_;
	bool detectShadow_;

	cv::Mat transferM1_;
	cv::Mat transferM2_;
	cv::Size sz1_;
	cv::Size sz2_;
	cv::Mat subimg_;
public:
	OutOfWindows()
	{
		struct helper
		{
			static void getConfigFromFile(std::string path, cv::Point2f &x1, cv::Point2f &x2, cv::Point2f &x3,
				float &dist1, float &dist2, float &dist3, float &dist4, 
				float &thresh, float &scaleFactor, int &history, double &varParam, bool &detectShadow)
			{
				std::ifstream fid("../config.txt");
				x1 = getPoint(fid);
				x2 = getPoint(fid);
				x3 = getPoint(fid);
				dist1 = getDist(fid);
				dist2 = getDist(fid);
				dist3 = getDist(fid);
				dist4 = getDist(fid);
				std::string buffer;
				fid >> buffer;
				thresh = atof(buffer.c_str());
				fid >> buffer;
				scaleFactor = atof(buffer.c_str());
				fid >> buffer;
				history = atoi(buffer.c_str());
				fid >> buffer;
				varParam = atof(buffer.c_str());
				fid >> buffer;
				detectShadow = atoi(buffer.c_str());
				x1 *= scaleFactor;
				x2 *= scaleFactor;
				x3 *= scaleFactor;
				dist1 *= scaleFactor;
				dist2 *= scaleFactor;
				dist3 *= scaleFactor;
				dist4 *= scaleFactor;
			}

			static float getDist(std::ifstream &fid)
			{
				std::string buffer;
				fid >> buffer;
				return atof(buffer.c_str());
			}

			static cv::Point2f getPoint(std::ifstream &fid)
			{
				std::string buffer;
				fid >> buffer;
				float a = atof(buffer.c_str());
				fid >> buffer;
				float b = atof(buffer.c_str());
				return cv::Point2f(a, b);
			}

			static void getPtsBy2Pts(cv::Point2f &pt1, cv::Point2f &pt2, float dist1, float dist2, std::vector<cv::Point2f> &vPts)
			{
				cv::Point2f k = pt1 - pt2;
				float dist = cv::norm(k);
				cv::Point2f k2(-k.y / dist, k.x / dist);
				cv::Point p;
				p = dist1*k2 + pt1;
				vPts.push_back(p);
				p = dist1*k2 + pt2;
				vPts.push_back(p);
				p = -dist2*k2 + pt2;
				vPts.push_back(p);
				p = -dist2*k2 + pt1;
				vPts.push_back(p);
			}

			static void getTransferM(cv::Point2f &pt1, cv::Point2f &pt2, float dist1, float dist2, cv::Mat &transferM, cv::Size &sz)
			{
				std::vector<cv::Point2f> rect1, rect2;
				getPtsBy2Pts(pt1, pt2, dist1, dist2, rect1);
				float width, height;
				width = cv::norm(rect1[1] - rect1[0]);
				height = cv::norm(rect1[1] - rect1[2]);
				rect2.push_back(cv::Point(0, 0));
				rect2.push_back(cv::Point(width, 0));
				rect2.push_back(cv::Point(width, height));
				rect2.push_back(cv::Point(0, height));
				transferM=cv::getPerspectiveTransform(rect1, rect2);
				sz = cv::Size(width, height);
			}
		};

		helper::getConfigFromFile("../config.txt", x1_, x2_, x3_, dist1_, dist2_, dist3_, dist4_, thresh_, scaleFactor_, history_, varParam_, detectShadow_);
		bkgSubTrackor_ = cv::createBackgroundSubtractorMOG2(history_, varParam_, detectShadow_);
		helper::getTransferM(x1_, x2_, dist1_, dist2_, transferM1_, sz1_);
		helper::getTransferM(x2_, x3_, dist3_, dist4_, transferM2_, sz2_);
		float maxHeight = cv::max(sz1_.height, sz2_.height);
		subimg_ = cv::Mat(maxHeight, sz1_.width + sz2_.width, CV_8UC1, cv::Scalar(0, 0, 0));
	}
	bool isOut(cv::Mat &frame)
	{
		cv::cvtColor(frame, frame, CV_BGR2GRAY);
		cv::resize(frame, frame, cv::Size(), scaleFactor_, scaleFactor_);
		cv::GaussianBlur(frame, frame, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		cv::Mat grad_x;
		cv::Sobel(frame, grad_x, CV_16SC1, 1, 0, 3);
		cv::convertScaleAbs(grad_x, frame);

		cv::Mat img1, img2;
		cv::warpPerspective(frame, img1, transferM1_, sz1_);
		cv::warpPerspective(frame, img2, transferM2_, sz2_);
		cv::Point2f p(0, 0);
		cv::Point2f p2(sz1_.width, 0);
		img1.copyTo(subimg_(cv::Rect(p, sz1_)));
		img2.copyTo(subimg_(cv::Rect(p+p2, sz2_)));
		cv::Mat mask;
		bkgSubTrackor_->apply(subimg_, mask, 0.001);
		cv::imshow("mx", mask);
		cv::waitKey(1);
		float totalSize = mask.rows*mask.cols;
		float count = cv::sum(mask)[0];
		if (count / totalSize > thresh_)
			return true;
		else
			return false;
	}
	
};

int main(){  
	OutOfWindows x;
    cv::VideoCapture video;
	video.open("../v_TaiChi_g01_c01.avi");
    cv::Mat frame,mask,thresholdImage, output;  
    while(true){  
        video.read(frame);  
		if (frame.empty())
			break;
		x.isOut(frame);
		
 
        cv::imshow("img", frame);
        cv::waitKey(1);  
    }  
    return 0;  
}  