#pragma once
#include <sys/time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "rknn_api.h"
#include <chrono>
#include <array>
#include <vector>
#include <stdint.h>
#include <array>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>				
#include <string>
#include "baseStruct.h"
#include "BYTETracker.h"



// different task
enum TASK { GeneralDetection = 0, FaceWithLandMark, HumanPose2D};

// rknn class
class RKNN {
public:	
	RKNN() { }
	RKNN(std::string f, uint32_t rknn_init_flag, int task) : task(task) { 
		this->Init(f, rknn_init_flag);
	}
	~RKNN() {	
		// this->Release();
	}
	
	// load rknn & rknn init
	rknn_context ctx = 0;				// rknn ctx
	rknn_input_output_num io_num;		// get rknn model num_input. num_output
	
	// rknn inputs & outputs
	// rknn_input* rknn_inputs;			// rknn_input inputs[io_num.n_input]; 
	// rknn_output* rknn_outputs;		// rknn_output outputs[io_num.n_output]; 
	rknn_input rknn_inputs[1];
	rknn_output rknn_outputs[4]; 
	
	// new 
	int task = TASK::GeneralDetection;

	// rknn model key params
	int nc;   			// number of classes, compute automaticlly!
	int height;			// model input height size, compute automaticlly!    @jamjon (448, 800) better and faster than (640, 640); 
	int width;			// model input width size, compute automaticlly! 
	int ch;   			// number of channels, compute automaticlly! 

	// functions
	void Init(std::string f, uint32_t rknn_init_flag);  // load model and get model params
	// virtual void IOConfig();	
	void Infer();	
	virtual void Release();		// release rknn ctx & free mem
};



// yolov5 model.
class YOLOv5: public RKNN {

public:

	// rknn init & default anchor
	YOLOv5(std::string f, uint32_t rknn_init_flag, int task) {
		// choose anchors for different tasks
		this->task = task;

		// rknn init
		RKNN::Init(f, rknn_init_flag);		

		
		if (this->task == TASK::FaceWithLandMark) {		// FaceWithLandMark
			std::cout << "> Task: FaceDetectingWithLandMark\n";	
			this->anchors = { {4,5,  8,10,  13,16},
							   {23,29,  43,55,  73,105}, 
							   {146,217,  231,300,  335,433} };

		} else if (this->task == TASK::GeneralDetection) {
			if (this->io_num.n_output == 3) {			
				std::cout << "> Task: General Detection(P3, p4, P5)\n";	
				this->anchors = { {10, 13, 16, 30, 33, 23},
							   		{30, 61, 62, 45, 59, 119}, 
							   		{116, 90, 156, 198, 373, 326} };
			}	else if (this->io_num.n_output == 4) {	
				std::cout << "> Task: General Detection(P3, p4, P5, P6)\n";	
				this->anchors = { {19, 27, 44, 40, 38, 94},
								   {96, 68, 86, 152, 180, 137}, 
								   {140, 301, 303, 264, 238, 542}, 
								   {436, 615, 739, 380, 925, 792} };
			} 

		} else if (this->task == TASK::HumanPose2D) {
			if (this->io_num.n_output == 3) {			
				std::cout << "> Task: Human Pose-2D(P3, p4, P5)\n";	
				this->anchors = { {10, 13, 16, 30, 33, 23},
							   		{30, 61, 62, 45, 59, 119}, 
							   		{116, 90, 156, 198, 373, 326} };
			} else if (this->io_num.n_output == 4) {
				std::cout << "> Task: Human Pose-2D(P3, p4, P5, P6)\n";	
				this->anchors = { {19, 27, 44, 40, 38, 94},
								   {96, 68, 86, 152, 180, 137}, 
								   {140, 301, 303, 264, 238, 542}, 
								   {436, 615, 739, 380, 925, 792} };
			}
		} 

		

		std::cout << "> Anchors: ";
		this->_getAnchorsInfo();
	}

	// constructor with anchors & rknn init
	YOLOv5(std::string f, uint32_t rknn_init_flag, std::vector<std::array<int, 6>>& anchors, int task): anchors(anchors) {
		this->task = task;
		RKNN::Init(f, rknn_init_flag);		// rknn init
		std::cout << "> Using customized anchors: ";
		_getAnchorsInfo();
	};

	// destructor
	~YOLOv5() {
		this->Release();
	};

	/* 	About Tracking   */
	// int TRACKER_BUFFER = 30;
	// int source_fps = -1;		
	// BYTETracker tracker = BYTETracker(source_fps, TRACKER_BUFFER);		// create object tracker

	std::vector<BBOX> detections;  					// bboxes for saving detection results
	std::vector<STrack> stracks;  					// tracked tracker
	std::vector<STrack> stracks_all;  				// lost_tracks + tracked tracks
	std::unordered_map<int, std::pair<std::vector<cv::Point>, bool>> trajectories;  // trajectory  ==>  {tracker_id: <[(cx, cy), ...], state>}
	//TODO: std::unordered_map<int, int> tracks_lasts;		//  {id: time_lasts}, for others use
	// std::vector<TRAJECTORY> trajectory;
	// int num_passed_line = 0;  // countint

	void Detect(cv::Mat& frame, float conf_thresh, float nms_thresh);  // Detect
	void Track(BYTETracker& tracker, cv::Mat& frame, bool enable_trajectory=true);	// Track
	void Draw(cv::Mat& img);	// draw results
	void Release();		// release rknn model
	
	
private:
	// vars for input pre-process
	std::array<int, 4> padding = {0, 0, 0, 0};  //  top, left, bottom, right
	float scale4lb = 1.0f;						// scale for letterbox
	std::vector<std::array<int, 6>> anchors;	// anchors

	void _PreProcess(cv::Mat& image_in, cv::Mat& image_out);		// pre-process
	void _Decode(float* input, std::array<int, 6>& anchor, int stride, float threshold, std::vector<BBOX>& bboxes);	// de-qnt & get bboxes from rknn output
	void _NonMaxSuppression(float threshold, std::vector<BBOX>& bboxes);	// NMS
	void _PostProcess(float conf_thresh, float nms_thresh); // post-process 
	void _xyScaleCoords(float& x, float& y) {		// de-leterbox & de-scaled
		x -= this->padding[1];
		y -= this->padding[0];
		x = std::min(std::max(0.f, x), (float)this->width) / this->scale4lb;		// or: clamp(x1 / c_scale4lb, 0, image_width)
		y = std::min(std::max(0.f, y), (float)this->height) / this->scale4lb;	
		
	}
	void _whScaleCoords(float& w, float& h) {		// de-scaled
		w = std::min(std::max(0.f, w), (float)this->width) / this->scale4lb;		// or: clamp(x1 / c_scale4lb, 0, image_width)
		h = std::min(std::max(0.f, h), (float)this->height) / this->scale4lb;	
	}

	float _bboxes_iou(BBOX& a, BBOX& b) {
		// xywh
		float w = std::max(0.f, std::min(a.rect.x + a.rect.width, b.rect.x  + b.rect.width) - std::max(a.rect.x, b.rect.x) + 1.0f);
		float h = std::max(0.f, std::min(a.rect.y + a.rect.height, b.rect.y + b.rect.height) - std::max(a.rect.y, b.rect.y) + 1.0f);
		float intersection = w * h;	// intersection
		float union_ = (a.rect.width + 1.0f) * (a.rect.height + 1.0f) + (b.rect.width + 1.0f) * (b.rect.height + 1.0f) - intersection;	// union
		return union_ <= 0.f ? 0.f : (intersection / union_);		
	}
	cv::Scalar _get_color(int idx);		// get color 
	float _sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }	// sigmoid 
	void _getAnchorsInfo() const {	// get anchors model used
		std::cout << "{ ";
		for (int i = 0; i < anchors.size(); ++i)
			for (int j = 0; j < anchors[i].size(); ++j) {
				std::cout << anchors[i][j] << " ";
			}
		printf("} (%ld x %ld)\n", anchors.size(), anchors[0].size());
	}



};




