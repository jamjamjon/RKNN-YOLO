/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <iostream>
#include <string>
#include <chrono>
// #include "omp.h"
#include "yolov5.h"





/*-------------------
	Main Functions
---------------------*/
int main(int argc, char **argv)
{
	
	// [optional] for testing
    if (argc < 5) {
        printf("> Usage: %s <task[0:Detection, 1:FaceLandmarks]> <rknn model> <image/video> conf_thresh nms_thresh[optional]\n", argv[0]);
        return -1;
    }

    // argvs
    int MODEL_TASK = std::stoi(argv[1]);
    char* RKNN_WEIGHTS = (char *)argv[2];
    std::string SOURCE = argv[3];
    float CONF_THRESH = std::stof(argv[4]);
    float NMS_THRESH = 0.4;
    if (argv[5] != NULL) {
    	NMS_THRESH = std::stof(argv[5]);
    }

	// opencv deal with input source
	cv:: VideoCapture capture;	// capture instance
	capture.open(SOURCE);		// open video
	std::cout << "> Source Done Loading. | OpenCV Version: " << CV_VERSION << std::endl;
	
	// get source info
	double source_fps = capture.get(CV_CAP_PROP_FPS);
	int source_width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int source_height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	printf("> Source Size: width = %d, height = %d\n", source_width, source_height);
    long source_frames = static_cast<long>(capture.get(CV_CAP_PROP_FRAME_COUNT));
    std::cout << "> Source fps: " << source_fps << "  |  Total frames: " << source_frames << std::endl;

	// [optional] save video
	float VIDEO_SCALE = 1.0;
	cv::VideoWriter video_writer;
	// int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
	// video_writer.open("../out_video.mp4", fourcc, source_fps, cv::Size(int(source_width * VIDEO_SCALE), int(source_height * VIDEO_SCALE)), true);
	// printf("> Saving Video! Screen rescale rate = %.3f.\n", VIDEO_SCALE);

	// load model
	YOLOv5* model = new YOLOv5(RKNN_WEIGHTS, RKNN_FLAG_ASYNC_MASK, MODEL_TASK);   // RKNN_FLAG_PRIOR_HIGH, RKNN_FLAG_ASYNC_MASK
	// YOLOv5 model(RKNN_WEIGHTS, RKNN_FLAG_ASYNC_MASK, anchors);
	// YOLOv5* model = new YOLOv5(RKNN_WEIGHTS, RKNN_FLAG_ASYNC_MASK, TASK::FaceWithLandMark); 


	/* 	Tracking   */
	int TRACKER_BUFFER = 30;
	BYTETracker tracker = BYTETracker(source_fps, TRACKER_BUFFER);		// create object tracker
		
	// opencv loop
	if (capture.isOpened()) {
		cv::Mat frame;			// original frame  
		int num_frames = 0;		// number of frame
		int ms_each = 0;		// time each frame cost 
		int ms = 1;				// time accumulating


		int idx = 0;  // test
		// #pragma omp parallel for 
		for(;;) { 
			num_frames++;	// counting frame
			// capture >> frame;	// read frame  capture.read(frame)	
			// if (frame.empty()) break;
			if (!capture.read(frame)) {
			    capture.read(frame);
	        }

			
			auto start = chrono::system_clock::now();  			// start time
			model->Detect(frame, CONF_THRESH, NMS_THRESH);		// detect
			// model->Track(tracker, frame, false);				// tracking	enable_trajectory=true
			auto end = chrono::system_clock::now();   			// end time
			ms_each = chrono::duration_cast<chrono::microseconds>(end - start).count();
			ms += ms_each;		
			
			// calculate FPS
			putText(frame, 
					format("Frame: %d/%d, FPS: %d, Frame Cost: %.2f ms", num_frames, source_frames, num_frames * 1000000 / ms, float(ms_each) / 1000), 
    				Point(0, 30), 0, 0.6, Scalar(0, 255, 0), 2, 8);
    		
    		model->Draw(frame);  // draw result

			// cv::resize(frame, frame, cv::Size(int(source_width * VIDEO_SCALE), int(source_height * VIDEO_SCALE)), 0, 0, cv::INTER_LINEAR);  					
			cv::imshow("opencv_display", frame); 

			// video_writer << frame;  	// [optional] write frame to video
			// cv::imwrite(saveout, frame);  		
			
			if (cv::waitKey(1) == 'q') break;	// exit mannually

		}

		 
	} else {
		std::cout << "==> No video captured!" << std::endl;	
		exit(1);
	}


	video_writer.release();
	

	
}




