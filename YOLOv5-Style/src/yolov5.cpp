#include "yolov5.h"



/* --------------------
 	RKNN parts
-----------------------*/ 

// init
void RKNN::Init(std::string f, uint32_t rknn_init_flag) {

	// step 1. load rknn model
	FILE* fp;
	int ret;

	fp = fopen(f.c_str(), "rb");
	if (NULL == fp) 	
		printf("Open file %s failed.\n", f.c_str());
	fseek(fp, 0, SEEK_END);

	int	rknn_model_size = ftell(fp);	// cancel rknn_model_size in model class
	ret = fseek(fp, 0, SEEK_SET);
	if (ret != 0)	
		printf("SEEK_SET failure.\n");
	
	unsigned char* rknn_model_data = (unsigned char*)malloc(rknn_model_size);	// allocate mem for rknn model	
	if (rknn_model_data == NULL)	
		printf("rknn model malloc failure.\n");
	
	ret = fread(rknn_model_data, 1, rknn_model_size, fp);	// read model
	fclose(fp);
	std::cout << "> RKNN Model Load.\n"; 	// info

	// rknn init
    ret = rknn_init(&this->ctx, rknn_model_data, rknn_model_size, rknn_init_flag);			// RKNN_FLAG_ASYNC_MASK 
    if (ret < 0)	
    	printf("rknn_init error ret=%d\n", ret);
	printf("> RKNN model init succeed.\n");

	// free memory(rknn model data) on the fly
	if (rknn_model_data)	
		free(rknn_model_data);

	// Rknn input num & output num ==>  rknn_input_output_num 
	ret = rknn_query(this->ctx, RKNN_QUERY_IN_OUT_NUM, &this->io_num, sizeof(this->io_num));
	if (ret < 0)	
		printf("\nrknn_init error ret=%d\n", ret);
	printf("> Num_input: %d  |  Num_output: %d\n", this->io_num.n_input, this->io_num.n_output);

	// input attr ==> rknn_tensor_attr 
	rknn_tensor_attr input_attrs[this->io_num.n_input];
	for (int i = 0; i < this->io_num.n_input; i++) {
	    input_attrs[i].index = i;
	    ret = rknn_query(this->ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
	    if (ret < 0)	
	    	printf("\nrknn_query_rknn_tensor_attr input_attrs error ret=%d\n", ret);
		printf("==>[Input] : index=%d name=%s n_dims=%d dims=[%d, %d, %d, %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
			   	input_attrs[i].index, input_attrs[i].name, input_attrs[i].n_dims, input_attrs[i].dims[3], 
				input_attrs[i].dims[2], input_attrs[i].dims[1], input_attrs[i].dims[0], input_attrs[i].n_elems, 
			  	input_attrs[i].size, input_attrs[i].fmt, input_attrs[i].type, input_attrs[i].qnt_type, 
			   	input_attrs[i].fl, input_attrs[i].zp, input_attrs[i].scale); 	
	}
		
	// output attr
	rknn_tensor_attr output_attrs[this->io_num.n_output];	
	for (int i = 0; i < this->io_num.n_output; i++) {
	    output_attrs[i].index = i;
	    ret = rknn_query(this->ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
	    if (ret < 0)
	    	printf("\nrknn_query_rknn_tensor_attr output_attrs error ret=%d\n", ret);
		printf("==>[Output]: index=%d name=%s n_dims=%d dims=[%d, %d, %d, %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
			   	output_attrs[i].index, output_attrs[i].name, output_attrs[i].n_dims, output_attrs[i].dims[3], 
				output_attrs[i].dims[2], output_attrs[i].dims[1], output_attrs[i].dims[0], output_attrs[i].n_elems, 
			  	output_attrs[i].size, output_attrs[i].fmt, output_attrs[i].type, output_attrs[i].qnt_type, 
			   	output_attrs[i].fl, output_attrs[i].zp, output_attrs[i].scale);   
		
	}	

	// get model num_class according to different task
	if (this->task == TASK::FaceWithLandMark) {
		this->nc = 1;
	} else if (this->task == TASK::GeneralDetection) {
		this->nc = output_attrs[0].dims[2] / 3 - 5;		// output_attrs[0].dims[2] = 3 * (5 + N)
	} else if (this->task == TASK::HumanPose2D) {
		this->nc = 1;
	}
	std::cout << "> Num_class: " << this->nc << "\n";	

	// rknn shape format 
	if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
	    this->width = input_attrs[0].dims[0];
		this->height = input_attrs[0].dims[1];
	    this->ch = input_attrs[0].dims[2];
	} else {
	    this->width = input_attrs[0].dims[1];
	    this->height = input_attrs[0].dims[2];
	    this->ch = input_attrs[0].dims[0];
	}
	printf("> Model Input Size: [%d, %d, %d]\n", this->height, this->width, this->ch);
	

	// Model input data setting
	// this->rknn_inputs = new rknn_input[this->io_num.n_input];		
	for (int i = 0; i < this->io_num.n_input; i++) {
		this->rknn_inputs[i].index = i;
		this->rknn_inputs[i].type = RKNN_TENSOR_UINT8;		// RKNN_TENSOR_UINT8
		this->rknn_inputs[i].size = input_attrs[i].size;    // this->width * this->height * this->num_channel; 
		this->rknn_inputs[i].fmt = RKNN_TENSOR_NHWC;
		this->rknn_inputs[i].pass_through = false;  		//  = 1 or True ==> it will give the data in inputs[0].buf directly to model's input node, and do nothing pre-process at all; // used with pre-compile in the process of converting to rknn model.(maybe)
	}

	// output sets 
	// TODO
	// this->rknn_outputs = new rknn_output[this->io_num.n_output];	
	memset(this->rknn_outputs, 0, sizeof(this->rknn_outputs)); 		// remove will cause problems 
	for (int i = 0; i < this->io_num.n_output; i++) {
		this->rknn_outputs[i].want_float = true;	//  use uint8_t type or float type
		// rknn_outputs[i].is_prealloc = false; 	// = 1, user decide where to allocate buffer and release Mem by himself; = 0, rknn auto mode. 
		// this->rknn_outputs[i].index = i;			
		// this->rknn_outputs[i].size = output_attrs[i].size;
		// std::cout << "index: " << this->rknn_outputs[i].index << " => " << "size: " << this->rknn_outputs[i].size << std::endl;
	}

}



// rknn run
void RKNN::Infer() { 
	int ret = rknn_run(this->ctx, NULL);		// rknn inference	 
	if (ret < 0) 
		std::cout << "rknn_run() error = " << ret << "\n";

	ret = rknn_outputs_get(this->ctx, this->io_num.n_output, this->rknn_outputs, NULL); 		// get model outputs
	if (ret < 0) 
		std::cout << "rknn_outputs_get() error = " << ret << "\n";
}


// rknn ctx destory
void RKNN::Release() {
	if (this->ctx > 0)	rknn_destroy(this->ctx);	// ctx
	// if (rknn_inputs)	
	// 	delete[] rknn_inputs;		// inputs
	// if (rknn_outputs)	
	// 	delete[] rknn_outputs;		// outputs

}


/* --------------------
 	YOLOv5 Parts
-----------------------*/

void YOLOv5::_PreProcess(cv::Mat& image_in, cv::Mat& image_out) {

	// resize 
    this->scale4lb = std::min((float)this->width / image_in.cols, (float)this->height / image_in.rows); 

    // TODO: if this->scale4lb > 0 { }
    if (this->scale4lb < 1) {
		cv::resize(image_in, image_out, cv::Size(), this->scale4lb, this->scale4lb, cv::INTER_AREA);
    } else {
    	image_out = image_in;
    }

	// cv::resize(image_in, image_out, cv::Size(), this->scale4lb, this->scale4lb, cv::INTER_AREA);

	// padding
	this->padding[0] = floor((this->height - image_out.size().height) / 2.0);		// top
	this->padding[1] = floor((this->width - image_out.size().width) / 2.0);			// left
	this->padding[2] = ceil((this->height - image_out.size().height) / 2.0);		// bottom
	this->padding[3] = ceil((this->width - image_out.size().width) / 2.0);			// right

	// Set to left-top
	// this->padding[0] = 0;		// top
	// this->padding[1] = 0;		// left
	// this->padding[2] = this->height - image_out.size().height;	// bottom
	// this->padding[3] = this->width - image_out.size().width;		// right

	// make boarder
	cv::copyMakeBorder(image_out, image_out, this->padding[0], this->padding[2], this->padding[1], this->padding[3], cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));	
	cv::cvtColor(image_out, image_out, CV_BGR2RGB); // BGR -> RGB

	// cv::imshow("padding", image_out);
	// cv::waitKey(0);

	// [rknn] inputs set, send images to rknn inputs;
	for (int i = 0; i < this->io_num.n_input; i++) {
		this->rknn_inputs[0].buf = image_out.data;			
	}
	rknn_inputs_set(this->ctx, this->io_num.n_input, this->rknn_inputs);

}


// for un-permute type => shape like [(1,255,80,80), (1,255,40,40), (1,255,20,20)]
void YOLOv5::_Decode(float* input, 
					 std::array<int, 6>& anchor,    // 6 for p3, p4, p5
					 int stride,
					 float threshold, 
					 std::vector<BBOX>& bboxes) {

	int grid_h = this->height / stride; 
	int grid_w = this->width / stride;
	int grid_len = grid_h * grid_w;
	
	// loop 
    for (int b = 0; b < 3; b++) {   		// every grid predict 3 bbox. 
        for (int i = 0; i < grid_h; i++) {			
            for (int j = 0; j < grid_w; j++) {		

            	// offset
                int offset;
                if (this->task == TASK::GeneralDetection) {
                 	offset = ((this->nc + 5) * b) * grid_len + i * grid_w + j;
                } else if (this->task == TASK::FaceWithLandMark) {
                 	offset = ((this->nc + 5 + 10) * b) * grid_len + i * grid_w + j;
                } else if (this->task == TASK::HumanPose2D) {	// update
                	offset = ((this->nc + 5 + 3*17) * b) * grid_len + i * grid_w + j;
                }

                float *pos = input + offset;	// current pos
                float conf_obj = this->_sigmoid(pos[4 * grid_len]);	// conf
                // float bbox_confidence = input[((c_num_class + 5) * b + 4) * grid_len + i * grid_w + j];    // shape [3*(5+80)(x,y,w,h,c(4),pr), 80, 80]
				
				// filter
                for (int class_idx = 0; class_idx < this->nc; class_idx++) {

                	// conf
                    float conf_cls;
					if (this->task == TASK::GeneralDetection || this->task == TASK::HumanPose2D) { 	// update
	                 	conf_cls = this->_sigmoid(pos[(5 + class_idx) * grid_len]);		// class prob
	                } else if (this->task == TASK::FaceWithLandMark) {
	                 	conf_cls = this->_sigmoid(pos[(15 + class_idx) * grid_len]);	// class prob

	                }

					// filter
                    float score = conf_obj * conf_cls;
                    if (score >= threshold ) {

						// x1x2y1y2
		 				float cx = (this->_sigmoid(*pos) * 2.0 - 0.5 + j) * stride;										// bx =2*sigmoid(tx) - 0.5 + cx,  tx = *pos
				        float cy = (this->_sigmoid(pos[grid_len]) * 2.0 - 0.5 + i) * stride;							// by =2*sigmoid(ty) - 0.5 + cy,  ty = pos[grid_len]               
				        float w = std::pow(this->_sigmoid(pos[2 * grid_len]) * 2.0, 2) * (float)anchor[b * 2];			// bw = pw(2*sigmoid(tw))^2
				        float h = std::pow(this->_sigmoid(pos[3 * grid_len]) * 2.0, 2) * (float)anchor[b * 2 + 1];		// bh = ph(2*sigmoid(th))^2
						float x1 = cx - w / 2.0;			// bbox.left
						float y1 = cy - h / 2.0;			// bbox.top

						this->_xyScaleCoords(x1, y1);
						this->_whScaleCoords(w, h);

						// temp bbox to save
			            BBOX bbox_temp;
			            bbox_temp.rect = cv::Rect_<float>(x1, y1, w, h);
			            bbox_temp.id = class_idx;
			            bbox_temp.score = score;


						/***************************************************
							[Human-Pose-2D] with 17 keypoints
							[format] => |x|y|w|h|conf|cls|x1|y1|conf1|x2|y2|conf2|...|x17|y17|conf17|
						******************************************************/
						if (this->task == TASK::HumanPose2D) {
							KEYPOINT kpt_temp;
							for (int k = 0; k < 17 * 3; k+=3) {
								kpt_temp.kpt.x = (pos[(k + 6) * grid_len] * 2.0 - 0.5 + j) * stride;
								kpt_temp.kpt.y = (pos[(k + 6 + 1) * grid_len] * 2.0 - 0.5 + i) * stride;
								kpt_temp.conf = this->_sigmoid(pos[(k + 6 + 2) * grid_len]);
								bbox_temp.kpts.emplace_back(kpt_temp);
							}
				   		    // de-letterbox
							for (int kk = 0; kk < bbox_temp.kpts.size(); kk++) 
								this->_xyScaleCoords(bbox_temp.kpts[kk].kpt.x, bbox_temp.kpts[kk].kpt.y);
						}

						/***************************************************
							landmark for [yolo-face] with 5 keypoints
							[format] =>  |x|y|w|h|conf|x1|y1|x2|y2|...|x5|y5|cls|
						******************************************************/
						if (this->task == TASK::FaceWithLandMark) {
							KEYPOINT kpt_temp;
							for (int k = 5; k < 5 * 3; k+=2) {
								kpt_temp.kpt.x = pos[k * grid_len] * (float)anchor[b * 2] + j * stride ;	
								kpt_temp.kpt.y = pos[(k + 1) * grid_len] * (float)anchor[b * 2 + 1] + i * stride;
								bbox_temp.kpts.emplace_back(kpt_temp);
							}
							// de-letterbox
							for (int kk = 0; kk < bbox_temp.kpts.size(); kk++) 
								this->_xyScaleCoords(bbox_temp.kpts[kk].kpt.x, bbox_temp.kpts[kk].kpt.y);
						}
						
						// save results
			            bboxes.emplace_back(bbox_temp);
                    }
                }
            }
        }
    }
}


// post-process
void YOLOv5::_PostProcess(float conf_thresh, float nms_thresh) {

	// clear before push
	this->detections.clear();
	
	// compute all valid detections & save all bboxes
	for (int i = 0; i < this->io_num.n_output; i++) {
		_Decode((float*)this->rknn_outputs[i].buf, this->anchors[i], std::pow(2, 3 + i), conf_thresh, this->detections);   // stride = std::pow(2, 3 + i): 8, 16, 32, (64)
	}

	// clear current frame results
	int ret = rknn_outputs_release(this->ctx, this->io_num.n_output, this->rknn_outputs);	
	if (ret < 0)	
		std::cout << "rknn_outputs_release() error!\n";

	// NMS
	_NonMaxSuppression(nms_thresh, this->detections);

	// bboxes info
	// for(int j=0; j < detections.size(); j++) {
	//  	printf("#bboxes [num %d] | id: [%d] %f | bbox: (%d, %d, %d, %d)\n", j, 
	//  			detections[j].id, detections[j].score, detections[j].rect.x, 
	//  			detections[j].rect.y, detections[j].rect.width, detections[j].rect.height);
	//  } 
}

//TODO: By class_id
void YOLOv5::_NonMaxSuppression(float threshold, std::vector<BBOX>& bboxes) {

	// printf("==>Object number after NMS: (%ld ==> ", bboxes.size());	
  	std::sort(bboxes.begin(), bboxes.end(), [](BBOX a, BBOX b) { return a.score > b.score; } );  // sort(>=) by score
    for (int i = 0; i < int(bboxes.size()); i++) {
        for (int j = i + 1; j < int(bboxes.size()); ) {
        	if (this->_bboxes_iou(bboxes[i], bboxes[j]) >= threshold) {
                bboxes.erase(bboxes.begin() + j);
            } else { j++; }
        }
    }
	// printf("%ld)\n", bboxes.size());
}



// pre-process + inference + post-process
void YOLOv5::Detect(cv::Mat& frame, float conf_thresh, float nms_thresh) {

	// #1. pre-process
	// auto start = chrono::system_clock::now();  			// start time
	cv::Mat image;		// do letterbox, for inference
	this->_PreProcess(frame, image);    
	// auto end = chrono::system_clock::now();  			// start time
	// int ms_pre = chrono::duration_cast<chrono::microseconds>(end - start).count();
	// printf("## pre_process each frame cost %d ms\n", ms_pre / 1000);

	// #2. inference
	RKNN::Infer(); 

	// #3. post-process
	// start = chrono::system_clock::now();  			// start time
	this->_PostProcess(conf_thresh, nms_thresh);   	 //	conf_thresh, nms_thresh 
	// end = chrono::system_clock::now();  			// start time
	// int ms_post = chrono::duration_cast<chrono::microseconds>(end - start).count();
	// printf("## post_process each frame cost %d ms\n", ms_post / 1000);
}


// track 
void YOLOv5::Track(BYTETracker& tracker, cv::Mat& frame, bool enable_trajectory) {

	// tracking 
	this->stracks = tracker.update(this->detections, this->stracks_all);
	
	// calculate trajectory ==>  {tracker_id: <[(cx, cy), ...], state>} 
	if (enable_trajectory) {
	
		// saving all tracks
		for (int i = 0; i < this->stracks_all.size(); i++) {
			// std::cout << "idx: " << this->stracks_all[i].track_id << " frame_id: " << this->stracks_all[i].frame_id << " state: " << this->stracks_all[i].state << " length: " << this->stracks_all[i].tracklet_len << std::endl;
			
			// calculate center point when tracked
			std::vector<cv::Point> center;
			float cx, cy;
			if (stracks_all[i].state == 1) {	// tracked: 1
				cx = stracks_all[i].tlwh[0] + stracks_all[i].tlwh[2] / 2;
				cy = stracks_all[i].tlwh[1] + stracks_all[i].tlwh[3] / 2;			
				center.emplace_back(cv::Point(cx, cy));	
			}
			
			// update state
			trajectories[stracks_all[i].track_id].second = stracks_all[i].state;	
			
			// generate trajectories
			if (trajectories.find(stracks_all[i].track_id) == trajectories.end()) { 	// not find track_id  ==>  create new one
				trajectories[stracks_all[i].track_id].first = center;
			} else {	// find track_id 
				if (stracks_all[i].state == 1) {	// lost & removed  ==>  update;  lost & removed  ==>  do nothing
					trajectories[stracks_all[i].track_id].first.emplace_back(cv::Point(cx, cy));	// update state
					
				} 
			}
		
		}
	}
	
}

// draw
void YOLOv5::Draw(cv::Mat& frame) {


	// 1.for tracking
	if (stracks.size() > 0) {
	
		// tracking labels * bboxes
		for (int i = 0; i < stracks.size(); i++) {
			vector<float> tlwh = stracks[i].tlwh;
			bool vertical = tlwh[2] / tlwh[3] > 1.6;
			// if (tlwh[2] * tlwh[3] > 20 && !vertical)   // min_bbox_area
			// {
				cv::putText(frame, 
							format("%d [%d] [%d]: %.2f", stracks[i].track_id, stracks[i].class_id, stracks[i].tracklet_len, stracks[i].score), 
							Point(tlwh[0], tlwh[1] - 5), 
			            	cv::FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 
			            	1, LINE_AA);
			    cv::rectangle(frame, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), 
			    		      this->_get_color(stracks[i].track_id), 2);
			// }
		}
		
		// draw trajecories
		if (trajectories.size() > 0) {
			for (auto trajectory: trajectories) {
				if (trajectory.second.second == 1) {	// tracked, not lost or removed
					for (auto it: trajectory.second.first) {	// trajectory.second  => <center_points,state>
						cv::circle(frame, it, 4, this->_get_color(trajectory.first), -1);
					}
				} 
			}
		}
		
			

	} else {	// 2.for detection


		for (std::vector<BBOX>::iterator it = this->detections.begin(); it != this->detections.end(); ++it) {

			// if ((it->rect.width <= 40) && (it->rect.height <= 40)) continue;	

			// detection 
			cv::rectangle(frame, it->rect.tl(), it->rect.br(), this->_get_color(it->id), 2);
			cv::putText(frame, 
						format("%d: %.2f", it->id, it->score), 						
						cv::Point(it->rect.tl().x, it->rect.tl().y - 5), 
					    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 
					    2, LINE_AA);	

			if (this->task == TASK::FaceWithLandMark) {

				for (int i = 0; i < it ->kpts.size(); i++) {
					if(!it->rect.contains(it->kpts[i].kpt))
						continue;
					cv::circle(frame, it->kpts[i].kpt, 3, cv::Scalar(0, 255, 255), -1);
				}

			}


			if (this->task == TASK::HumanPose2D) {
				
				// keypoints
				for (int i = 0; i < it->kpts.size(); i++) {
					if (it->kpts[i].conf < 0.5)
						continue;
					if (!it->rect.contains(it->kpts[i].kpt))
						continue;

					cv::circle(frame, it->kpts[i].kpt, 3, cv::Scalar(0, 255, 255), -1);
					// kpt num order
					// cv::putText(frame, format("%d", i), 						
					// 			cv::Point(it->kpts[i].kpt.x, it->kpts[i].kpt.y - 3), 
					// 		    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 
					// 		    1, LINE_AA);
				}

				// skeleton
				std::vector<std::array<int, 2>> skeleton = {
						{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12},
						{7, 13}, {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11}, {1, 2}, 
						{1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7},	
						{2, 3},  // left -> right eye
				};
				for (int i = 0; i < skeleton.size(); i++) {
					if (it->kpts[skeleton[i][0]-1].conf < 0.5 || it->kpts[skeleton[i][1]-1].conf < 0.5)
						continue;
					if ((!it->rect.contains(it->kpts[skeleton[i][0]-1].kpt)) || (!it->rect.contains(it->kpts[skeleton[i][1]-1].kpt)))
						continue;
					cv::line(frame, it->kpts[skeleton[i][0]-1].kpt, it->kpts[skeleton[i][1]-1].kpt, this->_get_color(i+232), 2);
				}
				
			}
			
		}

	}

}


// generate color
cv::Scalar YOLOv5::_get_color(int idx)
{
	idx += 3;
	return cv::Scalar(137 * idx % 255, 77 * idx % 255, 129 * idx % 255);
}


// release
void YOLOv5::Release() {

	// release rknn
	RKNN::Release();

	// others mem 
	std::cout << "==>YOLOv5 released!" << std::endl;

}




















