#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"

//using namespace std;
using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    namedWindow("Dallin Briggs",1);

    int exit_key = 0;
    VideoWriter VOut; // Create a video write object.
    // Initialize video write object (only done once). Change frame size to match your camera resolution.
    VOut.open("VideoOut.avi", CV_FOURCC('M', 'P', 'E', 'G') , 30, Size(640, 480), 0);

    while(1)
    {
        Mat edges;
        Mat current_frame;
        Mat previous_frame;
        Mat frame_gray;
        Mat frame_thresh;
        Mat cdst;
        Mat motion;



        switch(waitKey(1) & 0xFF){
        case '1':
            while(1){
                cap >> current_frame; // get a new current_frame from camera
                cvtColor(current_frame,frame_gray,COLOR_BGR2GRAY);
                imshow("Briggs", current_frame);
                VOut << frame_gray;
                if(waitKey(1) >= 0)
                    break;
            }
            break;
        case '2':
            while(1){
                cap >> current_frame; // get a new current_frame from camera
                cvtColor(current_frame, edges, COLOR_BGR2GRAY);
                GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
                Canny(edges, edges, 0, 30, 3);
                imshow("Briggs", edges);
                VOut << edges;
                if(waitKey(1) >= 0)
                    break;
            }
            break;
        case '3':
            while(1){
                cap >> current_frame;
                cvtColor(current_frame, frame_gray, CV_BGR2GRAY);
                threshold(frame_gray, frame_thresh,100,255,0);
                imshow("Briggs", frame_thresh);
                VOut << frame_thresh;
                if(waitKey(1) >= 0)
                    break;
            }
            break;
        case '4':
            while(1){
                std::vector< cv::Point2f > frame_corners;
                /// Detector parameters
                int blockSize = 3;
                int apertureSize = 3;
                double k = 0.04;
                int thresh = 170;
                Mat copy;

                cap >> current_frame;
                cvtColor(current_frame, frame_gray, CV_BGR2GRAY);
                copy = current_frame.clone();
                goodFeaturesToTrack(frame_gray,frame_corners, 120, .01, 10, Mat(), blockSize, false, .04);

                /// Drawing a circle around corners
                for( size_t i = 0; i < frame_corners.size(); i++ )
                {
                    cv::circle(frame_gray, frame_corners[i], 10, cv::Scalar( 255. ), -1 );
                }

                imshow("Briggs", frame_gray);
                VOut << frame_gray;
                if(waitKey(1) >= 0)
                    break;
            }
            break;
        case '5':
            while(1){
                cap >> current_frame; // get a new current_frame from camera
                cvtColor(current_frame, edges, COLOR_BGR2GRAY);
                GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
                Canny(edges, edges, 0, 30, 3);
                cvtColor(edges, cdst, CV_GRAY2BGR);

                vector<Vec4i> lines;
                HoughLinesP(edges, lines, 1, CV_PI/180, 50, 50, 10 );
                for( size_t i = 0; i < lines.size(); i++ )
                {
                    Vec4i l = lines[i];
                    line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
                }

                cvtColor(cdst, cdst, CV_BGR2GRAY);
                imshow("source", current_frame);
                imshow("detected lines", cdst);
                VOut << cdst;
                if(waitKey(1) >= 0)
                    break;
            }
            break;

        case '6':
            while(1){
                frame_gray.copyTo(previous_frame);
                cap >> current_frame; // get a new current_frame from camera
                cvtColor(current_frame,frame_gray,COLOR_BGR2GRAY);
                if(previous_frame.empty())
                {
                    previous_frame = Mat::zeros(frame_gray.size(), frame_gray.type()); // prev frame as black
                    //signed 16bit mat to receive signed difference
                }

                absdiff(frame_gray, previous_frame, motion);

                imshow("Briggs", motion);
                VOut << motion;
                if(waitKey(1) >= 0)
                    break;

            }
            break;
        case '7':
            exit_key = 1;
            break;
        }

        if(exit_key == 1)
            break;

    }
    return 0;
}
