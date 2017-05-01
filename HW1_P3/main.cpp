#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    Mat image;
    Mat current_frame;
    Mat previous_frame;
    Mat frame_thresh;
    Mat motion;
    Mat first_frame;
    Mat image_color;
    string filename;
    string header;
    string tail;
    vector<Vec3f> circles;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;


    VideoWriter VOut; // Create a video write object.
    // Initialize video write object (only done once). Change frame size to match your camera resolution.
    VOut.open("VideoOut.avi", CV_FOURCC('M', 'P', 'E', 'G') , 30, Size(640, 480), 1);

    //    namedWindow("Dallin Briggs",1);
    //    SimpleBlobDetector blobby;

    // Setup SimpleBlobDetector parameters.
    SimpleBlobDetector::Params params;

    // Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 255;

    // Filter by Area.
    params.filterByArea = true;
    params.minArea = 100;

    // Filter by Circularity
    params.filterByCircularity = true;
    params.minCircularity = 0.1;

    // Filter by Convexity
    params.filterByConvexity = true;
    params.minConvexity = 0.87;

    // Filter by Inertia
    params.filterByInertia = true;
    params.minInertiaRatio = .3;

#if CV_MAJOR_VERSION < 3   // If you are using OpenCV 2

    // Set up detector with params
    SimpleBlobDetector detector(params);

    // You can use the detector this way
    // detector.detect( im, keypoints);

#else

    // Set up detector with params
    Ptr<SimpleBlobDetector> blobby = SimpleBlobDetector::create(params);

    // SimpleBlobDetector::create creates a smart pointer.
    // So you need to use arrow ( ->) instead of dot ( . )
    // detector->detect( im, keypoints);

#endif


    for(int i = 14; i < 46; i++)
    {
        header = "/home/dallin/robotic_vision/HW4/ImagesDallin/Ball_testL";
        tail = ".bmp";
        filename = header + to_string(i) + tail;

        first_frame = imread(header + "14" + tail, CV_LOAD_IMAGE_GRAYSCALE);
        image = imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
        image_color = imread(filename,CV_LOAD_IMAGE_COLOR);

        current_frame.copyTo(previous_frame);

        current_frame = image;
        if(previous_frame.empty())
        {
            previous_frame = Mat::zeros(current_frame.size(), current_frame.type()); // prev frame as black
            //signed 16bit mat to receive signed difference
        }
        absdiff(current_frame, first_frame, motion);
        GaussianBlur(motion,motion, Size(11,11), 0,0);
        threshold(motion, frame_thresh,4,255,1);
        //        HoughCircles(frame_thresh,circles,CV_HOUGH_GRADIENT,1,20,100,5,4,10);


        // Detect blobs.
        std::vector<KeyPoint> keypoints;
        //        blobby->detect(frame_thresh, keypoints);

        // Draw detected blobs as red circles.
        // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
        Mat im_with_keypoints;
        //        drawKeypoints(current_frame, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

        findContours(frame_thresh,contours,hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
//        drawContours(current_frame,contours,0,Scalar(255,255,255),2,LINE_8,hierarchy,0x7fffffff,Point(0,0));
        vector<Moments> contour_moments(contours.size());
        for( int i = 0; i < contours.size(); i++ )
        { contour_moments[i] = moments( contours[i], true ); }

        ///  Get the mass centers:
        vector<Point2f> mc( contours.size() );
        for( int i = 0; i < contours.size(); i++ )
        { mc[i] = Point2f( contour_moments[i].m10/contour_moments[i].m00 , contour_moments[i].m01/contour_moments[i].m00 ); }

        for(int i=0; i<contours.size(); i++)
        {
            circle(current_frame,mc[i],8,Scalar(255,255,255),2,LINE_8,0);
        }

        cout << contours.size() << endl;
        //        for( size_t i = 0; i < circles.size(); i++ )
        //        {
        //           Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        //           int radius = cvRound(circles[i][2]);
        //           // circle center
        //           circle( frame_thresh, center, 3, Scalar(0,255,0), -1, 8, 0 );
        //           // circle outline
        //           circle( frame_thresh, center, radius, Scalar(0,0,255), 3, 8, 0 );
        //         }
        // Show blobs


        imshow("keypoints", current_frame );
        //        VOut << im_with_keypoints;
        //        imshow("Briggs", frame_circles);

        waitKey(500);

    }

    return 0;
}
