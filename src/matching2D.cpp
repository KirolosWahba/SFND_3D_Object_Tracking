#include <numeric>
#include "matching2D.hpp"

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {

        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        std::cout << "BF matching. cross-check=" << crossCheck << std::endl;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        {   
            // OpenCV bug workaround : convert binary descriptors to floating point...
            // due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::FlannBasedMatcher::create();
        std::cout << "FLANN matching" << std::endl;
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        double t = (double)cv::getTickCount();

        // Finds the best match for each descriptor in desc1
        matcher->match(descSource, descRef, matches);

        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << std::endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        std::vector<std::vector<cv::DMatch>> matchesPair;
        double t = (double)cv::getTickCount();

        matcher->knnMatch(descSource, descRef, matchesPair, 2);

        // Filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for( auto match : matchesPair ){
            if(match.at(0).distance/match.at(1).distance <= minDescDistRatio) 
                matches.emplace_back(match.at(0));
        }

        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << " (KNN & Distance Ratio Filtering) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << std::endl;
        std::cout << "# keypoints removed = " << matchesPair.size() - matches.size() << std::endl;          
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else if(descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if(descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << std::endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / std::max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;

    // visualize results
    if (bVis)
    {
        visualizekeypoints(keypoints, img, "Shi-Tomasi Corner Detection Results");
    }
}

// Detect keypoints in image using the traditional Harris detector
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);

    auto startTime = std::chrono::steady_clock::now();

    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled); // Scales, calculates absolute values, and converts the result to 8-bit
    

    float IoU_Threshold = 0;
    for(int rows=0; rows<dst_norm.rows; rows++)
    {
        for(int cols=0; cols<dst_norm.cols; cols++)
        {    
            int pixelIntensity = dst_norm.at<float>(rows,cols);
            
            if(pixelIntensity > minResponse)
            {
                cv::KeyPoint tempKeyPoint;
                tempKeyPoint.pt = cv::Point2f(cols,rows);
                tempKeyPoint.response = pixelIntensity;
                tempKeyPoint.size = 2*apertureSize;

                // NMS procedure 
                bool isOverlapOccurred = false;
                for(auto it = keypoints.begin(); it != keypoints.end(); it++)
                {
                    float overlapArea = cv::KeyPoint::overlap(tempKeyPoint, *it);

                    if(overlapArea > IoU_Threshold)
                    {   
                        isOverlapOccurred = true;
                        if(tempKeyPoint.response > it->response)
                            *it = tempKeyPoint; // Replacing the old keypoint with the new keypoint due to higher intensity of the new keypoint 
                    }
                }

                if(! isOverlapOccurred )
                    keypoints.push_back(tempKeyPoint);
            }
        }
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Harris Corner KeyPoint Detection Took: " << elapsedTime.count() << " milliseconds" << std::endl;

    // visualize results
    if (bVis)
    {
        visualizekeypoints(keypoints, dst_norm_scaled, "Harris Corner Detection Results");
    }
}

void detKeypointsSIFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();

    cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();
    detector->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "SIFT detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;
    
    // visualize results
    if (bVis)
    {
        visualizekeypoints(keypoints, img, "SIFT Detection Results");
    }
}

void detKeypointsFAST(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();

    cv::Ptr<cv::FastFeatureDetector> detector=cv::FastFeatureDetector::create();
    detector->detect(img,keypoints);
    
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "FAST detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;

    // visualize results
    if (bVis)
    {
        visualizekeypoints(keypoints, img, "FAST Detection Results");
    }
}

void detKeypointsBRISK(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();

    cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
    detector->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "BRISK detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;

    // visualize results
    if (bVis)
    {
        visualizekeypoints(keypoints, img, "BRISK Detection Results");
    } 
}

void detKeypointsORB(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    detector->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "ORB detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;
    
    // visualize results
    if (bVis)
    {
        visualizekeypoints(keypoints, img, "ORB Detection Results");
    } 
}

void detKeypointsAKAZE(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();

    cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
    detector->detect(img, keypoints);
    
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "AKAZE detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;

    // visualize results
    if (bVis)
    {
        visualizekeypoints(keypoints, img, "AKAZE Detection Results");
    }  
}

void visualizekeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string windowName)
{
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    imshow(windowName, visImage);
    cv::waitKey(0);
}