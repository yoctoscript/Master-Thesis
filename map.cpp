#include <opencv2/opencv.hpp>
#include "map.hpp"
#include "logger.hpp"

Segments Map::find_segments(){
    static Logger log(__FUNCTION__);

    /// Performs image thresholding on a grayscale image using a threshold value of 250 and creates a binary image.
    log.debug("threshold()");
    cv::Mat binary;
    threshold(image, binary, 250, 255, cv::THRESH_BINARY_INV);
    cv::imshow("Image", image);
    cv::imshow("Binary", binary);

    /// Performs morphological dilation on a binary image using a 3x3 rectangular structuring element, repeated 'i_obstacles_width' times.
    log.debug("dilate()");
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    for (int i = 0; i < dilation; ++i){
        dilate(binary, binary, kernel);
    }
    cv::imshow("Dilated", binary);

    /// Detects contours on a binary image using 'findContours()' function and stores the detected contours and their hierarchy in 'contours' and 'hierarchy' respectively.
    log.debug("findContours()");
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    log.debug("Contoured Objects: {}", contours.size());

    cv::Mat inverted_polygon_image = cv::Mat::zeros(height, width, CV_8UC1);
    Segments obstacles_segments;
    log.debug("approxPolyDP()");
    for (int i = 0; i < contours.size(); i++){

        /// Approximates a contour 'contours[i]' to a polygon using the 'approxPolyDP()' function with an epsilon value based on 'i_epsilon' times the contour's arc length. The result is stored in polygon.
        std::vector<cv::Point> polygon;
        double _epsilon = epsilon * arcLength(contours[i], true);
        approxPolyDP(contours[i], polygon, _epsilon, true);
        int polygon_size = polygon.size();

        /// Checks for a single point obstacle and creates a Segment object with its coordinates as start and end points. It then adds the Segment object to a vector of obstacle segments.
        if (polygon_size == 1){
            cv::Point a = polygon[0];
            Segment seg(a, a);
            obstacles_segments.push_back(seg);
            line(inverted_polygon_image, a, a, cv::Scalar(255), 1, cv::LINE_AA);
            log.trace("Obstacle Point ({}, {})",a.x, a.y);
        }

        /// Checks for a two-point obstacle and creates a Segment object with the two points as start and end points. It then adds the Segment object to a vector of obstacle segments.
        else if (polygon_size == 2){
            cv::Point a = polygon[0];
            cv::Point b = polygon[1];
            Segment seg(a, b);
            obstacles_segments.push_back(seg);
            line(inverted_polygon_image, a, b, cv::Scalar(255), 1, cv::LINE_AA);
            log.trace("Obstacle Segment ({}, {}) ({}, {})", a.x, a.y, b.x, b.y);

        }

        /// Checks if the polygon has more than 2 points. If so, it creates segments between each pair of adjacent points and adds them to a vector of obstacle segments.
        else if (polygon_size > 2){
            for (int j = 0; j < polygon_size; ++j){
                if (j == 0){
                    cv::Point b = polygon[polygon_size-1];
                    cv::Point a = polygon[0];
                    Segment seg(a, b);
                    obstacles_segments.push_back(seg);
                    line(inverted_polygon_image, a, b, cv::Scalar(255), 1, cv::LINE_AA);
                    log.trace("Obstacle Segment ({}, {}) ({}, {})", a.x, a.y, b.x, b.y);
                }
                else {
                    cv::Point a = polygon[j-1];
                    cv::Point b = polygon[j];
                    Segment seg(a, b);
                    obstacles_segments.push_back(seg);
                    line(inverted_polygon_image, a, b, cv::Scalar(255), 1, cv::LINE_AA);
                    log.trace("Obstacle Segment ({}, {}) ({}, {})", a.x, a.y, b.x, b.y);
                }
            }
        }
    }

    /// Show the result
    cv::Mat polygon_image;
    threshold(inverted_polygon_image, polygon_image, 128, 255, cv::THRESH_BINARY_INV);
    imshow("Polygon", polygon_image);
    log.debug("Total Segments: {}", obstacles_segments.size());
    return obstacles_segments;
}