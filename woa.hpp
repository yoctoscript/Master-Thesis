#ifndef WOA_HPP
#define WOA_HPP
#include "objects.hpp"
#include "logger.hpp"
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

class WOA
{
    public:
        WOA(Path path)
        {
            this->path = path;
        }
        Path optimize(Path);
    private:
        Path path;
        int population;
        int iterations;
        double explorationFactor;
        double SpiralParameter;
        double probabilityParameter;
        double lowerBound;
        double upperBound;
};


#endif