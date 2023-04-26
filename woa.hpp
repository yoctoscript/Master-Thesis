#ifndef WOA_HPP
#define WOA_HPP
#include "objects.hpp"
#include "logger.hpp"
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include "debug.hpp"

extern nlohmann::json mySettings;

class WOA
{
    public:
        WOA(State sNear, State sRand, State sGoal)
        {
            #ifdef DEBUG
                static Logger log(__FUNCTION__);
            #endif
            this->sNear = sNear;
            this->sRand = sRand;
            this->sGoal = sGoal;
            this->_a = mySettings["WOA_a"];
            this->a = this->_a;
            this->b = mySettings["WOA_b"];
            this->population = mySettings["WOA_population"];
            this->_iterations = mySettings["WOA_iterations"];
            this->iterations = this->_iterations;
            this->minLinearVelocity = mySettings["WOA_min_linear_velocity"];
            this->maxLinearVelocity = mySettings["WOA_max_linear_velocity"];
            this->minAngularVelocity = mySettings["WOA_min_angular_velocity"];
            this->maxAngularVelocity = mySettings["WOA_max_angular_velocity"];
            #ifdef DEBUG
                log.trace("a: {:.2f} | population: {} | iterations: {}", this->_a, this->population, this->iterations);
            #endif
        }
        State Apply(); /// Main function.
        /// Fields.
        State sNear; /// Near state.
        State sRand; /// Random state.
        State sGoal; /// Goal state.
        State sBest; /// Best state.
        int i;
        int population;
        int _iterations; /// Constant
        int iterations;
        long double _a; /// Constant
        long double a;
        long double r;
        long double A;
        long double C;
        long double p;
        long double l;
        long double b;
        long double minLinearVelocity;
        long double maxLinearVelocity;
        long double minAngularVelocity;
        long double maxAngularVelocity;
        State* whales; /// Array of particles.
        /// Methods.
        void InitializePopulation();
        void CalculateFitness();
        void CoefficientUpdate();
        void CircleUpdate();
        void SpiralUpdate();
        void RandomUpdate();
        void CheckBoundary();
        void RenderParticles();
        void CleanUp();
        /// Helper methods.
        long double GenerateRandom(long double a, long double b);
        int GenerateRandom(int a, int b);
        long double CalculateEuclideanDistance(long double& x, long double& y, long double& a, long double& b);
        long double CalculateDistance(long double& v);
        long double CalculateAngle(long double& w);
        int XConvertToPixel(long double& x);
        int YConvertToPixel(long double& y);
        long double NormalizeAngle(long double angle);
        bool IsObstacleFree(State& sNew);
        bool DoIntersect(cv::Point& p1, cv::Point& q1, cv::Point& p2, cv::Point& q2);
        int Orientation(cv::Point& p, cv::Point& q, cv::Point& r);
        bool OnSegment(cv::Point& p, cv::Point& q, cv::Point& r);



};
#endif