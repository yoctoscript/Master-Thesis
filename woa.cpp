#include "woa.hpp"
#include "map.hpp"
#include "logger.hpp"
#include "objects.hpp"
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <random>
#include "debug.hpp"

extern nlohmann::json mySettings;
extern Map myMap;
extern cv::VideoWriter* myVideo;

State WOA::Apply()
{
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
        log.debug("InitializePopulation()");
    #endif
    InitializePopulation();
    #ifdef DEBUG
        log.debug("CalculateFitness()");
    #endif
    CalculateFitness();
    while (this->iterations--)
    {
        for (this->i = 0; this->i < this->population; ++(this->i))
        {
            #ifdef DEBUG
                log.debug("CoefficientUpdate()");
            #endif
            CoefficientUpdate();
            if ((this->p) < 0.5L)
            {
                if (fabs(this->A) < 0.5L)
                {
                    #ifdef DEBUG
                        log.debug("CircleUpdate()");
                    #endif
                    CircleUpdate();
                }
                else
                {
                    #ifdef DEBUG
                        log.debug("RandomUpdate()");
                    #endif
                    RandomUpdate();
                }
            }
            else
            {
                #ifdef DEBUG
                    log.debug("SpiralUpdate()");
                #endif
                SpiralUpdate();
            }
        }
        #ifdef DEBUG
            log.debug("CheckBoundary");
        #endif
        CheckBoundary();
        #ifdef DEBUG
            log.debug("CalculateFitness");
        #endif
        CalculateFitness();
        #ifdef DEBUG
            log.debug("RenderParticles()");
        #endif
        //RenderParticles();
    }
    #ifdef DEBUG
        log.debug("CleanUp()");
    #endif
    CleanUp();
    return this->sBest;
}

void WOA::InitializePopulation()
{
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
    #endif
    this->whales = new State[this->population];
    for (int i = 0; i < this->population; ++i)
    {
        long double linearVelocity = GenerateRandom(this->minLinearVelocity, this->maxLinearVelocity);
        long double angularVelocity = GenerateRandom(this->minAngularVelocity, this->maxAngularVelocity);
        this->whales[i].v = linearVelocity;
        this->whales[i].w = angularVelocity;
        #ifdef DEBUG
            log.trace("Whale {} (Linear: {:.2f}, Angular: {:.2f})", i, this->whales[i].v, this->whales[i].w);
        #endif
    }
}

void WOA::CalculateFitness()
{
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
    #endif
    double long bestFitness = std::numeric_limits<long double>::max();
    long double fitness;
    long double x, y, z;
    long double distanceToRand, distanceToGoal, isObstacleFree;
    long double w1 = mySettings["WOA_random_weight"];
    long double w2 = mySettings["WOA_goal_weight"];
    int bestIndex = 0;
    for (int i = 0; i < this->population; ++i)
    {
        x = this->whales[i].x = (this->sNear.x) - CalculateDistance(this->whales[i].v)*sin(this->sNear.z + CalculateAngle(this->whales[i].w)/2.0l);
        y = this->whales[i].y = (this->sNear.y) + CalculateDistance(this->whales[i].v)*cos(this->sNear.z + CalculateAngle(this->whales[i].w)/2.0l);
        z = this->whales[i].z = (this->sNear.z) + CalculateAngle(this->whales[i].w);
        distanceToRand = CalculateEuclideanDistance(this->whales[i].x, this->whales[i].y, sRand.x, sRand.y);
        distanceToGoal = CalculateEuclideanDistance(this->whales[i].x, this->whales[i].y, sGoal.x, sGoal.y);
        fitness = w1 * distanceToRand + w2 * distanceToGoal; // to do (add more factors)(angle nominator: cos(alpha))
        if (fitness < bestFitness)
        {
            bestFitness = fitness;
            bestIndex = i;
        }
        #ifdef DEBUG
            log.trace("Whale {} (x: {:.2f}, y:{:.2f}, z:{:.2f}) | Fitness: {:.2f}", i, x, y, z, fitness);
        #endif
    }
    this->sBest = this->whales[bestIndex];
    #ifdef DEBUG
        log.trace("Best Whale: {}, Fitness: {:.2f}", bestIndex, bestFitness);
    #endif
}

void WOA::CoefficientUpdate()
{
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
    #endif
    this->a -= (this->_a / (long double)(this->_iterations * this->population));
    this->r = GenerateRandom(0.0l, 1.0l);
    this->A = (2 * this->a * this->r) - this->a;
    this->C = (2 * this->r);
    this->l = GenerateRandom(-1.0l, 1.0l);
    this->p = GenerateRandom(0.0l, 1.0l);
    #ifdef DEBUG
        log.trace("a: {:.3f} | r: {:.2f} | A: {:.2f} | C: {:.2f} | l: {:.2f} | p: {:.2f} | b: {:.2f}", this->a, this->r, this->A, this->C, this->l, this->p, this->b);
    #endif
}

void WOA::CircleUpdate()
{
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
        long double oldV = this->whales[this->i].v;
        long double oldW = this->whales[this->i].w;
    #endif
    long double Dv = fabs((this->C * this->sBest.v) - this->whales[this->i].v);
    long double Dw = fabs((this->C * this->sBest.w) - this->whales[this->i].w);
    this->whales[this->i].v = this->sBest.v - (this->A * Dv);
    this->whales[this->i].w = this->sBest.w - (this->A * Dw);
    #ifdef DEBUG
    long double newV = this->whales[this->i].v;
    long double newW = this->whales[this->i].w;
        log.trace("Whale {} (Linear: {:.2f} => {:.2f})(Angular: {:.2f} => {:.2f})", this->i, oldV, newV, oldW, newW);
    #endif
}

void WOA::RandomUpdate()
{
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
        long double oldV = this->whales[this->i].v;
        long double oldW = this->whales[this->i].w;
    #endif
    int index = GenerateRandom(0, this->population);
    State sRand = this->whales[index];
    long double Dv = fabs((this->C * sRand.v) - this->whales[this->i].v);
    long double Dw = fabs((this->C * sRand.w) - this->whales[this->i].w);
    this->whales[this->i].v = sRand.v - (this->A * Dv);
    this->whales[this->i].w = sRand.w - (this->A * Dw);
    #ifdef DEBUG
        long double newV = this->whales[this->i].v;
        long double newW = this->whales[this->i].w;
        log.trace("Whale {} (Linear: {:.2f} => {:.2f})(Angular: {:.2f} => {:.2f})", this->i, oldV, newV, oldW, newW);
    #endif
}

void WOA::SpiralUpdate()
{
    #ifdef DEBUG
        static Logger log(__FUNCTION__);
        long double oldV = this->whales[this->i].v;
        long double oldW = this->whales[this->i].w;
    #endif
    long double Dv = fabs(this->sBest.v - this->whales[this->i].v);
    long double Dw = fabs(this->sBest.w - this->whales[this->i].w);
    this->whales[this->i].v = Dv * exp(b*l) * cos(2* M_PI * l) + whales[this->i].v;
    this->whales[this->i].w = Dv * exp(b*l) * cos(2* M_PI * l) + whales[this->i].w;
    #ifdef DEBUG
        long double newV = this->whales[this->i].v;
        long double newW = this->whales[this->i].w;
        log.trace("Whale {} (Linear: {:.2f} => {:.2f})(Angular: {:.2f} => {:.2f})", this->i, oldV, newV, oldW, newW);
    #endif
}

void WOA::CheckBoundary()
{
    static Logger log(__FUNCTION__);
    for (int i = 0; i < this->population; ++i)
    {
        if (this->whales[i].v > this->maxLinearVelocity)
        {
            this->whales[i].v = this->maxLinearVelocity;
        }
        else if (this->whales[i].v < this->minLinearVelocity)
        {
            this->whales[i].v = this->minLinearVelocity;
        }
        if (this->whales[i].w > this->maxAngularVelocity)
        {
            this->whales[i].w = this->maxAngularVelocity;
        }
        else if (this->whales[i].w < this->minAngularVelocity)
        {
            this->whales[i].w = this->minAngularVelocity;
        }
    }
}

void WOA::RenderParticles()
{
    static Logger log(__FUNCTION__);
    cv::Mat image(500, 500, CV_8UC1, cv::Scalar(255, 255, 255));
    image = myMap.image.clone();
    /// Draw all particles.
    int x, y;
    for (int i = 0; i < this->population; ++i){
        x = XConvertToPixel(this->whales[i].x);
        y = YConvertToPixel(this->whales[i].y);
        cv::Point p(x, y); 
        circle(image, p, 1, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);

    }

    /// Goal configuration
    x = XConvertToPixel(this->sGoal.x);
    y = YConvertToPixel(this->sGoal.y);      
    cv::Point goal(x, y);
    circle(image, goal, 3, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);

    /// Init configuration
    x = XConvertToPixel(this->sNear.x);
    y = YConvertToPixel(this->sNear.y);
    cv::Point init(x, y);
    circle(image, init, 3, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);

    cv::imshow("WOA", image);
    cv::waitKey(0);
    
    cv::Mat grayImage3C;
    cv::cvtColor(image, grayImage3C, cv::COLOR_GRAY2BGR);

    myVideo->write(grayImage3C);
}

void WOA::CleanUp()
{
    delete this->whales;
}

/// ------------------------------------------------------------------------------------ 

long double WOA::GenerateRandom(long double a, long double b)
{
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_real_distribution<long double> distribution(a, b);
    return distribution(engine);
}

int WOA::GenerateRandom(int a, int b)
{
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_real_distribution<long double> distribution((long double)a, (long double)b);
    long double random = distribution(engine);
    return (int)(std::floor(random));  
}

long double WOA::CalculateEuclideanDistance(long double& x, long double& y, long double& a, long double& b){
    return sqrt(pow(x-a, 2)+pow(y-b, 2));
}

/// todo
long double WOA::CalculateDistance(long double& v){
    long double timeStep = mySettings["RRT*_time_step"];
    return (v * timeStep);
}
/// todo
long double WOA::CalculateAngle(long double& w){
    long double timeStep = mySettings["RRT*_time_step"];
    return (w * timeStep);
}

int WOA::XConvertToPixel(long double& x){

    return (int)((x + (-myMap.origin.x)) / myMap.resolution);
}

int WOA::YConvertToPixel(long double& y){
    return (int)((y + (-myMap.origin.x)) / myMap.resolution);
}

bool WOA::IsObstacleFree(State& sNew)
{
    #ifdef DEBUG
        static Logger log("IsObstacleFree [Point]");
    #endif
    int x = XConvertToPixel(sNew.x);
    int y = YConvertToPixel(sNew.y);
    cv::Point p(x, y);
    for (auto& segment: myMap.segments)
    {
        if (DoIntersect(p, p, segment.p, segment.q))
        {
            #ifdef DEBUG
                log.trace("State (x:{:+.2f}, y:{:+.2f}) is not free", sNew.x, sNew.y);
            #endif
            return false;
        }
    }
    #ifdef DEBUG
        log.trace("State (x:{:+.2f}, y:{:+.2f}) is free", sNew.x, sNew.y);
    #endif
    return true;
}

bool WOA::DoIntersect(cv::Point& p1, cv::Point& q1, cv::Point& p2, cv::Point& q2)
{
    int o1 = Orientation(p1, q1, p2);
    int o2 = Orientation(p1, q1, q2);
    int o3 = Orientation(p2, q2, p1);
    int o4 = Orientation(p2, q2, q1);
    if (o1 != o2 && o3 != o4)
        return true;
    if (o1 == 0 && OnSegment(p1, p2, q1)) return true;
    if (o2 == 0 && OnSegment(p1, q2, q1)) return true;
    if (o3 == 0 && OnSegment(p2, p1, q2)) return true;  
    if (o4 == 0 && OnSegment(p2, q1, q2)) return true;
    return false;
}

int WOA::Orientation(cv::Point& p, cv::Point& q, cv::Point& r)
{
    int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (val == 0) return 0;  // collinear
    return (val > 0)? 1: 2; // clock or counterclock wise
}

bool WOA::OnSegment(cv::Point& p, cv::Point& q, cv::Point& r)
{
    if (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) && q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y))
       return true;
    return false;
}