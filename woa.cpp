#include "woa.hpp"
#include "logger.hpp"
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <random>


State WOA::Optimize()
{
    static Logger log(__FUNCTION__);
    log.debug("WOA optimization initiated ..");
    log.debug("InitializePopulation()");
    InitializePopulation();
    log.debug("CalculateFitness()");
    CalculateFitness();
    while (this->iterations--)
    {
        for (this->i = 0; this->i < this->population; ++(this->i))
        {
            log.debug("CoefficientUpdate()");
            CoefficientUpdate();
            if ((this->p) < 1.0L)
            {
                if (fabs(this->A) < 1.0L)
                {
                    log.debug("CircleUpdate()");
                    CircleUpdate();
                }
                else
                {
                    log.debug("RandomUpdate()");
                    RandomUpdate();
                }
            }
            else
            {
                log.debug("SpiralUpdate()");
                SpiralUpdate();
            }
        }
        log.debug("CheckBoundary");
        CheckBoundary();
        log.debug("CalculateFitness");
        CalculateFitness();
        log.debug("RenderParticles()");
        RenderParticles();
    }
    log.debug("CleanUp()");
    CleanUp();
    return this->sBest;
}

void WOA::InitializePopulation()
{
    static Logger log(__FUNCTION__);
    this->whales = new State[this->population];
    long double linearVelocityMin = 0.0L;
    long double linearVelocityMax = 25.0L;
    long double angularVelocityMin = -2.0L;
    long double angularVelocityMax = 2.0L;
    for (int i = 0; i < this->population; ++i)
    {
        long double linearVelocity = GenerateRandom(linearVelocityMin, linearVelocityMax);
        long double angularVelocity = GenerateRandom(angularVelocityMin, angularVelocityMax);
        this->whales[i].v = linearVelocity;
        this->whales[i].w = angularVelocity;
        log.trace("Whale {} (Linear: {:.2f}, Angular: {:.2f})", i, this->whales[i].v, this->whales[i].w);
    }
}

void WOA::CalculateFitness()
{
    static Logger log(__FUNCTION__);
    double long bestFitness = 0.0L, x, y, t;
    int bestIndex = 0;
    for (int i = 0; i < this->population; ++i)
    {
        x = this->whales[i].x = (this->sInit.x) - CalculateDistance(this->whales[i].v)*sin(this->sInit.t + CalculateAngle(this->whales[i].w)/2.0l);
        y = this->whales[i].y = (this->sInit.y) + CalculateDistance(this->whales[i].v)*cos(this->sInit.t + CalculateAngle(this->whales[i].w)/2.0l);
        t = this->whales[i].t = (this->sInit.t) + CalculateAngle(this->whales[i].w);
        long double distanceDifference = CalculateEuclideanDistance(x, y, this->sGoal.x, this->sGoal.y);
        long double angleDifference = fabs(this->sGoal.t - t);
        long double fitness = 1 / distanceDifference; // to do (add more factors)(angle nominator: cos(alpha))
        if (fitness > bestFitness)
        {
            bestFitness = fitness;
            bestIndex = i;
        }
        log.trace("Whale {} (x: {:.2f}, y:{:.2f}, t:{:.2f}) | Fitness: {:.2f}", i, x, y, t, fitness);
    }
    this->sBest = this->whales[bestIndex];
    log.trace("Best Whale: {}, Fitness: {:.2f}", bestIndex, bestFitness);
}

void WOA::CoefficientUpdate()
{
    static Logger log(__FUNCTION__);
    this->a -= (this->_a / (long double)(this->_iterations * this->population));
    log.trace((this->_a / (long double)(this->_iterations * this->population)));
    this->r = GenerateRandom(0.0l, 1.0l);
    this->A = (2 * this->a * this->r) - this->a;
    this->C = (2 * this->r);
    this->l = GenerateRandom(-1.0l, 1.0l);
    this->p = GenerateRandom(0.0l, 1.0l);
    this->b = GenerateRandom(0.0l, 1.0l);
    log.trace("a: {:.3f} | r: {:.2f} | A: {:.2f} | C: {:.2f} | l: {:.2f} | p: {:.2f} | b: {:.2f}", this->a, this->r, this->A, this->C, this->l, this->p, this->b);
}

void WOA::CircleUpdate()
{
    static Logger log(__FUNCTION__);
    long double oldV = this->whales[this->i].v;
    long double oldW = this->whales[this->i].w;
    long double Dv = fabs((this->C * this->sBest.v) - this->whales[this->i].v);
    long double Dw = fabs((this->C * this->sBest.w) - this->whales[this->i].w);
    this->whales[this->i].v = this->sBest.v - (this->A * Dv);
    this->whales[this->i].w = this->sBest.w - (this->A * Dw);
    long double newV = this->whales[this->i].v;
    long double newW = this->whales[this->i].w;
    log.trace("Whale {} (Linear: {:.2f} => {:.2f})(Angular: {:.2f} => {:.2f})", this->i, oldV, newV, oldW, newW);
}

void WOA::RandomUpdate()
{
    static Logger log(__FUNCTION__);
    long double oldV = this->whales[this->i].v;
    long double oldW = this->whales[this->i].w;
    int index = GenerateRandom(0, this->population);
    State sRand = this->whales[index];
    long double Dv = fabs((this->C * sRand.v) - this->whales[this->i].v);
    long double Dw = fabs((this->C * sRand.w) - this->whales[this->i].w);
    this->whales[this->i].v = sRand.v - (this->A * Dv);
    this->whales[this->i].w = sRand.w - (this->A * Dw);
    long double newV = this->whales[this->i].v;
    long double newW = this->whales[this->i].w;
    log.trace("Whale {} (Linear: {:.2f} => {:.2f})(Angular: {:.2f} => {:.2f})", this->i, oldV, newV, oldW, newW);

}

void WOA::SpiralUpdate()
{
    static Logger log(__FUNCTION__);
    long double oldV = this->whales[this->i].v;
    long double oldW = this->whales[this->i].w;
    long double Dv = fabs(this->sBest.v - this->whales[this->i].v);
    long double Dw = fabs(this->sBest.w - this->whales[this->i].w);
    this->whales[this->i].v = Dv * exp(b*l) * cos(2* M_PI * l) + whales[this->i].v;
    this->whales[this->i].w = Dv * exp(b*l) * cos(2* M_PI * l) + whales[this->i].w;
    long double newV = this->whales[this->i].v;
    long double newW = this->whales[this->i].w;
    log.trace("Whale {} (Linear: {:.2f} => {:.2f})(Angular: {:.2f} => {:.2f})", this->i, oldV, newV, oldW, newW);
}

void WOA::CheckBoundary()
{
    static Logger log(__FUNCTION__);
    log.trace("Checking boundaries");
}

void WOA::RenderParticles()
{
    cv::Mat image(500, 500, CV_8UC3, cv::Scalar(255, 255, 255));

    /// Draw all particles.
    int x, y;
    for (int i = 0; i < this->population; ++i){
        x = XConvertToPixel(this->whales[i].x);
        y = YConvertToPixel(this->whales[i].y);
        cv::Point p(x, y); 
        circle(image, p, 3, cv::Scalar(0, i*255/this->population, 255), -1, cv::LINE_AA);

    }

    /// Goal configuration
    x = XConvertToPixel(this->sGoal.x);
    y = YConvertToPixel(this->sGoal.y);
    cv::Point goal(x, y);
    circle(image, goal, 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);



    cv::imshow("RRT*", image);
    cv::waitKey(0);
    char filename[256];
    std::sprintf(filename, "frames/frame%d.jpg", this->_iterations - this->iterations);
    cv::imwrite(filename, image);
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
    double long random = distribution(engine);
    return (int)(std::floor(random));  
}

long double WOA::CalculateEuclideanDistance(long double& x, long double& y, long double& a, long double& b){
    return sqrt(pow(x-a, 2)+pow(y-b, 2));
}

/// todo
long double WOA::CalculateDistance(long double& v){
    return (v * 1.0);
}
/// todo
long double WOA::CalculateAngle(long double& w){
    return (w * 1.0L);
}

int WOA::XConvertToPixel(long double& x){
    return (int)((x + (12.5L)) / 0.05L);
}

int WOA::YConvertToPixel(long double& y){
    return (int)((y + (12.5L)) / 0.05L);
}