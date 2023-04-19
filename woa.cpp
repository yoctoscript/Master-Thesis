#include "woa.hpp"
#include "logger.hpp"
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
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
        for (; this->i < this->population; ++(this->i))
        {
            CoefficientUpdate();
            if ((this->p) < 1.0L)
            {
                if (fabs(this->A) < 1.0L)
                {
                    CircleUpdate();
                }
                else
                {
                    RandomUpdate();
                }
            }
            else
            {
                SpiralUpdate();
            }
        }
        CheckBoundary();
        CalculateFitness();
        RenderParticles();
    }
    return this->sBest;
}

void WOA::InitializePopulation()
{
    static Logger log(__FUNCTION__);
    this->whales = new State[this->population];
    long double linearVelocityMin = 0.0L;
    long double linearVelocityMax = 5.0L;
    long double angularVelocityMin = -1.0L;
    long double angularVelocityMax = 1.0L;
    for (int i = 0; i < this->population; ++i)
    {
        long double linearVelocity = GenerateRandom(linearVelocityMin, linearVelocityMax);
        long double angularVelocity = GenerateRandom(angularVelocityMin, angularVelocityMax);
        this->whales[i].v = linearVelocity;
        this->whales[i].w = angularVelocity;
        log.trace("Whale {} (Linear: {:.2f}, Angular: {:.2f})", i, this->whales[i].v, this->whales[i].w);
    }
    this->a = this->_a;
    this->iterations = this->_iterations;
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
    this->a -= (this->_a / (this->iterations * this->population));
    this->r = GenerateRandom(0.0l, 1.0l);
    this->A = (2 * this->a * this->r) - this->a;
    this->C = (2 * this->r);
    this->l = GenerateRandom(-1.0l, 1.0l);
    this->p = GenerateRandom(0.0l, 1.0l);
    this->b = GenerateRandom(0.0l, 1.0l);
}

void WOA::CircleUpdate()
{
    long double Dv = fabs((this->C * this->sBest.v) - this->whales[i].v);
    long double Dw = fabs((this->C * this->sBest.w) - this->whales[i].w);
    this->whales[i].v = this->sBest.v - (this->A * Dv);
    this->whales[i].w = this->sBest.w - (this->A * Dw);
}

void WOA::RandomUpdate()
{
    int index = GenerateRandom(0, this->population);
    State sRand = this->whales[index];
    long double Dv = fabs((this->C * sRand.v) - this->whales[i].v);
    long double Dw = fabs((this->C * sRand.w) - this->whales[i].w);
    this->whales[i].v = sRand.v - (this->A * Dv);
    this->whales[i].w = sRand.w - (this->A * Dw);
}

void WOA::SpiralUpdate()
{
    long double Dv = fabs(this->sBest.v - this->whales[i].v);
    long double Dw = fabs(this->sBest.w - this->whales[i].w);
    this->whales[i].v = Dv * exp(b*l) * cos(2* M_PI * l) + whales[i].v;
    this->whales[i].w = Dv * exp(b*l) * cos(2* M_PI * l) + whales[i].w;
}

void WOA::CheckBoundary()
{
    static Logger log(__FUNCTION__);
    log.trace("Checking boundaries");
}

void WOA::RenderParticles()
{
    cv::Mat output(image.size(), CV_8UC3);
    cv::cvtColor(image, output, cv::COLOR_GRAY2BGR);
    std::vector<cv::Mat> channels(3);
    cv::split(output, channels);
    channels[0] = image.clone();
    channels[1] = image.clone();
    channels[2] = image.clone();
    cv::merge(channels, output);
    int x, y;
    /// Draw all states.
    for (int i = 1; i < state_count; ++i){
        State a = state_array[i];
        State b = *(a.p);
        x = x_convert_to_pixel(a.x);
        y = y_convert_to_pixel(a.y);
        cv::Point p(x, y); 
        x = x_convert_to_pixel(b.x);
        y = y_convert_to_pixel(b.y);
        cv::Point q(x, y); 
        line(output, p, q, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
    /// Initial configuration.
    x = x_convert_to_pixel(s_init.x);
    y = y_convert_to_pixel(s_init.y);
    cv::Point init(x, y);
    circle(output, init, 5, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    /// Goal threshold region and configuration.
    x = x_convert_to_pixel(s_goal.x);
    y = y_convert_to_pixel(s_goal.y);
    cv::Point goal(x, y);
    int goal_radius = (int)(goal_threshold / resolution);
    circle(output, goal, goal_radius, cv::Scalar(157, 157, 255), -1, cv::LINE_AA);
    circle(output, goal, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    /// Draw shortest path.
    for (int i = 1; i < path.size; ++i){
        State a = path.array[i];
        State b = *(a.p);
        x = x_convert_to_pixel(a.x);
        y = y_convert_to_pixel(a.y);
        cv::Point p(x, y); 
        x = x_convert_to_pixel(b.x);
        y = y_convert_to_pixel(b.y);
        cv::Point q(x, y); 
        line(output, p, q, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    }

    cv::imshow("RRT*", output);
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
    return (int)((x + (20.0L)) / 0.05L);
}

int WOA::YConvertToPixel(long double& y){
    return (int)((y + (20.0L)) / 0.05L);
}