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

extern nlohmann::json mySettings;
extern Map myMap;
extern cv::VideoWriter* myVideo;

void WOA::Apply(Path path)
{
    static Logger log(__FUNCTION__);
    optimizedPath.push_back(path.array[0]);
    int lastValid;
    bool encounteredObstacle;
    long double maxSteeringAngle = mySettings["RRT*_max_steering_angle"];
    int j = 1;
    for (int i = 1; i < path.size; ++i)
    {
        bool isFree = TestCollision(optimizedPath.back(), path.array[i]);
        bool isSteerable = TestAngle(optimizedPath.back(), path.array[i]);
        if (isFree && isSteerable)
        {
            lastValid = i;
        }
        if (i == (path.size - 1) || !isFree)
        {
            MinimalWOA woa(optimizedPath.back(), path.array[lastValid]);
            State s = woa.Optimize();
            s.z = path.array[lastValid].z;

            int x, y;
            State a, b;
            a = optimizedPath.back();
            b = s;
            x = XConvertToPixel(a.x);
            y = YConvertToPixel(a.y);
            cv::Point p(x, y); 
            x = XConvertToPixel(b.x);
            y = YConvertToPixel(b.y);
            cv::Point q(x, y); 
            line(myMap.image, p, q, cv::Scalar(0x0), 2, cv::LINE_AA);

            optimizedPath.push_back(s);

            i = lastValid;

           
        }
        if (i == (path.size - 1))
        {
            cv::Mat grayImage3C;
            cv::cvtColor(myMap.image, grayImage3C, cv::COLOR_GRAY2BGR);
            myVideo->write(grayImage3C);
        }
            /// /// state to push = WOA(optimizedPath.back(), path.array())
            /// optimizedPath.push_back(state to push);
            /// i = lastValid;
    }
    for (State s: this->optimizedPath)
    {
        log.trace("{} {} {}", s.x, s.y, s.z);
    }
}

bool WOA::TestCollision(State& a, State& b){
    static Logger log(__FUNCTION__);
    int pX = XConvertToPixel(a.x);
    int pY = YConvertToPixel(a.y);
    cv::Point p(pX, pY);
    int qX = XConvertToPixel(b.x);
    int qY = YConvertToPixel(b.y);
    cv::Point q(qX, qY);
    for (auto& segment: myMap.segments){
        if (DoIntersect(p, q, segment.p, segment.q)){
            log.warn("Segment p({:+.2f}, {:+.2f}) q({:+.2f}, {:+.2f}) is not free", a.x, a.y, b.x, b.y);
            return false;
        }
    }
    log.trace("Segment p({:+.2f}, {:+.2f}) q({:+.2f}, {:+.2f}) is free", a.x, a.y, b.x, b.y);
    return true;
}

bool WOA::TestAngle(State& a, State& b)
{
    static Logger log(__FUNCTION__);
    long double maxSteeringAngle = mySettings["RRT*_max_steering_angle"];
    if (fabs(a.z - b.z) < maxSteeringAngle)
    {
        return true;
    }
    else
    {
        return false;
    }
}

State MinimalWOA::Optimize()
{
    static Logger log(__FUNCTION__);
    log.debug("Whale Optimization Algorithm");
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
            if ((this->p) < 0.5L)
            {
                if (fabs(this->A) < 0.5L)
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
    log.info("WOA Finished");
    return this->sBest;
}

void MinimalWOA::InitializePopulation()
{
    static Logger log(__FUNCTION__);
    this->whales = new State[this->population];
    for (int i = 0; i < this->population; ++i)
    {
        long double linearVelocity = GenerateRandom(this->linearVelocityMin, this->linearVelocityMax);
        long double angularVelocity = GenerateRandom(this->angularVelocityMin, this->angularVelocityMax);
        this->whales[i].v = linearVelocity;
        this->whales[i].w = angularVelocity;
        log.trace("Whale {} (Linear: {:.2f}, Angular: {:.2f})", i, this->whales[i].v, this->whales[i].w);
    }
}

void MinimalWOA::CalculateFitness()
{
    static Logger log(__FUNCTION__);
    double long bestFitness = 0.0L, x, y, z;
    int bestIndex = 0;
    for (int i = 0; i < this->population; ++i)
    {
        x = this->whales[i].x = (this->sInit.x) - CalculateDistance(this->whales[i].v)*sin(this->sInit.z + CalculateAngle(this->whales[i].w)/2.0l);
        y = this->whales[i].y = (this->sInit.y) + CalculateDistance(this->whales[i].v)*cos(this->sInit.z + CalculateAngle(this->whales[i].w)/2.0l);
        z = this->whales[i].z = (this->sInit.z) + CalculateAngle(this->whales[i].w);
        long double distanceDifference = CalculateEuclideanDistance(x, y, this->sGoal.x, this->sGoal.y);
        long double angleDifference = fabs(this->sGoal.z - z);
        long double fitness = 1 / distanceDifference; // to do (add more factors)(angle nominator: cos(alpha))
        if (fitness > bestFitness)
        {
            bestFitness = fitness;
            bestIndex = i;
        }
        log.trace("Whale {} (x: {:.2f}, y:{:.2f}, z:{:.2f}) | Fitness: {:.2f}", i, x, y, z, fitness);
    }
    this->sBest = this->whales[bestIndex];
    log.trace("Best Whale: {}, Fitness: {:.2f}", bestIndex, bestFitness);
}

void MinimalWOA::CoefficientUpdate()
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

void MinimalWOA::CircleUpdate()
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

void MinimalWOA::RandomUpdate()
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

void MinimalWOA::SpiralUpdate()
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

void MinimalWOA::CheckBoundary()
{
    static Logger log(__FUNCTION__);
    for (int i = 0; i < this->population; ++i)
    {
        if (this->whales[i].v > this->linearVelocityMax)
        {
            log.trace("Whale {} (Linear: {:.2f} => {:.2f})", i, this->whales[i].v, this->linearVelocityMax);
            this->whales[i].v = this->linearVelocityMax;
        }
        else if (this->whales[i].v < this->linearVelocityMin)
        {
            log.trace("Whale {} (Linear: {:.2f} => {:.2f})", i, this->whales[i].v, this->linearVelocityMin);
            this->whales[i].v = this->linearVelocityMin;
        }
        if (this->whales[i].w > this->angularVelocityMax)
        {
            log.trace("Whale {} (Angular: {:.2f} => {:.2f})", i, this->whales[i].w, this->angularVelocityMax);
            this->whales[i].w = this->angularVelocityMax;
        }
        else if (this->whales[i].w < this->angularVelocityMin)
        {
            log.trace("Whale {} (Angular: {:.2f} => {:.2f})", i, this->whales[i].w, this->angularVelocityMin);
            this->whales[i].w = this->angularVelocityMin;
        }
    }
}

void MinimalWOA::RenderParticles()
{
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
    x = XConvertToPixel(this->sInit.x);
    y = YConvertToPixel(this->sInit.y);
    cv::Point init(x, y);
    circle(image, init, 3, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);

    cv::imshow("WOA", image);
    cv::waitKey(0);
    
    cv::Mat grayImage3C;
    cv::cvtColor(image, grayImage3C, cv::COLOR_GRAY2BGR);


    myVideo->write(grayImage3C);

}

void MinimalWOA::CleanUp()
{
    delete this->whales;
}

/// ------------------------------------------------------------------------------------ 

long double MinimalWOA::GenerateRandom(long double a, long double b)
{
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_real_distribution<long double> distribution(a, b);
    return distribution(engine);
}

int MinimalWOA::GenerateRandom(int a, int b)
{
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_real_distribution<long double> distribution((long double)a, (long double)b);
    double long random = distribution(engine);
    return (int)(std::floor(random));  
}

long double MinimalWOA::CalculateEuclideanDistance(long double& x, long double& y, long double& a, long double& b){
    return sqrt(pow(x-a, 2)+pow(y-b, 2));
}

/// todo
long double MinimalWOA::CalculateDistance(long double& v){
    return (v * 1.0);
}
/// todo
long double MinimalWOA::CalculateAngle(long double& w){
    return (w * 1.0L);
}

int MinimalWOA::XConvertToPixel(long double& x){
    return (int)((x + (12.5L)) / 0.05L);
}

int MinimalWOA::YConvertToPixel(long double& y){
    return (int)((y + (12.5L)) / 0.05L);
}

int WOA::Orientation(cv::Point& p, cv::Point& q, cv::Point& r){
    int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (val == 0) return 0;  // collinear
    return (val > 0)? 1: 2; // clock or counterclock wise
}

bool WOA::OnSegment(cv::Point& p, cv::Point& q, cv::Point& r){
    if (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) && q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y))
       return true;
    return false;
}

int WOA::XConvertToPixel(long double& x){
    return (int)((x + (-myMap.origin.x)) / myMap.resolution);
}

int WOA::YConvertToPixel(long double& y){
    return (int)((y + (-myMap.origin.y)) / myMap.resolution);
}

bool WOA::DoIntersect(cv::Point& p1, cv::Point& q1, cv::Point& p2, cv::Point& q2){
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
