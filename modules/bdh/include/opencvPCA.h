/**
* @file    opencvPCA.h
* @author  T.Sato
* @date    2015.05.06
* @version 1.0
*/

#ifndef __opencvPCA__
#define __opencvPCA__
#include <superPCA.h>
#include <string>
#include <iostream>
#include <fstream>
#include <measure.h>

using namespace std;
using namespace cv;

//#define SHOW_PROGRESS

/**
* @brief Pricipal Component Analysis
*/
class PrincipalComponentAnalysis : public superPCA
{

public:
    // constructor
    PrincipalComponentAnalysis()
        : superPCA()
    {}

    /**
    * @brief PCA (covariance)
    */
    template <typename data_t>
    void executePCA(int dim, size_t num, data_t** data);

    /**
    * @brief executePCA (coefficient)
    */
    template <typename data_t>
    bool executePCAcorelationCoefficient(
        int dim, int num, data_t** data);
};

template <typename data_t>
void PrincipalComponentAnalysis::executePCA(
    int _dim, 
    size_t num, 
    data_t** data)
{
    // reallocate the memory when dimension changed
    resetDimension(_dim);

    // calculate the mean and covariance matrix of each eigen vector
#ifdef SHOW_PROGRESS
    cout << "calclate covariance matrix" << endl;
    double timeStart = GetCPUTime();
#endif
    double* Mean = new double[_dim];
    double** covariance = new double*[_dim];
    for (int d = 0; d < _dim; ++d)
    {
        covariance[d] = new double[_dim];
    }
    calculateCovarianceMatrix(_dim, num, data, Mean, covariance);

    Mat Cov(_dim, _dim, CV_64FC1);
    for (int d = 0; d < _dim; ++d)
    {
        Cov.at<double>(d, d) = covariance[d][d];
        for (int d2 = d + 1; d2 < _dim; ++d2)
        {
            Cov.at<double>(d, d2) = Cov.at<double>(d2, d) = covariance[d][d2];
        }
    }
#ifdef SHOW_PROGRESS
    double timeEnd = GetCPUTime();
    cout << static_cast<int>(timeEnd-timeStart)/1000 << " sec\t" << endl;
    cout << "Eigenvalue decomposition" << endl;
    timeStart = GetCPUTime();
#endif

    //calc eigen vector
    Mat EigVal, EigVec;
    eigen(Cov, EigVal, EigVec);

    /*copy the Eigen values and eigen vectors*/
    for (int d = _dim - ZeroCount; d<_dim; d++)
    {
        EigVal.at<double>(d) = 0.0;
    }

    for (int d2, d = 0; d < _dim; d++)
    {
        for (d2 = 0; d2 < _dim; d2++)
        {
            pcDir[d].direction[d2] = EigVec.at<double>(d, d2);
        }

        pcDir[d].variance = EigVal.at<double>(d);
        pcDir[d].mean = innerProduct(_dim, Mean, pcDir[d].direction);
    }

    //sort int Descending order of variance
    sort(pcDir, pcDir + _dim);

#ifdef SHOW_PROGRESS
    timeEnd = GetCPUTime();
    cout << static_cast<int>(timeEnd - timeStart) / 1000 << " sec\t" << endl;
#endif

    //delete
    Cov.release();
    EigVal.release();
    EigVec.release();
    delete[] Mean;
    for (int d = 0; d < _dim; ++d)
    {
        delete[] covariance[d];
    }
    delete[] covariance;
}

#endif
