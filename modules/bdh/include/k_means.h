// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __k_means__
#define __k_means__

#include <iostream>
#include <limits.h>
#include <float.h>
#include <memory.h>

#include "NearestNeighbor.h"

#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

/**
* @brief k-means
*/
template<typename data_t, typename centroid_t>
class K_Means
{

private:
    int dim;     //!< dimension
    int K;       //!< number of centroid
    double error;//!< quantization error

    unsigned* centroid_size;//!< number of point included in corespond cluster
    double* centroid_error; //!< quantization error of correspond cluster
    centroid_t** centroid;  //!< centroid

    double CONVERGENCE_RATE;  //!< convergence rate 
    unsigned CONVERGENCE_LOOP;//!< convergence loop

public:

    /**
    * @brief default constructor
    */
    K_Means()
        : dim(0)
        , K(0)
        , error(0.0)
        , centroid_size(nullptr)
        , centroid_error(nullptr)
        , centroid(nullptr)
        , CONVERGENCE_RATE(1.0e-5)
        , CONVERGENCE_LOOP(UINT_MAX)
    {}

    /**
    * @brief destructor
    */
    ~K_Means();

    /**
    * @brief error getter
    * @return error
    */
    double getError()
    {
        return error;
    }

    /**
    * @brief centroid getter
    * @return centroid
    */
    double** getCentroid() const
    {
        return centroid;
    }

    /**
    * @brief calculate centroid
    */
    void calclateCentroid(
        int dim,       //!< [in] dimension
        int num,       //!< [in] number of sample
        data_t** point,//!< [in] sample point set
        int K          //!< [in] number of centroid
        );

private:

    void setParam(int dim, int K);

    void IniCentScala(int num, data_t** point);

    void updateMinMax(double& Min, double& Max, const double& val);

    /**
    * @brief find maximam value
    */
    unsigned findMaxIndex(
        int num, 
        double* MinDisTable
        );

    void IniCent_PlaPla(
        int num,
        data_t** point);

    void updateMinDisTable(
        centroid_t* centroid,
        int num,
        data_t** point, 
        double* MinDisTable
        );

    double CalCent(
        int num,
        data_t** point
        );

    int ResetCent(
        long num, 
        data_t** point
        );

};

template<typename data_t, typename centroid_t>
K_Means<data_t, centroid_t>::~K_Means()
{
    if (centroid != nullptr)
    {
        for (int i = 0; i < K; ++i)
        {
            delete[] centroid[i];
        }
        delete[] centroid;
        centroid = nullptr;
    }

    if (centroid_error != nullptr)
    {
        delete[] centroid_error;
        centroid_error = nullptr;
    }

    if (centroid_size != nullptr)
    {
        delete[] centroid_size;
        centroid_size = nullptr;
    }
}

template<typename data_t, typename centroid_t>
void K_Means<data_t, centroid_t>::calclateCentroid(
    int _dim,      //!< dimension
    int _num,      //!< number of sample
    data_t** point,//!< sample point set
    int _K         //!< number of centroid
    )
{
    setParam(_dim, _K);

    if (_dim == 1)
    {
        IniCentScala(_num, point);
    }
    else
    {
        IniCent_PlaPla(_num, point);
    }

    CalCent(_num, point);
}

template<typename data_t, typename centroid_t>
void K_Means<data_t, centroid_t>::setParam(int _dim, int _K)
{

    if (dim == _dim &&
        K == _K)
    {
        return;
    }
    this->~K_Means();


    dim = _dim;
    K = _K;
    centroid_size = new unsigned[K];
    centroid_error = new double[K];
    centroid = new double*[K];
    for (int i = 0; i < K; ++i){
        centroid[i] = new double[dim];
    }
}

template<typename data_t, typename centroid_t>
void K_Means<data_t, centroid_t>::IniCentScala(int num, data_t** point)
{

    double Min = DBL_MAX;
    double Max = -DBL_MAX;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < num; n++)
    {
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                updateMinMax(Min, Max, *point[n]);
            }
    }

    double delta = (Max - Min) / K;
    for (int c = 0; c < K; c++)
    {
        *centroid[c] = (centroid_t)(Min + delta*(c + 0.5));
    }
}

template<typename data_t, typename centroid_t>
void K_Means<data_t, centroid_t>::updateMinMax(double& Min, double& Max, const double& val)
{
    if (val < Min)
    {
        Min = val;
    }

    if (val > Max)
    {
        Max = val;
    }
}

template<typename data_t, typename centroid_t>
void K_Means<data_t, centroid_t>::IniCent_PlaPla(
    int num,
    data_t** point)
{

    /* Calculate the centroid */
    double* Mean = new double[dim]; /* mean = centroid */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int d = 0; d < dim; d++)
    {
        Mean[d] = 0;
        for (int n = 0; n < num; n++)
        {
            Mean[d] += point[n][d];
        }
        Mean[d] /= num;
    }

    // distance to the nearest neighbor centroid
    double* MinDisTable = new double[num];

    // compute the distance to each centroid of sample
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < num; n++)
    {
        MinDisTable[n] = Distance(dim, Mean, point[n]);
    }

    /* choose the most far centroid
    * and use it as initial value
    */
    {
        unsigned MinDisIndex_max = findMaxIndex(num, MinDisTable);

        for (int d = 0; d < dim; d++)
        {
            centroid[0][d] = point[MinDisIndex_max][d];
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < num; n++)
    {
        MinDisTable[n] = DBL_MAX;
    }

    /* update the distance for other centroids */
    for (int c = 1; c < K; c++)
    {

        updateMinDisTable(centroid[c - 1], num, point, MinDisTable);

        /* choose the furthest point(except CentIdx)
        * it is a next centroid
        */
        const int MinDisIndex_max = findMaxIndex(num, MinDisTable);

        for (int d = 0; d < dim; d++)
        {
            centroid[c][d] = point[MinDisIndex_max][d];
        }
    }

    /* release the memory */
    delete[] Mean;
    delete[] MinDisTable;
}

/* find the furthest index
*/
template<typename data_t, typename centroid_t>
unsigned K_Means<data_t, centroid_t>::findMaxIndex(int num, double* MinDisTable)
{

    unsigned MaxIndex = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int n = 1; n<num; n++)
    {

#ifdef _OPENMP
#pragma omp critical
#endif
            {
                if (MinDisTable[n] > MinDisTable[MaxIndex]){
                    MaxIndex = n;
                }
            }
    }

    return MaxIndex;
}

template<typename data_t, typename centroid_t>
void K_Means<data_t, centroid_t>::updateMinDisTable(centroid_t* _centroid, int num, data_t** point, double* MinDisTable){

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < num; n++)
    {
        //check the distance to the nearest centroid
        double dist = Distance(dim, _centroid, point[n]);

        if (dist < MinDisTable[n])
        {
            MinDisTable[n] = dist;
        }
    }

}

template<typename data_t, typename centroid_t>
double K_Means<data_t, centroid_t>::CalCent(
    int num,
    data_t** point
    ){

    centroid_t** tmpCent;
    centroid_t** lastCentroid = new centroid_t*[K];
    for (int i = 0; i < K; ++i){
        lastCentroid[i] = new centroid_t[dim];
    }

    double rate = DBL_MAX;
    error = DBL_MAX;

    for (unsigned loop = 0;
        rate > CONVERGENCE_RATE &&
        loop < CONVERGENCE_LOOP &&
        error > 1.0e-5
        ; ++loop)
    {

        //swap
        tmpCent = lastCentroid;
        lastCentroid = centroid;
        centroid = tmpCent;

        const double error0 = error;
        for (int i = 0; i < K; ++i)
        {
            centroid_size[i] = 0;
            centroid_error[i] = 0.0;

            for (int d = 0; d < dim; ++d)
            {
                centroid[i][d] = 0.0;
            }
        }

        //calculate new centroid
        point_t<centroid_t> NNcent;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) private(NNcent)
#endif
        for (int n = 0; n < num; ++n)
        {
            data_t* point_n_itr = point[n];
            data_t* point_n_itr_end = point_n_itr + dim;
            NearestNeighbor(dim, K, lastCentroid, point_n_itr, NNcent);
            size_t index = NNcent.index;
            double* centroid_index_itr = centroid[index];

#ifdef _OPENMP
#pragma omp critical
#endif
            {
                for (; point_n_itr != point_n_itr_end; ++point_n_itr)
                {
                    *centroid_index_itr += *point_n_itr;
                    ++centroid_index_itr;
                }
                ++centroid_size[index];
                centroid_error[index] += NNcent.distance;
            }
        }

        error = 0;
        for (int c = 0; c < K; c++)
        {
            error += centroid_error[c];
            centroid_error[c] /= centroid_size[c];

            if (centroid_size[c] == 0){

                int CentID = ResetCent(num, point);

                for (int d = 0; d < dim; d++)
                {
                    centroid[c][d] = point[CentID][d];
                }
            }
            else{
                for (int d = 0; d < dim; d++)
                {
                    centroid[c][d] /= centroid_size[c];
                }
            }
        }

        error /= num;
        rate = (error0 - error) / error0;
    }

    for (int i = 0; i < K; ++i){
        delete[] lastCentroid[i];
    }
    delete[] lastCentroid;


    return error;
}

template<typename data_t, typename centroid_t>
int K_Means<data_t, centroid_t>::ResetCent(long num, data_t** point){

    double distanceMax = DBL_MAX;
    long indexMax = 0;

    point_t<centroid_t> NNcent;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) private(NNcent)
#endif
    for (long n = 0; n < num; n++){
        NearestNeighbor(dim, K, centroid, point[n], NNcent);

#ifdef _OPENMP
#pragma omp critical
#endif
        {
            if (NNcent.distance > distanceMax){
                distanceMax = NNcent.distance;
                indexMax = n;
            }
        }
    }

    return indexMax;
}

#endif
