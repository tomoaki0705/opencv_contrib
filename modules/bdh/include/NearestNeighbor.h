/**
* @file    NearestNeighbor.h
* @author  T.Sato
* @date    2015.05.05
* @version 1.0
*/

#ifndef __NEAREST_NEIGHBOR__
#define __NEAREST_NEIGHBOR__

#include "point.h"
#include <vector>

/**
* @brief distance calculation
* @return distance
*/
template<typename data_t, typename query_t>
double Distance(
	int dim,	   //!< [in] dimension
	data_t* sample,//!< [in] data
	query_t* query //!< [in] query
	);


/**
* @brief k-means
*/
template<typename data_t, typename query_t>
int NearestNeighbor(
	int dim,
	int num,
	data_t** sample,
	query_t* query
	);

/**
* @brief k-means
*/
template<typename data_t, typename query_t>
void NearestNeighbor(
	int dim,
	int num,
	data_t** sample,
	query_t* query,
	point_t<data_t>& NNpoint
	);

template<typename data_t, typename query_t>
double Distance(
    int dim,
    data_t* sample,
    query_t* query
)
{
    double dist = 0.0;

    for (int d = 0; d < dim; ++d) {
        dist += NORM(query[d] - sample[d]);
    }

    return dist;
}

template<typename data_t, typename query_t>
double Distance(
    int dim,
    const std::vector<data_t> &sample,
    query_t* query
)
{
    double dist = 0.0;

    for (int d = 0; d < dim; ++d) {
        dist += NORM(query[d] - sample[d]);
    }

    return dist;
}

template<typename data_t, typename query_t>
int NearestNeighbor(
	int dim, 
	int num, 
	data_t** sample, 
	query_t* query
	)
{

	int NNidx = 0;
	double NNdis = Distance(dim, sample[0], query);
	for (int n = 1; n < num; n++){

        double distance = Distance(dim, sample[n], query);
		if (distance < NNdis){
			NNdis = distance;
			NNidx = n;
		}
	}
	return NNidx;
}

template<typename data_t, typename query_t>
int NearestNeighbor(
    int dim,
    int num,
    const std::vector<std::vector<data_t> > &sample,
    query_t* query
)
{

    int NNidx = 0;
    double NNdis = Distance(dim, sample[0], query);
    for (int n = 1; n < num; n++) {

        double distance = Distance(dim, sample[n], query);
        if (distance < NNdis) {
            NNdis = distance;
            NNidx = n;
        }
    }
    return NNidx;
}

template<typename data_t, typename query_t>
void NearestNeighbor(
	int dim, 
	int num, 
	data_t** sample, 
	query_t* query,
	point_t<data_t>& NNpoint
	)
{
	NNpoint.index = 0;
	NNpoint.distance = Distance(dim, sample[0], query);
	for (int n = 1; n < num; n++)
	{

        double distance = Distance(dim, sample[n], query);
		if (distance < NNpoint.distance)
		{
			NNpoint.distance = distance;
			NNpoint.index = n;
		}
	}
}

#endif// __NEAREST_NEIGHBOR__
