// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "baseset.h"
#include <opencv2/core.hpp>
#include <opencv2/bdh.hpp>
#include <NearestNeighbor.h>
#include <k_means.h>
#include <algorithm>

namespace cv { namespace bdh {

size_t Subspace::getSubHashValue(
    InputArray _data
) const
{
    Mat data = _data.getMat().reshape(1, baseVector.cols);

    //work space
    double* PCAdata = new double[subDim];

    getPCAdata(data, PCAdata);
    int idx = NearestNeighbor(subDim, subHashSize, centroidVector, PCAdata);
    delete[] PCAdata;

    return hashKey[idx];
}

void Subspace::setParameters(const baseset_t& baseSet)
{
    bit = baseSet.bit;
    subDim = baseSet.subDim;
    subHashSize = baseSet.k;
    variance = baseSet.variance;

    for (size_t i = 0; i < (unsigned)subHashSize; i++)
    {
        cellVariance.push_back(baseSet.cellVariance[i]);
    }

    for (int h = 0; h < subHashSize; h++)
    {
        std::vector<double> stub;
        for (auto i = 0; i < subDim; i++)
        {
            stub.push_back(baseSet.centroid[h][i]);
        }
        centroidVector.push_back(stub);
    }

    for (int d = 0; d < subDim; d++)
    {
        Mat stub = Mat(baseSet.base[d].direction);
        baseVector.push_back(stub.t());
    }
}

double Subspace::getDistanceToCentroid(double* PCAquery, int centroidIndex) const
{
    double t = cellVariance[centroidIndex];
    std::vector<double>::const_iterator centroid_p = centroidVector[centroidIndex].begin();
    std::vector<double>::const_iterator centroid_p_end = centroidVector[centroidIndex].end();
    do
    {
        double x = (*PCAquery++) - (*centroid_p++);
        t += x * x;
    } while (centroid_p != centroid_p_end);

    return t;
}

} } // namespace
