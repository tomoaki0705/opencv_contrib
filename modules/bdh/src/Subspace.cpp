
#include <Subspace.h>
#include <NearestNeighbor.h>
#include <k_means.h>
#include <algorithm>

namespace cv { namespace bdh {

size_t Subspace::getSubHashValue(
    InputArray _data
) const
{
    Mat data = _data.getMat().reshape(1, subDim);
    CV_Assert(subDim == baseVector.rows);

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

    for (size_t i = 0; i < subHashSize; i++)
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
        Mat stub(1, dim, CV_64FC1, baseSet.base[d].direction);
        baseVector.push_back(stub);
    }
}

double Subspace::getDistanceToCentroid(double* PCAquery, int centroidIndex) const
{
    double t = cellVariance[centroidIndex];
    std::vector<double>::const_iterator centroid_p = centroidVector[centroidIndex].begin();
    std::vector<double>::const_iterator centroid_p_end = centroidVector[centroidIndex].end();
    do
    {
        t += NORM((*PCAquery++) - (*centroid_p++));
    } while (centroid_p != centroid_p_end);

    return t;
}

} } // namespace