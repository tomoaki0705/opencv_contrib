
#include <Subspace.h>
#include <NearestNeighbor.h>
#include <k_means.h>
#include <algorithm>

namespace cv { namespace bdh {
int Subspace::dim;

void Subspace::setParameters(const baseset_t& baseSet)
{
	bit = baseSet.bit;
	subDim = baseSet.subDim;
	subHashSize = baseSet.k;
	variance = baseSet.variance;

	cellVariance = new double[subHashSize];
	memcpy(cellVariance, baseSet.cellVariance, sizeof(double)*subHashSize);

	for (int h = 0; h < subHashSize; ++h)
	{
        std::vector<double> stub;
        for (auto i = 0; i < subDim; i++)
        {
            stub.push_back(baseSet.centroid[h][i]);
        }
        centroidVector.push_back(stub);
	}

	for (int d = 0; d < subDim; ++d){
        std::vector<double> stub;
        for (auto i = 0; i < dim; i++)
        {
            stub.push_back(baseSet.base[d].direction[i]);
        }
	}
}

void Subspace::clear(){

	if (cellVariance != nullptr){
		delete[] cellVariance;
		cellVariance = nullptr;
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