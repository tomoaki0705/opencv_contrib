#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/bdh.hpp"
#include "opencvPCA.h"
#include "Subspace.h"
#include <BDHtraining.h>
#include <BDH.h>
#include <limits>
const double deltaRate = 50;

///************** Indexing *******************/
namespace cv {
namespace bdh {
int Subspace::dim;

void Subspace::setNodeParam(node_t * node, InputArray _query) const
{
    Mat query = _query.getMat().reshape(1, 1);

    double* PCAquery = new double[subDim];
    getPCAdata(query, PCAquery);

    for (int b = 0; b < subHashSize; ++b)
    {
        node[b].distance = getDistanceToCentroid(PCAquery, b);
        node[b].hashKey = hashKey[b];
    }
    std::sort(node, node + subHashSize);

    delete[] PCAquery;
}

void Index::setLayerParam(layer_t * layer, InputArray _query) const
{

    layer_t* layer_p = layer;
    layer_t* layer_p_end = layer + M;
    for (int m = 0; layer_p != layer_p_end; ++m, ++layer_p)
    {
        layer_p->k = subspace[m].subHashSize;
        subspace[m].setNodeParam(layer_p->node, _query);
        layer_p->calc_gap();
    }
    std::sort(layer, layer + M);

    // compute the maximum and minimum distance to the remaining subspace from m-th subspace
    double distRestMin = 0;
    double distRestMax = 0;
    layer_p_end = layer - 1;
    for (layer_p = layer + M - 1; layer_p != layer_p_end; --layer_p)
    {
        layer_p->restMin = distRestMin += layer_p->node[0].distance;
        layer_p->restMax = distRestMax += layer_p->node[layer_p->k - 1].distance;
    }

}


int Index::NearBucket_R(
    const double Radius,          // search radius
    layer_t* const layer,         // subspace distance from query to each layer
    const status_t& status,       // express the status of node
    vector<hashKey_t>& bucketList //![out] collect hash key of buckets near than Radius from query
)const
{
    const int m_plus1 = status.m + 1;

    int count = 0;

    if (m_plus1 == M)
    {
        index_t collision;
        node_t* node = layer[status.m].node;
        const double layerBound = Radius - status.dist;
        for (; node->distance <= layerBound; ++node)
        {
            size_t hashKey = status.hashKey + node->hashKey;

            collision = hashTable.getCollision(hashKey);

            if (collision > 0)
            {
                count += collision;
                bucketList.push_back(
                    hashKey_t(hashKey, status.dist + node->distance)
                );
            }
        }
    }
    else
    {
        status_t statusNext(m_plus1);
        node_t* node = layer[status.m].node;

        const double layerBound = Radius - status.dist - layer[m_plus1].restMin;
        for (; node->distance <= layerBound; ++node)
        {
            statusNext.hashKey = status.hashKey + node->hashKey;
            statusNext.dist = status.dist + node->distance;

            count += NearBucket_R(
                Radius,
                layer,
                statusNext,
                bucketList
            );
        }
    }

    return count;
}


int Index::NearBucket_C(
    const double& Lbound, // lower boundary of search space
    const double& Ubound, // upper boundary of search space
    layer_t* const layer, // subspace distance from query to each layer
    const status_t& status,
    vector<hashKey_t>& bucketList
) const
{

    const int m_plus1 = status.m + 1;
    int count = 0;

    if (m_plus1 == M)
    {
        hashKey_t Key;
        address_t bucket_p;

        const double layerLowerBound = Lbound - status.dist;
        const double layerUpperBound = Ubound - status.dist;
        node_t* node = layer[status.m].node;
        for (; node->distance <= layerLowerBound; ++node) {}
        for (; node->distance <= layerUpperBound; ++node)
        {
            size_t hashKey = status.hashKey + node->hashKey;
            bucket_p = hashTable.getPointer(hashKey);
            if (bucket_p)
            {
                Key.setVariable(hashKey, status.dist + node->distance);
                bucketList.push_back(Key);
                count += *reinterpret_cast<collision_t*>(bucket_p);
            }
        }
    }
    else
    {
        const double layerLowerBound = Lbound - status.dist - layer[m_plus1].restMax;
        const double layerUpperBound = Ubound - status.dist - layer[m_plus1].restMin;

        status_t statusNext(m_plus1);

        node_t* node = layer[status.m].node;
        for (; node->distance <= layerLowerBound; ++node) {}
        for (; node->distance <= layerUpperBound; ++node)
        {
            statusNext.hashKey = status.hashKey + node->hashKey;
            statusNext.dist = status.dist + node->distance;

            count += NearBucket_C(
                Lbound, Ubound,
                layer, statusNext, bucketList
            );

        }

    }

    return count;
}


int Index::NearBucket_C_list(
    const double Rbound,                // search space radius
    layer_t* const layer,               // subspace distance from query to each layer
    std::list<status_t>& statusQue,     // status of node while searching
    std::list<status_t>::iterator* itr, // status of current node
    std::vector<hashKey_t>& bucketList
) const
{

    const int m = (*itr)->m;
    const int m_plus1 = m + 1;
    node_t* const node = layer[m].node;

    int count = 0;
    double layerBound = Rbound - (*itr)->dist;
    int i = (*itr)->nodeIdx;
    if (m_plus1 == M)
    {
        index_t collision;

        for (; node[i].distance <= layerBound; ++i)
        {
            size_t hashKey = (*itr)->hashKey + node[i].hashKey;
            collision = hashTable.getCollision(hashKey);
            if (collision > 0)
            {
                bucketList.push_back(
                    hashKey_t(hashKey, (*itr)->dist + node[i].distance)
                );
                count += collision;
            }
        }
    }
    else
    {
        layerBound -= layer[m_plus1].restMin;

        status_t statusNext(m_plus1);
        list<status_t>::iterator itr_next;
        for (; node[i].distance <= layerBound; ++i) {

            statusNext.hashKey = (*itr)->hashKey + node[i].hashKey;
            statusNext.dist = (*itr)->dist + node[i].distance;
            statusQue.push_front(statusNext);

            itr_next = statusQue.begin();
            count += NearBucket_C_list(
                Rbound, layer, statusQue, &itr_next, bucketList
            );
        }
    }

    // was all node accessed ?
    if (i == layer[m].k)
    {
        // remove the current node and move to the next node
        statusQue.erase((*itr)++);
    }
    else
    {
        // update the status of current node and move to the next node
        (*itr)->nodeIdx = i;
        ++(*itr);
    }

    return count;
}

template <typename data_t>
double computeNorm(Mat& query, const data_t *data, double cutoffDistance)
{
    double distance = 0.0;
    for (int i = 0; i < query.rows && distance < cutoffDistance; i++)
    {
        distance += NORM(query.data[i] - (*data++));
    }
    return distance;
}

#define COMPUTE_NORM(type2, src1, src2, cutoff) computeNorm<type2>(src1, (type2*)src2, cutoff)

int Index::searchInBucket(
    InputArray _query,
    size_t hashKey,
    std::priority_queue<point_t>& NNpointQue
) const {
    CV_Assert((_query.getMat().rows == 1 && _query.getMat().cols == dim) || (_query.getMat().rows == dim && _query.getMat().cols == 1));

    bin_t bin;
    hashTable.getBin(hashKey, bin);

    /*set point's pointer*/
    collision_t coll = bin.collision;
    address_t addr = bin.addressOfChainList;
    address_t addr_end = addr + coll*entrySize;
    double KNNdist = NNpointQue.top().distance;
    Mat query = _query.getMat().reshape(1, dim);

    while (addr != addr_end)
    {
        /*Distance Calculation*/
        double dist = 0.0;
        switch (query.depth())
        {
        case CV_8U:  dist = COMPUTE_NORM(unsigned char,  query, addr, KNNdist); break;
        case CV_8S:  dist = COMPUTE_NORM(signed char,    query, addr, KNNdist); break;
        case CV_16U: dist = COMPUTE_NORM(unsigned short, query, addr, KNNdist); break;
        case CV_16S: dist = COMPUTE_NORM(short,          query, addr, KNNdist); break;
        case CV_32S: dist = COMPUTE_NORM(int,            query, addr, KNNdist); break;
        case CV_32F: dist = COMPUTE_NORM(float,          query, addr, KNNdist); break;
        case CV_64F: dist = COMPUTE_NORM(double,         query, addr, KNNdist); break;
        default:
            CV_Error(Error::StsUnsupportedFormat, "unsupported type");
            break;
        }

        if (dist < KNNdist)
        {
            NNpointQue.pop();

            NNpointQue.push(
                point_t(
                    *reinterpret_cast<index_t*>(addr + pointSize),
                    dist)
            );

            KNNdist = NNpointQue.top().distance;
        }
        addr += entrySize;
    }

    return coll;
}

///////////// Search Function ////////////////////////
void Index::linearSearchInNNcandidates(InputArray _query, point_t * point, int K, double epsilon, std::vector<hashKey_t>& bucketList) const
{
    // search the nearest point from bucket based on generated hash key start//
    // priority queue to hold the nearest point
    priority_queue<point_t> NNpointQue;
    // initialize the priority queue
    for (int i = 0; i < K; ++i)
    {
        NNpointQue.push(point_t(ULLONG_MAX, epsilon));
    }

    // search the nearest point using the hash key
    vector<hashKey_t>::iterator keyList_itr = bucketList.begin();
    vector<hashKey_t>::iterator keyList_itr_end = bucketList.end();
    for (; keyList_itr != keyList_itr_end; ++keyList_itr)
    {
        searchInBucket(_query, (*keyList_itr).hashKey, NNpointQue);
    }
    // search the nearest point from bucket based on generated hash key end//

    // copy the nearest point from the priority queue
    for (int i = K - 1; i >= 0; --i)
    {
        point[i] = NNpointQue.top();
        NNpointQue.pop();
    }
}



int Index::NearestNeighbor(
    InputArray _query,
    point_t* point,
    double searchParam,
    search_mode searchMode,
    int K,
    double epsilon
)const
{

    // Generate the Bucket Hash key of the needle (begin)//
    vector<hashKey_t> bucketList;
    int NNC = getBucketList(_query, searchParam, searchMode, bucketList);

    // Generate the Bucket Hash key of the needle (end)//
    linearSearchInNNcandidates(_query, point, K, epsilon, bucketList);

    return NNC;// Number of points used to compute the distance
}

bool readCorrectClass(const String& filename, std::vector<int>& correctClass)
{
    ifstream ifs(filename, ios::in);
    if (ifs.is_open() == false)
    {
        return false;
    }
    correctClass.clear();

    while (ifs.eof() == false)
    {
        int index, classNumber;
        ifs >> index;
        ifs >> classNumber;
        assert(correctClass.size() == index);
        correctClass.push_back(classNumber);
    }

    ifs.close();
    return true;
}

bool readBinary(const String &path, unsigned &dim, unsigned &num, OutputArray data, int type)
{
    ifstream ifs(path, ios::in | ios::binary);
    if (ifs.is_open() == false)
    {
        return false;
    }

    {
        uint32_t dimension, dataSize;
        ifs.read((char*)&dimension, sizeof(dimension));
        ifs.read((char*)&dataSize, sizeof(dataSize));
        dim = dimension;
        num = dataSize;
    }

    data.create(num, dim, type);
    Mat stub = data.getMat();

    for (size_t n = 0; n < num; n++)
    {
        ifs.read((char*)stub.row((int)n).data, CV_ELEM_SIZE(type)*dim);
    }
    ifs.close();

    return true;
}

template <typename data_t>
void parameterTuning(int dim, index_t num, const cv::Mat& data, base_t * const base, int P, int bit, int &M, size_t &hashSize, size_t &pointSize, size_t &entrySize, HashTable &hashTable, double &delta, std::vector<Subspace> &subspace, Subspace& lestspace, double bit_step = 1.0)
{
    hashSize = (size_t(1) << bit);//hash size is 2^bit
    Subspace::dim = dim;

    pointSize = sizeof(data_t)*dim;//byte size of a point's value
    entrySize = pointSize + sizeof(index_t);//byte size to entry a point into hash table

    double variance = 0;
    for (int d = 0; d < dim; ++d)
    {
        variance += base[d].variance;   // pca.eigenvalues
    }
    delta = variance / deltaRate;

    hashTable.initialize(entrySize, hashSize);

    BDHtraining<data_t> BDHtrainer;

    BDHtrainer.training(
        dim, num, data,
        base, P, bit, bit_step
    );

    M = BDHtrainer.getM();
    const baseset_t* const baseSet = BDHtrainer.getBaseSet();
    const baseset_t& lestSet = BDHtrainer.getLestSet();

    size_t rank = 1;

    for (int m = 0; m < M; ++m)
    {
        Subspace stub;
        stub.setParameters(baseSet[m]);

        for (int i = 0; i < stub.subHashSize; ++i)
        {
            stub.hashKey.push_back(rank*i);
        }

        rank *= stub.subHashSize;
        subspace.push_back(stub);
    }

    lestspace.subDim = lestSet.subDim;
    lestspace.variance = lestSet.variance;

    std::vector<double> stubCentroid;
    for (int d = 0; d < lestspace.subDim; ++d)
    {
        stubCentroid.push_back(lestSet.base[d].mean);
        Mat stub(1, dim, CV_64FC1, lestSet.base[d].direction);
        lestspace.baseVector.push_back(stub);
    }
    lestspace.centroidVector.push_back(stubCentroid);

}

Index::Index(InputArray data, PCA::Flags order)
    : dim(0)
    , M(0)
    , P(10)
    , bit(0)
    , delta(0.0)
    , pointSize(0)
    , entrySize(0)
    , hashSize(0)
    , hashTable()
{
    Build(data, order);
}

#define CV_PARAMETER_TUNING(Tp, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15) parameterTuning<Tp>(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15);

void Index::Build(InputArray data, PCA::Flags order)
{
    cv::Mat _data = data.getMat();
    if (order == PCA::DATA_AS_ROW)
    {
        originalData = _data.clone();
    }
    else
    {
        originalData = _data.t();
    }
    cv::PCA pca(originalData, Mat(), order, order == PCA::DATA_AS_ROW ? originalData.rows : originalData.cols);

    int length = originalData.rows;
    dim = originalData.cols;

    // copy PCA direction to base_t for BDH
    base_t* base = new base_t[dim];
    for (int d = 0; d < dim; ++d)
    {
        base[d].mean = (double)pca.mean.at<float>(d);
        base[d].variance = pca.eigenvalues.at<float>(d);
        base[d].direction = new double[dim];    // pca.eigenvectors
        for (int x = 0; x < dim; x++)
        {
            base[d].direction[x] = (double)((float*)(pca.eigenvectors.data + d * pca.eigenvectors.step))[x];
        }
    }
    switch (_data.depth())
    {
    case CV_8U:  CV_PARAMETER_TUNING(unsigned char , dim, length, _data, base, P, 13, M, hashSize, pointSize, entrySize, hashTable, delta, subspace, lestspace, 0.1); break;
    case CV_8S:  CV_PARAMETER_TUNING(char          , dim, length, _data, base, P, 13, M, hashSize, pointSize, entrySize, hashTable, delta, subspace, lestspace, 0.1); break;
    case CV_16U: CV_PARAMETER_TUNING(unsigned short, dim, length, _data, base, P, 13, M, hashSize, pointSize, entrySize, hashTable, delta, subspace, lestspace, 0.1); break;
    case CV_16S: CV_PARAMETER_TUNING(short         , dim, length, _data, base, P, 13, M, hashSize, pointSize, entrySize, hashTable, delta, subspace, lestspace, 0.1); break;
    case CV_32S: CV_PARAMETER_TUNING(int           , dim, length, _data, base, P, 13, M, hashSize, pointSize, entrySize, hashTable, delta, subspace, lestspace, 0.1); break;
    case CV_32F: CV_PARAMETER_TUNING(float         , dim, length, _data, base, P, 13, M, hashSize, pointSize, entrySize, hashTable, delta, subspace, lestspace, 0.1); break;
    case CV_64F: CV_PARAMETER_TUNING(double        , dim, length, _data, base, P, 13, M, hashSize, pointSize, entrySize, hashTable, delta, subspace, lestspace, 0.1); break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "unsupported type");
        break;
    }

    //delete base
    if (base != NULL)
    {
        for (int d = 0; d < dim; ++d)
        {
            if (base[d].direction != NULL)
            {
                delete[] base[d].direction;
                base[d].direction = NULL;
            }
        }
        delete[] base;
        base = NULL;
    }

    // entory data points into hash table
    storePoint();

}

bool Index::loadTable(const String & path)
{
    return hashTable.readTable(path);
}

bool Index::saveTable(const String & path) const
{
    return hashTable.writeTable(path);
}

bool Index::loadParameters(const String& path)
{

    ifstream ifs(path);
    if (ifs.is_open() == false)
    {
        return false;
    }

    ifs >> dim
        >> M
        >> P
        >> bit
        >> delta
        >> pointSize
        >> entrySize
        >> hashSize;

    Subspace::dim = dim;

    for (int m = 0; m < M; ++m)
    {
        Subspace stubSpace;
        stubSpace.subDim = P;

        ifs >> stubSpace.subHashSize
            >> stubSpace.variance;

        for (int sd = 0; sd < P; ++sd)
        {
            Mat stub(1, dim, CV_64FC1);
            for (int d = 0; d < dim; ++d)
            {
                double v;
                ifs >> v;
                stub.at<double>(0, d) = v;
            }
            stubSpace.baseVector.push_back(stub);
        }

        for (int i = 0; i < stubSpace.subHashSize; ++i)
        {
            size_t stubHash;
            double c;
            ifs >> c >> stubHash;
            stubSpace.hashKey.push_back(stubHash);
            stubSpace.cellVariance.push_back(c);

            std::vector<double> stub;
            for (int d = 0; d < P; ++d)
            {
                double v;
                ifs >> v;
                stub.push_back(v);
            }
            stubSpace.centroidVector.push_back(stub);
        }
        subspace.push_back(stubSpace);
    }

    ifs >> lestspace.subDim
        >> lestspace.variance;

    std::vector<double> stubDouble;
    for (int sd = 0; sd < lestspace.subDim; ++sd)
    {
        double v;
        ifs >> v;
        stubDouble.push_back(v);
    }
    lestspace.centroidVector.push_back(stubDouble);
    lestspace.cellVariance.push_back(lestspace.variance);

    for (int sd = 0; sd < lestspace.subDim; ++sd)
    {
        Mat stub(1, dim, CV_64FC1);
        for (int i = 0; i < dim; i++)
        {
            double v;
            ifs >> v;
            stub.at<double>(0, i) = v;
        }
        lestspace.baseVector.push_back(stub);
    }

    ifs.close();

    hashTable.initialize(entrySize, hashSize);
    return true;
}


bool Index::saveParameters(const String& path) const
{

    ofstream ofs(path);
    if (ofs.is_open() == false)
    {
        return false;
    }

    ofs << dim << "\t"
        << M << "\t"
        << P << "\t"
        << bit << "\t"
        << delta << "\t"
        << pointSize << "\t"
        << entrySize << "\t"
        << hashSize << endl;

    for (int m = 0; m < M; ++m)
    {
        ofs << subspace[m].subHashSize << "\t"
            << subspace[m].variance << endl;

        for (int sd = 0; sd < P; ++sd)
        {
            for (int d = 0; d < dim; ++d)
            {
                ofs << subspace[m].baseVector.at<double>(sd, d) << "\t";
            }
            ofs << endl;
        }

        for (int i = 0; i < subspace[m].subHashSize; ++i)
        {

            ofs << subspace[m].cellVariance[i] << "\t" << subspace[m].hashKey[i] << endl;

            for (int d = 0; d < subspace[m].subDim; ++d)
            {
                ofs << subspace[m].centroidVector[i][d] << "\t";
            }

            ofs << endl;
        }
    }

    ofs << lestspace.subDim << "\t"
        << lestspace.variance << endl;
    for (int sd = 0; sd < lestspace.subDim; ++sd)
    {
        ofs << lestspace.centroidVector[0][sd] << "\t";
    }
    ofs << endl;

    for (int sd = 0; sd < lestspace.subDim; ++sd)
    {
        for (int d = 0; d < dim; ++d)
        {
            ofs << lestspace.baseVector.at<double>(sd, d) << "\t";
        }
        ofs << endl;
    }


    ofs.close();
    return true;
}

void Index::storePoint(/*index_t num, data_t** data*/)
{
    //alloc workspace
    collision_t* collision = new collision_t[hashSize];
    memset(collision, 0, sizeof(collision_t)*hashSize);

    size_t* hashKey = new size_t[originalData.rows];
    for (int n = 0; n < originalData.rows; n++)
    {
        //get hash value
        hashKey[n] = hashFunction(n);
        //increment collision
        ++collision[hashKey[n]];
    }

    hashTable.allocTable(collision);
    delete[] collision;

    char* entry = new char[entrySize];
    if (entry == nullptr)
    {
        exit(__LINE__);
    }

    for (int n = 0; n < originalData.rows; ++n)
    {
        memcpy(entry, originalData.row(n).data, pointSize);
        *reinterpret_cast<index_t*>(entry + pointSize) = n;

        hashTable.storeEntryWithoutAlloc(hashKey[n], entry);
    }
    delete[] entry;

    delete[] hashKey;
}

template<typename data_t>
double innerProduct(const std::vector<double>& base, const data_t* data)
{
    double val = 0.0;
    for (size_t i = 0; i < base.size(); i++)
    {
        val += base[i] * data[i];
    }
    return val;
}

template<typename base_t, typename data_t>
double innerProduct(const base_t *base, const data_t* data, size_t length)
{
    double val = 0.0;
    for (size_t i = 0; i < length; i++)
    {
        val += base[i] * data[i];
    }
    return val;
}

#define INNER_PRODUCT(index,dst,src1,src2,type2) for(index = 0;index < src1.rows;index++) { dst[index] = innerProduct((double*)src1.row(index).data, (type2*)src2.data, src1.cols); }

void Subspace::getPCAdata(const Mat &data, double* PCAdata) const
{
    int i = 0;
    switch (data.depth())
    {
    case CV_8U:  INNER_PRODUCT(i, PCAdata, baseVector, data, unsigned char ); break;
    case CV_8S:  INNER_PRODUCT(i, PCAdata, baseVector, data, char          ); break;
    case CV_16U: INNER_PRODUCT(i, PCAdata, baseVector, data, unsigned short); break;
    case CV_16S: INNER_PRODUCT(i, PCAdata, baseVector, data, short         ); break;
    case CV_32S: INNER_PRODUCT(i, PCAdata, baseVector, data, int           ); break;
    case CV_32F: INNER_PRODUCT(i, PCAdata, baseVector, data, float         ); break;
    case CV_64F: INNER_PRODUCT(i, PCAdata, baseVector, data, double        ); break;
    default:
        CV_Error(Error::StsUnsupportedFormat, "unsupported type");
        break;
    }
}

#undef INNER_PRODUCT

void cv::bdh::Subspace::getPCAdata(const Mat & data, Mat & PCAdata) const
{
    CV_Assert(data.size() == PCAdata.size());
    CV_Assert(data.type() == PCAdata.type());
    Mat stub = PCAdata.reshape(1, 1);
    for (int i = 0; i < baseVector.rows; i++)
    {
        stub.col(i) = baseVector.row(i).dot(data);
    }
}

size_t Index::hashFunction(int index)
{

    size_t hashKey = 0;
    for (int m = 0; m < M; ++m)
    {
        hashKey += subspace[m].getSubHashValue(originalData.row(index));
    }
    return hashKey;
}


int Index::getBucketList(
    InputArray _query,
    double searchParam,
    search_mode searchMode,
    std::vector<hashKey_t>& bucketList
    )const
{
    Mat query = _query.getMat();

    // compute the subspace distance and sort based on the priority
    CV_Assert(0 < M);
    layer_t* layer = new layer_t[M];
    for (int m = 0; m < M; ++m)
    {
        layer[m].node = new node_t[subspace[m].subHashSize + 1];
        layer[m].node[subspace[m].subHashSize].distance = DBL_MAX;
    }
    setLayerParam(layer, query);

    unsigned NNC = 0;
    status_t status;
    switch (searchMode)
    {
    case Radius:
    {
        // compute the distance from the centroid of each eigen vector
        // this computation becomes trivial if the number of dimension gets bigger compared to number of the node
        double* lestSpaceVal = new double[lestspace.dim];
        lestspace.getPCAdata(query, lestSpaceVal);
        delete[] lestSpaceVal;

        // search distance smaller than radius
        NNC = NearBucket_R(searchParam, layer, status, bucketList);

        break;
    }
    case NumPoints:
    {
        unsigned C = static_cast<unsigned>(searchParam);
        bucketList.reserve(C);

        for (double Lbound = 0, Ubound = layer[0].restMin + 1.0e-10
            ; NNC < C
            ; Ubound += delta)
        {
            NNC += NearBucket_C(Lbound, Ubound, layer, status, bucketList);
            Lbound = Ubound;
        }

        break;
    }

    case NumPoints2:
    {
        unsigned C = static_cast<unsigned>(searchParam);
        bucketList.reserve(C);

        // resume the cut-offed search
        list<status_t> statusQue;
        statusQue.push_front(status);// root node
        list<status_t>::iterator itr;
        for (double Rbound = layer[0].restMin + 1.0e-10
            ; NNC < C
            ; Rbound += delta)
        {
            size_t loop = statusQue.size();
            itr = statusQue.begin();
            for (size_t l = 0; l < loop; ++l)
            {
                NNC += NearBucket_C_list(Rbound, layer, statusQue, &itr, bucketList);
            }
        }
        break;
    }
    }

    // clean up
    for (int m = 0; m < M; ++m)
    {
        delete[] layer[m].node;
    }
    delete[] layer;

    return NNC;
}
} // bdh
} // cv
