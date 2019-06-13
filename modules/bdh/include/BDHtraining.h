/**
* @file BDHtraining.h
* @author Tomokazu Sato
* @date 2015/05/04
*/

#ifndef __BDH_TRAINING__
#define __BDH_TRAINING__

#include <baseset.h>
#include <string>
#include <opencv2/bdh.hpp>
using namespace std;

#include "k_means.h"


namespace cv {
namespace bdh {
    /**
    * @brief training BDH parameters
    */
    template <typename data_t>
    class BDHtraining
    {
        //for data sample
        int dim;      //!< dimension of dataspace
        unsigned num; //!< number of data samples

        //parameters
        int M;   //!< number of subspace
        int P;   //!< dimension of subspace
        int U;   //!< numer of base used for hashing = P*M
        int bit; //!< bits num of hash table

        //result values
        size_t hashSize;                //!< hash size
        std::vector<baseset_t> baseSet; //!< baseSet[M]
        baseset_t lestSet;              //!< baseSet not used for hashing

    public:

        /**
        * @brief default constructor
        */
        BDHtraining()
            : dim(0)
            , num(0)
            , M(0)
            , P(0)
            , U(0)
            , bit(0)
            , hashSize(0)
            , lestSet()
        {}

        /**
        * @brief destructor
        */
        ~BDHtraining()
        {
            baseSet.clear();
            lestSet.clear();
        }

        int getM()
        {
            return M;
        }

        /**
        * @brief training BDH parameters
        */
        void training(
            int _dim,                             //!< [in] dimension
            unsigned _num,                        //!< [in] number of sample
            const cv::Mat& data,                  //!< [in] sample data set
            const std::vector<base_t>& baseInput, //!< [in] base
            int _P,                               //!< [in] number of subspace
            int _bit,                             //!< [in] bits num of hash table
            double bit_step = 1.0                 //!< [in] training parameter.
        );

        /**
        * @brief getter
        * @return baseSet
        */
        const std::vector<baseset_t>& getBaseSet() const 
        {
            return baseSet;
        }

        /**
        * @brief training BDH parameters
        * @return lsetSet
        */
        const baseset_t& getLestSet()
        {
            return lestSet;
        }

        /**
        * @brief training BDH parameters
        * @return is file open
        */
        bool saveParameters(
            const string& path//!< [in] file path
        );

    private:

        // divide the data space in to M subspace
        void BDHtraining<data_t>::partitioningDataspace(const std::vector<base_t>& base);


        // compute the centroid of each subspace
        // the number of centroid will be computed automatically
        void calclateCentroid(
            float*** subPrjData,
            double bit_step
        );

        // compute the centroid of each subspace
        // the number of centroid will be computed automatically
        void calclateCentroid_ICCV2013(
            float*** subPrjData,
            double bit_step
        );

        void updateCentroid(
            double bit_step,
            float*** subPrjData,
            K_Means<float, double>*& k_means);

        // compute the variance of each cell for distance estimation
        void calculateCellVariance(
            float*** subPrjData
        );

    };
    int base_t::dim;

    template<typename data_t>
    double innerProduct(const std::vector<double>& base, const data_t* data)
    {
        double val = 0.0;
        for (int d = 0; d < base.size(); ++d)
        {
            val += base[d] * data[d];
        }
        return val;
    }

    template<typename data_t>
    double innerProduct(const double* base, const data_t* data, int length)
    {
        double val = 0.0;
        for (int d = 0; d < length; ++d)
        {
            val += base[d] * data[d];
        }
        return val;
    }

    template <typename data_t>
    void BDHtraining<data_t>::training(
        int _dim,
        unsigned _num,
        const cv::Mat& data,
        const std::vector<base_t>& baseInput,
        int _P,
        int _bit,
        double bit_step)
    {
        dim = _dim;
        num = _num;
        P = _P;
        bit = _bit;

        if (bit_step <= 0.0)
        {
            bit_step = 0.1;
        }

        base_t::dim = dim;
        int M_max = max(min(dim / P, bit), 1);
        M = M_max;
        CV_Assert(0 < M);
        partitioningDataspace(baseInput);

        float*** subPrjData = new float**[M_max];
        const auto stride = data.step.p[0];
        int length = (int)baseSet[0].base[0].direction.size();
        for (int d, m = 0; m < M_max; ++m)
        {
            subPrjData[m] = new float*[num];
            base_t* stub = baseSet[m].base;
            for (unsigned n = 0; n < num; ++n)
            {
                subPrjData[m][n] = new float[P];
                const data_t* data_p = (const data_t*)(data.data + stride*n);
                for (d = 0; d < P; ++d)
                {
                    // pca.eigenvectors
                    subPrjData[m][n][d]
                        = static_cast<float>(
                            innerProduct<data_t>((const double*)(&stub[d].direction[0]), data_p, length)
                            );
                }
            }
        }

        calclateCentroid_ICCV2013(subPrjData, bit_step);

        calculateCellVariance(subPrjData);

        for (int m = 0; m < M_max; m++)
        {
            for (unsigned n = 0; n < num; n++)
            {
                delete[] subPrjData[m][n];
            }
            delete[] subPrjData[m];
        }
        delete[] subPrjData;

        lestSet.subDim = dim - U;
        lestSet.base = new base_t[lestSet.subDim];
        lestSet.variance = 0;
        for (int d = 0; d < lestSet.subDim; ++d) {

            lestSet.base[d] = baseInput[U + d];
            // pca.eigenvectors
            lestSet.base[d].direction.assign(dim, 0);
            copy(baseInput[U + d].direction.begin(), baseInput[U + d].direction.end(), lestSet.base[d].direction.begin());
            // pca.eigenvalues
            lestSet.variance += baseInput[U + d].variance;
        }
    }


    template <typename data_t>
    void BDHtraining<data_t>::partitioningDataspace(const std::vector<base_t>& base)
    {

        for (int m = 0; m < M; ++m)
        {
            baseset_t stub;
            stub.variance = 0;
            stub.subDim = 0;
            stub.base = new base_t[P];
            baseSet.push_back(stub);
        }

        for (int m = 0, d = 0; m < M; ++m)
        {
            for (int p = 0; p < P; ++p, ++d)
            {
                // pca.eigenvalues
                baseSet[m].variance += base[d].variance;
                baseSet[m].base[baseSet[m].subDim] = base[d];
                // pca.eigenvectors
                baseSet[m].base[baseSet[m].subDim].direction.assign(dim, 0);

                copy(base[d].direction.begin(), base[d].direction.end(), baseSet[m].base[baseSet[m].subDim].direction.begin());

                ++baseSet[m].subDim;
            }
        }
    }

    inline double square(double x)
    {
        return x*x;
    }

    template <typename data_t>
    void BDHtraining<data_t>::calclateCentroid(
        float*** subPrjData,
        double bit_step
    )
    {
        K_Means<float, double>* k_means = new K_Means<float, double>[M];
        for (int m = 0; m < M; m++)
        {
            k_means[m].calclateCentroid(P, num, subPrjData[m], 2);

            baseSet[m].error = k_means[m].getError();
            baseSet[m].bit = 1;
            baseSet[m].k = 2;
        }

        int tmp = 5;
        int loop = static_cast<int>(((bit - M) / bit_step));
        for (int i = 0; i < loop; ++i)
        {
            updateCentroid(bit_step, subPrjData, k_means);

            int percent = static_cast<int>(((bit_step*i + M) / bit) * 100.);
            if (percent >= tmp)
            {
                cout << percent << "% done." << endl;
                tmp += 5;
            }
        }

        updateCentroid(bit - (M + bit_step*loop), subPrjData, k_means);

        cout << "index\tbit\tk\terror" << endl;

        for (int m = 0; m < M; m++)
        {
            cout << m << "\t" << baseSet[m].bit << "\t" << baseSet[m].k << "\t" << baseSet[m].error << endl;
        }
        cout << endl;

        hashSize = 1;
        for (int m = 0; m < M; m++)
        {
            hashSize *= baseSet[m].k;
            baseSet[m].centroid = new double*[baseSet[m].k];

            double** const tmpCent = k_means[m].getCentroid();
            for (int i = 0; i < baseSet[m].k; ++i)
            {
                baseSet[m].centroid[i] = new double[P];
                memcpy(baseSet[m].centroid[i], tmpCent[i], sizeof(double)*P);
            }
        }

        delete[] k_means;
    }

    template <typename data_t>
    void BDHtraining<data_t>::calclateCentroid_ICCV2013(
        float*** subPrjData,
        double bit_step
    )
    {

        K_Means<float, double>* k_means = new K_Means<float, double>[M];
        for (int m = 0; m < M; m++)
        {
            baseSet[m].error = baseSet[m].variance;
            baseSet[m].bit = 0;
            baseSet[m].k = 1;
        }

        int percent;
        int tmp = 5;
        int loop = static_cast<int>(bit / bit_step);
        double restBit = bit - bit_step*loop;
        for (int i = 0; i < loop; ++i)
        {
            updateCentroid(bit_step, subPrjData, k_means);
            percent = static_cast<int>((bit_step*i / bit) * 100.);
            if (percent >= tmp)
            {
                cout << percent << "% done. " << endl;
                tmp += 5;
            }
        }

        double nouseBit = 0;
        for (int m = 0; m < M; ++m)
        {
            if (baseSet[m].k == 1)
            {
                M = m;
                U = P*M;
                nouseBit = baseSet[m].bit;
                break;
            }
        }

        loop = static_cast<int>(nouseBit / bit_step + 0.5);
        for (int i = 0; i < loop; ++i)
        {
            updateCentroid(bit_step, subPrjData, k_means);
            percent = static_cast<int>(square(bit_step*i / bit) * 100);
            if (percent >= tmp)
            {
                cout << percent << "% done." << endl;
                tmp += 5;
            }
        }
        updateCentroid(restBit, subPrjData, k_means);

        cout << "index\tbit\tk\terror" << endl;
        hashSize = 1;
        for (int m = 0; m < M; m++)
        {
            hashSize *= baseSet[m].k;
            baseSet[m].centroid = new double*[baseSet[m].k];

            double** const tmpCent = k_means[m].getCentroid();
            for (int i = 0; i < baseSet[m].k; ++i)
            {
                baseSet[m].centroid[i] = new double[P];
                memcpy(baseSet[m].centroid[i], tmpCent[i], sizeof(double)*P);
            }
            cout << m << "\t" << baseSet[m].bit << "\t" << baseSet[m].k << "\t" << baseSet[m].error << endl;
        }
        cout << endl;

        delete[] k_means;
    }

    template<typename data_t, typename query_t>
    int NearestNeighbor(
        int dim,
        int num,
        data_t** sample,
        query_t* query
    )
    {
        point_t NNpoint;
        NearestNeighbor(dim, num, sample, query, NNpoint);
        return NNpoint.index;
    }

    template<typename data_t, typename query_t>
    void NearestNeighbor(
        int dim,
        int num,
        data_t** sample,
        query_t* query,
        point_t& NNpoint
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


    template <typename data_t>
    void BDHtraining<data_t>::calculateCellVariance(
        float*** subPrjData
    )
    {

        point_t point;
        for (int m = 0; m < M; ++m)
        {
            int* count = new int[baseSet[m].k];

            baseSet[m].cellVariance = new double[baseSet[m].k];
            for (int r = 0; r < baseSet[m].k; ++r)
            {
                count[r] = 0;
                baseSet[m].cellVariance[r] = 0.0;
            }

            for (unsigned n = 0; n < num; n++)
            {
                NearestNeighbor(
                    baseSet[m].subDim,
                    baseSet[m].k,
                    baseSet[m].centroid,
                    subPrjData[m][n],
                    point
                );

                ++count[point.index];
                baseSet[m].cellVariance[point.index] += point.distance;
            }

            for (int r = 0; r < baseSet[m].k; r++)
            {
                baseSet[m].cellVariance[r] /= count[r];
            }

            delete[] count;
        }
    }


    template <typename data_t>
    bool BDHtraining<data_t>::saveParameters(const string& path)
    {
        ofstream ofs(path);

        if (ofs.is_open() == false)
        {
            return false;
        }

        ofs << dim << "\t"
            << num << "\t"
            << M << "\t"
            << P << "\t"
            << U << "\t"
            << bit << "\t"
            << hashSize << endl;
        for (int m = 0; m < M; ++m)
        {

            ofs << baseSet[m].idx << "\t"
                << baseSet[m].subDim << "\t"
                << baseSet[m].variance << "\t"
                << baseSet[m].k << "\t"
                << baseSet[m].bit << "\t"
                << baseSet[m].error << endl;

            for (int d = 0; d < baseSet[m].subDim; ++d)
            {
                ofs << baseSet[m].base[d].idx << "\t"
                    << baseSet[m].base[d].mean << "\t"
                    << baseSet[m].base[d].variance << "\t";

                for (int d2 = 0; d2 < dim; ++d2)
                {
                    ofs << baseSet[m].base[d].direction[d2] << "\t";
                }
                ofs << endl;

            }

            for (int i = 0; i < baseSet[m].k; ++i)
            {
                ofs << baseSet[m].cellVariance[i] << endl;
                for (int d = 0; d < baseSet[m].subDim; ++d)
                {
                    ofs << baseSet[m].centroid[i][d] << "\t";
                }
                ofs << endl;
            }

        }

        ofs << lestSet.subDim << "\t"
            << lestSet.variance << endl;
        for (int d = 0; d < lestSet.subDim; ++d)
        {
            ofs << lestSet.base[d].mean << "\t";
            cout << __LINE__ << "\t" << lestSet.base[d].mean << endl;
        }
        ofs << endl;

        for (int d = 0; d < lestSet.subDim; ++d)
        {
            for (int d2 = 0; d2 < dim; ++d2)
            {
                ofs << lestSet.base[d].direction[d2] << "\t";
            }
            ofs << endl;
        }
        ofs.close();

        return true;
    }

    template <typename data_t>
    void BDHtraining<data_t>::updateCentroid(
        double bit_step,
        float*** subPrjData,
        K_Means<float, double>*& k_means)
    {

        int target = 0;
        for (int m = 1; m < M; m++)
        {
            if (baseSet[m].error > baseSet[target].error)
            {
                target = m;
            }
        }

        baseSet[target].bit += bit_step;
        int k2 = static_cast<int>(pow(2.0, baseSet[target].bit) + 1.0e-10);

        if (baseSet[target].k == k2)
        {
            return;
        }

        baseSet[target].k = k2;
        k_means[target].calclateCentroid(P, num, subPrjData[target], baseSet[target].k);
        baseSet[target].error = k_means[target].getError();

    }
} } // namespace

#endif
