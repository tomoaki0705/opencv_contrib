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
        int dim;	//!< dimension of dataspace
        unsigned num;	//!< number of data samples

                        //parameters
        int M;	//!< number of subspace
        int P;	//!< dimension of subspace
        int U;	//!< numer of base used for hashing = P*M
        int bit;//!< bits num of hash table

                //result values
        size_t hashSize;	//!< hash size
        baseset_t* baseSet;	//!< baseSet[M]
        baseset_t lestSet;	//!< baseSet not used for hashing

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
            , baseSet(nullptr)
            , lestSet()
        {}

        /**
        * @brief destructor
        */
        ~BDHtraining()
        {
            if (baseSet != nullptr)
            {
                for (int m = 0; m < M; ++m)
                {
                    baseSet[m].clear();
                }
                delete[] baseSet;
                baseSet = nullptr;
            }
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
            int _dim,						//!< [in] dimension
            unsigned _num,						//!< [in] number of sample
            data_t** data,					//!< [in] sample data set
            const base_t* const baseInput,	//!< [in] base
            int _M,							//!< [in] number of subspace
            int _P,							//!< [in] dimension of subspace
            int _bit,						//!< [in] bits num of hash table
            double bit_step = 1.0			//!< [in] training parameter.
        );

        /**
        * @brief training BDH parameters
        */
        void training_ICCV2013(
            int _dim,						//!< [in] dimension
            unsigned _num,					//!< [in] number of sample
            data_t** data,					//!< [in] sample data set
            const base_t* const baseInput,	//!< [in] base
            int _P,							//!< [in] number of subspace
            int _bit,						//!< [in] bits num of hash table
            double bit_step = 1.0			//!< [in] training parameter.
        );

        /**
        * @brief getter
        * @return baseSet
        */
        const baseset_t* const getBaseSet()
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

        //データ空間をM個の部分空間に分割する
        void partitioningDataspace(
            const base_t* const base
        );

        /**
        *上位の基底セットに上位の主成分をまとめて登録する
        *
        */
        void partitioningDataspace_ICCV2013(
            const base_t* const base
        );


        //各部分空間のセントロイドを求める
        //セントロイドの数は自動的に求められる
        void calclateCentroid(
            float*** subPrjData,
            double bit_step
        );

        //各部分空間のセントロイドを求める
        //セントロイドの数は自動的に求められる
        void calclateCentroid_ICCV2013(
            float*** subPrjData,
            double bit_step
        );

        void updateCentroid(
            double bit_step,
            float*** subPrjData,
            K_Means<float, double>*& k_means);

        //推定距離の計算用に各セル内の分散を求める
        void calculateCellVariance(
            float*** subPrjData
        );

        //内積
        double innerProduct(const double* base, const data_t* data) {
            double val = 0.0;
            for (int d = 0; d < dim; ++d) {
                val += base[d] * data[d];
            }
            return val;
        }


    };
    int base_t::dim;

    template <typename data_t>
    void BDHtraining<data_t>::training_ICCV2013(
        int _dim,
        unsigned _num,
        data_t** data,
        const base_t* const baseInput,
        int _P,
        int _bit,
        double bit_step)
    {
        //delete BDHtraining();

        this->dim = _dim;
        this->num = _num;
        this->P = _P;
        this->bit = _bit;

        if (bit_step <= 0.0)
        {
            bit_step = 0.1;
        }

        base_t::dim = dim;
        int M_max = min(dim / P, bit);
        M = M_max;
        partitioningDataspace_ICCV2013(baseInput);

        float*** subPrjData = new float**[M_max];
        for (int d, m = 0; m < M_max; ++m)
        {
            subPrjData[m] = new float*[num];

            for (unsigned n = 0; n < num; ++n)
            {
                subPrjData[m][n] = new float[P];

                for (d = 0; d < P; ++d)
                {
                    // pca.eigenvectors
                    subPrjData[m][n][d]
                        = static_cast<float>(
                            innerProduct(baseSet[m].base[d].direction, data[n])
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
            lestSet.base[d].direction = new double[dim];
            memcpy(lestSet.base[d].direction, baseInput[U + d].direction, sizeof(double)*dim);
            // pca.eigenvalues
            lestSet.variance += baseInput[U + d].variance;
        }
    }


    template <typename data_t>
    void BDHtraining<data_t>::training(
        int _dim,
        unsigned _num,
        data_t** data,
        const base_t* const baseInput,
        int _M,
        int _P,
        int _bit,
        double bit_step)
    {
        //delete BDHtraining();

        dim = _dim;
        num = _num;
        M = _M;
        P = _P;
        U = M*P;
        bit = _bit;

        if (bit < M)
        {
            this->M = bit;
        }

        if (bit_step <= 0.0)
        {
            bit_step = 0.1;
        }

        base_t::dim = dim;
        partitioningDataspace(baseInput);

        float*** subPrjData = new float**[M];
        for (int d, m = 0; m < M; ++m)
        {
            subPrjData[m] = new float*[num];

            for (unsigned n = 0; n < num; ++n)
            {
                subPrjData[m][n] = new float[P];

                for (d = 0; d < P; ++d)
                {
                    subPrjData[m][n][d]
                        = static_cast<float>(
                            innerProduct(baseSet[m].base[d].direction, data[n])
                            );
                }
            }
        }

        calclateCentroid(subPrjData, bit_step);

        calculateCellVariance(subPrjData);

        for (int m = 0; m < M; m++)
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

            lestSet.base[d].direction = new double[dim];
            memcpy(lestSet.base[d].direction, baseInput[U + d].direction, sizeof(double)*dim);

            lestSet.variance += baseInput[U + d].variance;
        }
    }

    template <typename data_t>
    void BDHtraining<data_t>::partitioningDataspace(
        const base_t* const base)
    {

        baseSet = new baseset_t[M];
        for (int m = 0; m < M; ++m)
        {
            baseSet[m].variance = 0;
            baseSet[m].subDim = 0;
            baseSet[m].base = new base_t[P];
        }

        for (int d = 0; d < U; ++d)
        {
            int target = 0;
            for (int m = 1; m < M; ++m)
            {
                if ((baseSet[target].subDim == P) ||
                    (baseSet[m].variance < baseSet[target].variance && baseSet[m].subDim < P))
                {
                    target = m;
                    continue;
                }
            }

            baseSet[target].variance += base[d].variance;
            baseSet[target].base[baseSet[target].subDim] = base[d];
            baseSet[target].base[baseSet[target].subDim].direction = new double[dim];

            memcpy(
                baseSet[target].base[baseSet[target].subDim].direction,
                base[d].direction,
                sizeof(double)*dim
            );

            ++baseSet[target].subDim;
        }

        std::sort(baseSet, baseSet + M);
    }

    template <typename data_t>
    void BDHtraining<data_t>::partitioningDataspace_ICCV2013(
        const base_t* const base)
    {

        baseSet = new baseset_t[M];
        for (int m = 0; m < M; ++m)
        {
            baseSet[m].variance = 0;
            baseSet[m].subDim = 0;
            baseSet[m].base = new base_t[P];
        }

        for (int m = 0, d = 0; m < M; ++m)
        {
            for (int p = 0; p < P; ++p, ++d)
            {
                // pca.eigenvalues
                baseSet[m].variance += base[d].variance;
                baseSet[m].base[baseSet[m].subDim] = base[d];
                // pca.eigenvectors
                baseSet[m].base[baseSet[m].subDim].direction = new double[dim];

                memcpy(
                    baseSet[m].base[baseSet[m].subDim].direction,
                    base[d].direction,
                    sizeof(double)*dim
                );

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

        int percent;
        int tmp = 5;
        int loop = static_cast<int>(((bit - M) / bit_step));
        for (int i = 0; i < loop; ++i)
        {
            updateCentroid(bit_step, subPrjData, k_means);

            percent = static_cast<int>(square((bit_step*i + M) / bit) * 100);
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
        double previousSquare = 0.0;
        double valueSquare = 0.0;
        double restBit = bit - bit_step*loop;
        for (int i = 0; i < loop; ++i)
        {
            updateCentroid(bit_step, subPrjData, k_means);
            previousSquare = valueSquare;
            valueSquare = square(bit_step*i / bit) * 100.;
            percent = static_cast<int>(valueSquare);
            if (percent >= tmp)
            {
                cout << percent << "% done. " << previousSquare << ',' << valueSquare << endl;
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

        int NNidx = 0;
        double NNdis = Distance(dim, sample[0], query);
        double distance;
        for (int n = 1; n < num; n++) {

            distance = Distance(dim, sample[n], query, NNdis);
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
        point_t& NNpoint
    )
    {
        NNpoint.index = 0;
        NNpoint.distance = Distance(dim, sample[0], query);
        double distance;
        for (int n = 1; n < num; n++)
        {

            distance = Distance(dim, sample[n], query);
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
