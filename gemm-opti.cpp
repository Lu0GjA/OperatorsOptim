#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <sys/time.h>
#include <arm_neon.h>


class MiliTimer
{
    public:
        MiliTimer();
        struct timeval ts1;
        struct timeval ts2;
        int64_t gap;

        void start(void);
        void end(void);
        void show_gap(void);


};


MiliTimer::MiliTimer()
{
    memset(&ts1, 0, sizeof(struct timeval));
    memset(&ts2, 0, sizeof(struct timeval));
    gap = 0;
}


void MiliTimer::start(void)
{
    gettimeofday(&ts1, NULL);
}


void MiliTimer::end(void)
{
    gettimeofday(&ts2, NULL);
    gap = (ts2.tv_sec * 1000 + ts2.tv_usec / 1000) - (ts1.tv_sec * 1000 + ts1.tv_usec / 1000);
}


void MiliTimer::show_gap(void)
{
    printf("%ld ms\n", gap);
}

class Matrix
{
    public:
        float *data;
        int64_t x;
        int64_t y;

        Matrix(const Matrix&);
        Matrix(int64_t, int64_t);
        ~Matrix();
        void transpose(void);
        void clear(void);
        void random(void);
        void print(void);
        
        void mul(const Matrix&, const Matrix&);
        void mul_neon(const Matrix&, const Matrix&);
};


Matrix::Matrix(const Matrix& mat)
{
    this->x = mat.x;
    this->y = mat.y;

    data = (float*)calloc(x * y, sizeof(float));
    memcpy(data, mat.data, x * y * sizeof(float));
}


Matrix::Matrix(int64_t x, int64_t y)
{
    this->x = x;
    this->y = y;
    data = (float*)calloc(x * y, sizeof(float));
}


Matrix::~Matrix()
{
    if (data != NULL)
    {
        free(data);
        data = NULL;
    }
}


void Matrix::transpose(void)
{
    float* tempMat = (float*)calloc(x * y, sizeof(float));
    memcpy(tempMat, data, sizeof(float) * x * y);

    for (int i = 0; i < y; i++)
        for (int j = 0; j < x; j++)
            data[i * x + j] = tempMat[j * y + i];

    x = x + y;
    y = x - y;
    x = x - y;

    free(tempMat);
}


void Matrix::clear(void)
{
    memset(data, 0 , x * y * sizeof(float));
}


void Matrix::random(void)
{
    for (int i = 0; i < x; i++)
        for (int j = 0; j < y; j++)
            data[i * y + j] = rand() % 10;
}


void Matrix::print(void)
{
    printf("\n");

    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
            printf("%6.1f", data[i * y + j]);
        printf("\n");
    }

    printf("\n");
}


void Matrix::mul(const Matrix& a, const Matrix& b)
{
    for (int i = 0; i < x; i++)
        for (int j = 0; j < y; j++)
            for (int k = 0; k < a.y; k++)
                data[i * y + j] += a.data[i * a.y + k] * b.data[k * b.y + j];
}


void Matrix::mul_neon(const Matrix& a, const Matrix& bt)
{
// 32 128-bit vector registers

    float32x4_t a0, a1, a2, a3; // 128bit 32 * 4 FP32
    float32x4_t b0, b1, b2, b3; // 128bit 32 * 4 FP32
    float32x4_t m0, m1, m2, m3; // mi = ai x bi
    float32x4_t s0, s1, s2, s3; // si = Emi

    const float *p2data_start = NULL;

    s0 = vmovq_n_f32(0);
    s1 = vmovq_n_f32(0);
    s2 = vmovq_n_f32(0);
    s3 = vmovq_n_f32(0);

    for (int i = 0; i < x; i++)
        for (int j = 0; j < y; j++)
        {
            for (int k = 0; k < a.y; k = k + 16)
            {
                p2data_start = a.data + (i * a.y + k);
                a0 = vld1q_f32(p2data_start);
                a1 = vld1q_f32(p2data_start + 4);
                a2 = vld1q_f32(p2data_start + 8);
                a3 = vld1q_f32(p2data_start + 12);

                p2data_start = bt.data + (j * bt.y + k);
                b0 = vld1q_f32(p2data_start);
                b1 = vld1q_f32(p2data_start + 4);
                b2 = vld1q_f32(p2data_start + 8);
                b3 = vld1q_f32(p2data_start + 12); 

                m0 = vmulq_f32(a0, b0);
                m1 = vmulq_f32(a1, b1);
                m2 = vmulq_f32(a2, b2);
                m3 = vmulq_f32(a3, b3);

                s0 = vaddq_f32(s0, m0);
                s1 = vaddq_f32(s1, m1);
                s2 = vaddq_f32(s2, m2);
                s3 = vaddq_f32(s3, m3);
            }

            s0 = vaddq_f32(s0, s1);
            s2 = vaddq_f32(s2, s3);

            s0 = vaddq_f32(s0, s2);



            data[i * y + j] += vgetq_lane_f32(s0, 0);
            data[i * y + j] += vgetq_lane_f32(s0, 1);
            data[i * y + j] += vgetq_lane_f32(s0, 2);
            data[i * y + j] += vgetq_lane_f32(s0, 3);

            s0 = vmovq_n_f32(0);
            s1 = vmovq_n_f32(0);
            s2 = vmovq_n_f32(0);
            s3 = vmovq_n_f32(0);
        }
}


int main(int argc, char** argv)
{
    Matrix matA(1024, 1024);
    Matrix matB(1024, 1024);
    Matrix matC(1024, 1024);

    srand((unsigned)time(0));
    int64_t debug_index = rand() % (1024 * 1024);

    MiliTimer timer;

    srand((unsigned int) time(0));

    matA.random();
    matB.random();

    matB.transpose();
    matC.clear();
    timer.start();
    matC.mul_neon(matA, matB);
    timer.end();
    timer.show_gap();
    printf("\n%f\n", matC.data[debug_index]);

    return 0;
}



