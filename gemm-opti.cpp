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
        void mul_asm_blocking_4x4F32(const Matrix&, const Matrix&);
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


/*
 * This routine impl a 4x4 FP32 kernel, with matrix 1024x1024
 */

void Matrix::mul_asm_blocking_4x4F32(const Matrix& A, const Matrix& B)
{
/*
    for (int outer_x = 0; outer_x < 256; outer_x++)
        for (int outer_y = 0; outer_y < 256; outer_y++)
            for (int k = 0; k < 256; k++)
                for (int inner_x = 0; inner_x < 4; inner_x++)
                    for (int inner_y = 0; inner_y < 4; inner_y++)
                        for (int i = 0; i < 4; i++)
                            data[(outer_x * 4 + inner_x) * y + (outer_y * 4 + inner_y)] += 
                            A.data[(outer_x * 4 + inner_x) * A.y + (k * 4 + i)] *
                            B.data[(k * 4 + i) * B.y + (outer_y * 4 + inner_y)];
*/
    asm volatile(
            "mov x0, %[adata]\n"
            "mov x1, %[bdata]\n"
            "mov x2, %[cdata]\n"
            "mov x28, #0\n"
            "mov x29, #4\n"
            "mov x30, #1024\n"

// Outer loop x

            "mov x3, #0\n"
            "x44_outer_x:\n"

// Outer loop y

            "mov x4, #0\n"
            "x44_outer_y:\n"

// Clear target vector registers

            "dup v0.4s, w28\n"
            "dup v1.4s, w28\n"
            "dup v2.4s, w28\n"
            "dup v3.4s, w28\n"

// Calculate offset for A

            "mul x6, x3, x29\n"
            "mul x6, x6, x30\n"
            "mul x6, x6, x29\n"
            "add x6, x6, x0\n"

// Calculate offset for B

            "mul x7, x4, x29\n"
            "mul x7, x7, x29\n"
            "add x7, x7, x1\n"

// Inner loop K
// Pre-load data for K = 0

            "ldr q4, [x6]\n"
            "ldr q5, [x6, #4096]\n"
            "ldr q6, [x6, #8192]\n"
            "ldr q7, [x6, #12288]\n"

            "ldr q8, [x7]\n"
            "ldr q9, [x7, #4096]\n"
            "ldr q10, [x7, #8192]\n"
            "ldr q11, [x7, #12288]\n"

            "mov x5, #0\n"
            "x44_k:\n"
/*
            "ldr q4, [x6]\n"
            "ldr q5, [x6, #4096]\n"
            "ldr q6, [x6, #8192]\n"
            "ldr q7, [x6, #12288]\n"

            "ldr q8, [x7]\n"
            "ldr q9, [x7, #4096]\n"
            "ldr q10, [x7, #8192]\n"
            "ldr q11, [x7, #12288]\n"
*/
            // ldr Q-form registers cost 6 Cycles     L
            // fmul Q-form registers cost 4 Cycles    F0/F1
            // fmla Q-form registers cost 7-3 Cycles  F0/F1
            // fadd Q-form registers cost 4 Cycles    F0/F1
            // L F0 F1, can parallelize
            //
// First dot 

            "fmul v12.4s, v8.4s, v4.4s[0]\n"
            "fmul v13.4s, v8.4s, v5.4s[0]\n"
            "fmul v14.4s, v8.4s, v6.4s[0]\n"
            "fmul v15.4s, v8.4s, v7.4s[0]\n"
            "add x7, x7, #16384\n"
            "add x6, x6, #16\n"

// Second dot

            "fmla v12.4s, v9.4s, v4.4s[1]\n"
            "ldr q8, [x7]\n"
            "fmla v13.4s, v9.4s, v5.4s[1]\n"
            "fmla v14.4s, v9.4s, v6.4s[1]\n"
            "fmla v15.4s, v9.4s, v7.4s[1]\n"

// Third dot

            "fmla v12.4s, v10.4s, v4.4s[2]\n"
            "ldr q9, [x7, #4096]\n"
            "fmla v13.4s, v10.4s, v5.4s[2]\n"
            "fmla v14.4s, v10.4s, v6.4s[2]\n"
            "fmla v15.4s, v10.4s, v7.4s[2]\n"

// Firth dot

            "fmla v12.4s, v11.4s, v4.4s[3]\n"
            "ldr q10, [x7, #8192]\n"
            "fmla v13.4s, v11.4s, v5.4s[3]\n"
            "ldr q4, [x6]\n"
            "fmla v14.4s, v11.4s, v6.4s[3]\n"
            "ldr q5, [x6, #4096]\n"
            "fmla v15.4s, v11.4s, v7.4s[3]\n"
            "ldr q6, [x6, #8192]\n"

// Add to target vector register

            "fadd v0.4s, v0.4s, v12.4s\n"
            "fadd v1.4s, v1.4s, v13.4s\n"
            "ldr q7, [x6, #12288]\n"
            "fadd v2.4s, v2.4s, v14.4s\n"
            "fadd v3.4s, v3.4s, v15.4s\n"
            "ldr q11, [x7, #12288]\n"

            //"add x6, x6, #16\n"
            //"add x7, x7, #16384\n"

            "add x5, x5, #1\n"
            "cmp x5, #256\n"
            "bne x44_k\n"

// Inner loop complete, write block results to memory

            "mul x6, x3, x29\n"
            "mul x6, x6, x30\n"
            "mul x7, x4, x29\n"
            "add x6, x6, x7\n"
            "mul x6, x6, x29\n"
            "add x6, x6, x2\n"

            "str q0, [x6]\n"
            "str q1, [x6, #4096]\n"
            "str q2, [x6, #8192]\n"
            "str q3, [x6, #12288]\n"

            "add x4, x4, #1\n"
            "cmp x4, #256\n"
            "bne x44_outer_y\n"

            "add x3, x3, #1\n"
            "cmp x3, #256\n"
            "bne x44_outer_x\n"

            :
            :[adata]"r"(A.data), [bdata]"r"(B.data), [cdata]"r"(data)
            :"x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x28", "x29", "x30"
            );
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

    matC.clear();
    timer.start();
    matC.mul_asm_blocking_4x4F32(matA, matB);
    timer.end();
    timer.show_gap();
    printf("\n%f\n", matC.data[debug_index]);

    matB.transpose();
    matC.clear();
    timer.start();
    matC.mul_neon(matA, matB);
    timer.end();
    timer.show_gap();
    printf("\n%f\n", matC.data[debug_index]);


    return 0;
}



