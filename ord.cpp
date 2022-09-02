#include <iostream>
// #include "mpi.h"
#include <omp.h>
#include <sys/time.h>
#include <algorithm>
#include <cmath>
using namespace std;

#define MY_PI 3.1415926
#define THREAD_NUM 2  // mpi线程数量
#define n 128  // 要处理的图像大小
#define loop_time 1  // 循环计算的次数
#define filter_box_size 13  // 卷积盒的大小

// 向量数据结构
struct Vector3f
{
    float x, y, z;  // 3个维度

    // 下面是各种运算符重载
    Vector3f operator+(const Vector3f& b)
    {
        Vector3f c;
        c.x = this->x + b.x;
        c.y = this->y + b.y;
        c.z = this->z + b.z;
        return c;
    }

    Vector3f operator-(const Vector3f& b)
    {
        Vector3f c;
        c.x = this->x - b.x;
        c.y = this->y - b.y;
        c.z = this->z - b.z;
        return c;
    }

    Vector3f operator*(float b)
    {
        Vector3f c;
        c.x = this->x * b;
        c.y = this->y * b;
        c.z = this->z * b;
        return c;
    }

    Vector3f operator*(const Vector3f& b)
    {
        Vector3f c;
        c.x = this->x * b.x;
        c.y = this->y * b.y;
        c.z = this->z * b.z;
        return c;
    }

    Vector3f operator=(const Vector3f& b)
    {
        this->x = b.x;
        this->y = b.y;
        this->z = b.z;
        return *this;
    }

    Vector3f operator+=(const Vector3f& b)
    {
        this->x += b.x;
        this->y += b.y;
        this->z += b.z;
        return *this;
    }

    Vector3f operator-=(const Vector3f& b)
    {
        this->x -= b.x;
        this->y -= b.y;
        this->z -= b.z;
        return *this;
    }

    Vector3f operator/=(const float& b)
    {
        this->x /= b;
        this->y /= b;
        this->z /= b;
        return *this;
    }

    float operator[](const int a)
    {
        if(a == 0) return this->x;
        if(a == 1) return this->y;
        return this->z;
    }
};

timeval t_start, t_end;  // timers

/* 傅里叶变换 */
void ft(int width, int height, Vector3f** fxRealTwo, Vector3f** fxImagTwo, Vector3f** RealTwo)
{
    // 结果数据初始化
    for (int v = 0; v < width; v++)
    {
        for (int u = 0; u < height; u++)
        {
            fxRealTwo[v][u] = {0, 0, 0};
            fxImagTwo[v][u] = {0, 0, 0};
        }
    }

    // 使用4层循环进行运算
	for (int v = 0; v < width; v++)
    {
        for (int u = 0; u < height; u++)
        {
            for (int j = 0; j < width; j++)
            {
                for (int i = 0; i < height; i++)
                {
                    fxRealTwo[v][u] += RealTwo[j][i] * cos(2 * MY_PI * u * i / height + 2 * MY_PI * v * j / width);
                    fxImagTwo[v][u] -= RealTwo[j][i] * sin(2 * MY_PI * u * i / height + 2 * MY_PI * v * j / width);
                }
            }
        }
    }
}

void ift(int width, int height, Vector3f** ResultReal, Vector3f** ResultImage, Vector3f** ifxRealTwo, Vector3f** ifxImageTwo)
{
    for (int v = 0; v < width; v++)
    {
        for (int u = 0; u < height; u++)
        {
            ifxRealTwo[v][u] = {0, 0, 0};
            ifxImageTwo[v][u] = {0, 0, 0};
        }
    }

    for (int v = 0; v < width; v++)
    {
        for (int u = 0; u < height; u++)
        {
            for (int j =0; j < width; j++)
            {
                for (int i = 0; i < height; i++)
                {
                    float w = 2 * MY_PI * u * i / height + 2 * MY_PI * v * j / width;
                    ifxRealTwo[v][u] += ResultReal[j][i] * cos(w) - ResultImage[j][i] * sin(w);
                    ifxImageTwo[v][u] += ResultImage[j][i] * cos(w) + ResultReal[j][i] * sin(w);
                }
            }
            ifxRealTwo[v][u] /= (width * height);
            ifxImageTwo[v][u] /= (width * height);
        }
    }
}

/* 卷积 */
void ord_convolution(Vector3f **ResultReal, Vector3f **original_color, Vector3f **FilterReal, Vector3f **FilterImage, int len_x, int len_y, int argc, char *argv[])
{
    // 初始化临时变量
    Vector3f** tempColor = new Vector3f*[filter_box_size];
    for(int i = 0; i < filter_box_size; i++) tempColor[i] = new Vector3f[filter_box_size];

    Vector3f** tempReal = new Vector3f*[filter_box_size];  // 实部
    for(int i = 0; i < filter_box_size; i++) tempReal[i] = new Vector3f[filter_box_size];

    Vector3f** tempImage = new Vector3f*[filter_box_size];  // 虚部
    for(int i = 0; i < filter_box_size; i++) tempImage[i] = new Vector3f[filter_box_size];

    Vector3f** tempResultReal = new Vector3f*[filter_box_size];  // 结果实部
    for(int i = 0; i < filter_box_size; i++) tempResultReal[i] = new Vector3f[filter_box_size];

    Vector3f** tempResultImage = new Vector3f*[filter_box_size];  // 结果虚部
    for(int i = 0; i < filter_box_size; i++) tempResultImage[i] = new Vector3f[filter_box_size];

    // 循环对每个区域都计算卷积
    // 前两个循环是遍历原图中的每个box。(i, j)是当前box的中心位置。
    for(int i = filter_box_size / 2; i < len_x - filter_box_size / 2; i++)
    {
        for(int j = filter_box_size / 2; j < len_y - filter_box_size / 2; j++)
        {
            // 获得当前区域的原始颜色
            for(int x = 0; x < filter_box_size; x++)
                for(int y = 0; y < filter_box_size; y++)
                    tempColor[x][y] = original_color[i - filter_box_size / 2 + x][j - filter_box_size / 2 + y];
            // 对原始区域进行卷积
            // 进行傅里叶变换
            ft(filter_box_size, filter_box_size, tempReal, tempImage, tempColor);
            // 循环进行乘积的计算
            int p = filter_box_size / 2;
            // 遍历卷积盒做乘积
            for(int x = 0; x < filter_box_size; x++)
            {
                if(i + x - filter_box_size/2 < 0 || i + x - filter_box_size/2 >= len_x) continue;
                for(int y = 0; y < filter_box_size; y++)
                {
                    if(j + y - filter_box_size/2 < 0 || j + y - filter_box_size/2 >= len_y) continue;
                    tempResultReal[x][y] = tempReal[x][y] * FilterReal[x][y] - tempImage[x][y] * FilterImage[x][y];
                    tempResultImage[x][y] = tempReal[x][y] * FilterImage[x][y] + FilterReal[x][y] * tempImage[x][y];
                }
            }
            // 逆傅里叶变换
            ift(filter_box_size, filter_box_size, tempResultReal, tempResultImage, tempReal, tempImage);
            // 保存运算结果
            ResultReal[i][j] = tempReal[p][p];
        }
    }

    // 收回分配出去的空间
    for(int i = 0; i < filter_box_size; i++) delete[] tempColor[i];
    delete[] tempColor;

    for(int i = 0; i < filter_box_size; i++) delete[] tempReal[i];
    delete[] tempReal;

    for(int i = 0; i < filter_box_size; i++) delete[] tempImage[i];
    delete[] tempImage;

    for(int i = 0; i < filter_box_size; i++) delete[] tempResultReal[i];
    delete[] tempResultReal;

    for(int i = 0; i < filter_box_size; i++) delete[] tempResultImage[i];
    delete[] tempResultImage;
}

int main(int argc, char *argv[])
{
    /* 初始化卷积盒并进行ft */
    Vector3f** filter = new Vector3f*[filter_box_size];
    for(int i = 0; i < filter_box_size; i++) filter[i] = new Vector3f[filter_box_size];
    
    // 卷积盒赋值
    for(int i = 0; i < filter_box_size; i++)
        for(int j = 0; j < filter_box_size; j++)
            filter[i][j] = {(float)1/(filter_box_size * filter_box_size), (float)1/(filter_box_size * filter_box_size), (float)1/(filter_box_size * filter_box_size)};

    // 结果数组
    Vector3f** FilterReal = new Vector3f*[filter_box_size];
    for(int i = 0; i < filter_box_size; i++) FilterReal[i] = new Vector3f[filter_box_size];

    Vector3f** FilterImage = new Vector3f*[filter_box_size];
    for(int i = 0; i < filter_box_size; i++) FilterImage[i] = new Vector3f[filter_box_size];

    // 傅里叶变换
    ft(filter_box_size, filter_box_size, FilterReal, FilterImage, filter);

    /* 原始数据数组初始化 */
    Vector3f **original_color = new Vector3f*[n];
    for(int i = 0; i < n; i++) original_color[i] = new Vector3f[n];

    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            if(i + j % 2 == 0) original_color[i][j] = {255, 255, 255};
            else original_color[i][j] = {0, 0, 0};

    // 结果数组初始化
    Vector3f **ResultReal = new Vector3f*[n];
    for(int i = 0; i < n; i++) ResultReal[i] = new Vector3f[n];

    // 输出参数
    cout << "n: " << n << endl;
    cout << "loop time: " << loop_time << endl;
    cout << "filter box size: " << filter_box_size << endl;

    /* 进行卷积 */
    gettimeofday(&t_start, NULL);
    for(int i = 0; i < loop_time; i++)
        ord_convolution(ResultReal, original_color, FilterReal, FilterImage, n, n, argc, argv);
    gettimeofday(&t_end, NULL);
    double delta_t = ((t_end.tv_sec - t_start.tv_sec) * 1000 + (t_end.tv_usec - t_start.tv_usec) / 1000.0) / loop_time;
    cout << delta_t << "ms" << endl;

    cout << "finish" << endl;

    // 收回分配出去的空间
    for(int i = 0; i < filter_box_size; i++) delete[] filter[i];
    delete[] filter;

    for(int i = 0; i < filter_box_size; i++) delete[] FilterReal[i];
    delete[] FilterReal;

    for(int i = 0; i < filter_box_size; i++) delete[] FilterImage[i];
    delete[] FilterImage;

    for(int i = 0; i < filter_box_size; i++) delete[] original_color[i];
    delete[] original_color;

    for(int i = 0; i < filter_box_size; i++) delete[] ResultReal[i];
    delete[] ResultReal;

    return 0;
}