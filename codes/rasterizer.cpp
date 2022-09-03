// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <iostream>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <sys/time.h>

using namespace std;

#define MY_PI 3.1415926
#define filter_box_size 3

timeval t_start, t_end;  // timers

extern double sum_time;

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();  // 获得下一个序号
    pos_buf.emplace(id, positions);  // 在map的序号位置插入位置
	// 返回序号
    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();  // 获得下一个序号
    ind_buf.emplace(id, indices);  // 在map的序号位置插入位置
	// 返回序号
    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();  // 获得下一个序号
    col_buf.emplace(id, cols);  // 在map的序号位置插入位置
	// 返回序号
    return {id};
}

// 变换为奇次坐标
auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(float x, float y, const Vector3f* _v)
{   
    // 检查点(x, y)是否在用_v[0], _v[1], _v[2]表示的三角形中
	
	Eigen::Vector2f p;
	p << x, y;

	Eigen::Vector2f AB = _v[1].head(2) - _v[0].head(2);
	Eigen::Vector2f BC = _v[2].head(2) - _v[1].head(2);
	Eigen::Vector2f CA = _v[0].head(2) - _v[2].head(2);

	Eigen::Vector2f AP = p - _v[0].head(2);
	Eigen::Vector2f BP = p - _v[1].head(2);
	Eigen::Vector2f CP = p - _v[2].head(2);
	
	// 判断每个z坐标是否统一
	return AB[0] * AP[1] - AB[1] * AP[0] > 0 
		&& BC[0] * BP[1] - BC[1] * BP[0] > 0
		&& CA[0] * CP[1] - CA[1] * CP[0] > 0;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

// 绘画。这个函数负责将所有三角形画出来，我们的重点在于最后一步的光栅化，其他部分不必过于在意。
void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
	// 类型都是std::vector<Eigen::Vector3f>
    auto& buf = pos_buf[pos_buffer.pos_id];  // 返回位置信息
    auto& ind = ind_buf[ind_buffer.ind_id];  // 返回索引信息
    auto& col = col_buf[col_buffer.col_id];  // 返回颜色信息

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;  // mvp的变换矩阵
	// 遍历所有的索引（遍历所有的三角形）
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        gettimeofday(&t_start, NULL);
        rasterize_triangle(t);
        gettimeofday(&t_end, NULL);
        double delta_t = (t_end.tv_sec - t_start.tv_sec) * 1000 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;
        sum_time += delta_t;
    }
}

void fft(int width, int height, Vector3f** fxRealTwo, Vector3f** fxImagTwo, Vector3f** RealTwo)
{
    for (int v = 0; v < width; v++)
    {
        for (int u = 0; u < height; u++)
        {
            fxRealTwo[v][u] = {0, 0, 0};
            fxImagTwo[v][u] = {0, 0, 0};
        }
    }

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

void ifft(int width, int height, Vector3f** ResultReal, Vector3f** ResultImage, Vector3f** ifxRealTwo, Vector3f** ifxImageTwo)
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
                    ifxImageTwo[v][u] += -ResultImage[j][i] * cos(w) + ResultReal[j][i] * sin(w);
                }
            }
            ifxRealTwo[v][u] /= (width * height);
            ifxImageTwo[v][u] /= (width * height);
        }
    }
}

void ord_convolution(Vector3f **ResultReal, Vector3f **original_color, Vector3f **FilterReal, Vector3f **FilterImage, int len_x, int len_y)
{
    // 初始化临时变量
    Vector3f** tempColor = new Vector3f*[filter_box_size];
    for(int i = 0; i < filter_box_size; i++) tempColor[i] = new Vector3f[filter_box_size];

    Vector3f** tempReal = new Vector3f*[filter_box_size];
    for(int i = 0; i < filter_box_size; i++) tempReal[i] = new Vector3f[filter_box_size];

    Vector3f** tempImage = new Vector3f*[filter_box_size];
    for(int i = 0; i < filter_box_size; i++) tempImage[i] = new Vector3f[filter_box_size];

    Vector3f** tempResultReal = new Vector3f*[filter_box_size];
    for(int i = 0; i < filter_box_size; i++) tempResultReal[i] = new Vector3f[filter_box_size];

    Vector3f** tempResultImage = new Vector3f*[filter_box_size];
    for(int i = 0; i < filter_box_size; i++) tempResultImage[i] = new Vector3f[filter_box_size];

    for(int i = filter_box_size / 2; i < len_x - filter_box_size / 2; i++)
    {
        for(int j = filter_box_size / 2; j < len_y - filter_box_size / 2; j++)
        {
            // 获得当前区域的原始颜色
            for(int x = 0; x < filter_box_size; x++)
                for(int y = 0; y < filter_box_size; y++)
                    tempColor[x][y] = original_color[i - filter_box_size / 2 + x][j - filter_box_size / 2 + y];
            // 对原始区域进行卷积
            // 傅里叶变换
            fft(filter_box_size, filter_box_size, tempReal, tempImage, tempColor);
            // 获得乘积
            int p = filter_box_size / 2;
            for(int x = 0; x < filter_box_size; x++)
            {
                if(i + x - filter_box_size/2 < 0 || i + x - filter_box_size/2 >= len_x) continue;
                for(int y = 0; y < filter_box_size; y++)
                {
                    if(j + y - filter_box_size/2 < 0 || j + y - filter_box_size/2 >= len_y) continue;
                    for(int k = 0; k < 3; k++)
                    {
                        tempResultReal[x][y][k] = tempReal[x][y][k] * FilterReal[x][y][k] - tempImage[x][y][k] * FilterImage[x][y][k];
                        tempResultImage[x][y][k] = tempReal[x][y][k] * FilterImage[x][y][k] + FilterReal[x][y][k] * tempImage[x][y][k];
                    }
                }
            }
            // 逆傅里叶变换
            ifft(filter_box_size, filter_box_size, tempResultReal, tempResultImage, tempReal, tempImage);
            // 保存结果
            ResultReal[i][j] = tempReal[p][p];
        }
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    bool aliasing = 1;

    auto v = t.toVector4();
    
	/* 算法流程
	// 确定当前三角形的bounding box
	// 遍历bounding box中的所有像素，并确定当前像素是否在三角形内

	// 如果当前像素在三角形内，则使用下面的代码来获得插值的z值
    //auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
    //float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    //float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    //z_interpolated *= w_reciprocal;

	// 如果当前像素有颜色，就使用getColor()函数和set_pixel()函数给像素上色
	*/

    // 不使用反走样算法
    if(!aliasing)
    {
        // 2维的bounding box
        float min_x = std::min(v[0][0], std::min(v[1][0], v[2][0]));  // 最小的x就是3个顶点中最x最小的。下面也是类似。
        float max_x = std::max(v[0][0], std::max(v[1][0], v[2][0]));
        float min_y = std::min(v[0][1], std::min(v[1][1], v[2][1]));
        float max_y = std::max(v[0][1], std::max(v[1][1], v[2][1]));

        min_x = (int)std::floor(min_x);  // 向下取整
        max_x = (int)std::ceil(max_x);  // 向上取整
        min_y = (int)std::floor(min_y);
        max_y = (int)std::ceil(max_y);

        // 遍历bounding box中所有像素
        for (int x = min_x; x <= max_x; x++) 
        {
            for (int y = min_y; y <= max_y; y++) 
            {
                // 如果当前像素在三角形中
                if (insideTriangle((float)x + 0.5, (float)y + 0.5, t.v)) 
                {
                    // 计算当前像素的深度
                    auto[alpha, beta, gamma] = computeBarycentric2D((float)x + 0.5, (float)y + 0.5, t.v);
                    float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;
                    // 如果深度小于当前深度，则显示这个像素
                    if (depth_buf[get_index(x, y)] > z_interpolated) 
                    {
                        Vector3f color = t.getColor();
                        Vector3f point(3);
                        point << (float)x, (float)y, z_interpolated;
                        depth_buf[get_index(x, y)] = z_interpolated;
                        set_pixel(point, color);
                    }
                }
            }
        }
    }

    // 使用低通滤波
    else
    {
        float min_x = std::min(v[0][0], std::min(v[1][0], v[2][0])) - 1;
        float max_x = std::max(v[0][0], std::max(v[1][0], v[2][0])) + 1;
        float min_y = std::min(v[0][1], std::min(v[1][1], v[2][1])) - 1;
        float max_y = std::max(v[0][1], std::max(v[1][1], v[2][1])) + 1;

        min_x = (int)std::floor(min_x);  // 向下取整
        max_x = (int)std::ceil(max_x);  // 向上取整
        min_y = (int)std::floor(min_y);
        max_y = (int)std::ceil(max_y);

        int len_x = max_x - min_x + 1;
        int len_y = max_y - min_y + 1;

        // 初始化bounding box
        Vector3f** original_color = new Vector3f*[len_x];
        for(int i = 0; i < len_x; i++) original_color[i] = new Vector3f[len_y];

        int a = max_x - min_x - 1, b = max_y - min_y - 1;
        for(int i = 0; i < a; i++)
        {
            int x = i + min_x + 1;
            for (int j = 0; j < b; j++) 
            {
                int y = j + min_y + 1;
                if (insideTriangle((float)x + 0.5, (float)y + 0.5, t.v)) original_color[int(x - min_x)][int(y- min_y)] = t.getColor();
                else original_color[int(x - min_x)][int(y- min_y)] = {0, 0, 0};
            }
        }

        // 初始化卷积盒并进行fft
        Vector3f** filter = new Vector3f*[filter_box_size];
        for(int i = 0; i < filter_box_size; i++) filter[i] = new Vector3f[filter_box_size];

        for(int i = 0; i < filter_box_size; i++)
            for(int j = 0; j < filter_box_size; j++)
                filter[i][j] = {(float)1/(filter_box_size * filter_box_size), (float)1/(filter_box_size * filter_box_size), (float)1/(filter_box_size * filter_box_size)};
        
        Vector3f** FilterReal = new Vector3f*[filter_box_size];
        for(int i = 0; i < filter_box_size; i++) FilterReal[i] = new Vector3f[filter_box_size];

        Vector3f** FilterImage = new Vector3f*[filter_box_size];
        for(int i = 0; i < filter_box_size; i++) FilterImage[i] = new Vector3f[filter_box_size];

        fft(filter_box_size, filter_box_size, FilterReal, FilterImage, filter);

        // 初始化临时变量
        Vector3f** tempColor = new Vector3f*[filter_box_size];
        for(int i = 0; i < filter_box_size; i++) tempColor[i] = new Vector3f[filter_box_size];

        Vector3f** tempReal = new Vector3f*[filter_box_size];
        for(int i = 0; i < filter_box_size; i++) tempReal[i] = new Vector3f[filter_box_size];

        Vector3f** tempImage = new Vector3f*[filter_box_size];
        for(int i = 0; i < filter_box_size; i++) tempImage[i] = new Vector3f[filter_box_size];

        Vector3f** tempResultReal = new Vector3f*[filter_box_size];
        for(int i = 0; i < filter_box_size; i++) tempResultReal[i] = new Vector3f[filter_box_size];

        Vector3f** tempResultImage = new Vector3f*[filter_box_size];
        for(int i = 0; i < filter_box_size; i++) tempResultImage[i] = new Vector3f[filter_box_size];

        // 结果数组初始化
        Vector3f** ResultReal = new Vector3f*[len_x];
        for(int i = 0; i < len_x; i++) ResultReal[i] = new Vector3f[len_y];

        Vector3f** ResultImage = new Vector3f*[len_x];
        for(int i = 0; i < len_x; i++) ResultImage[i] = new Vector3f[len_y];

        for (int v = 0; v < len_x; v++)
        {
            for (int u = 0; u < len_y; u++)
            {
                ResultReal[v][u] = {0, 0, 0};
                ResultImage[v][u] = {0, 0, 0};
            }
        }

        // 按位置进行卷积
        ord_convolution(ResultReal, original_color, FilterReal, FilterImage, len_x, len_y);

        // 遍历bounding box，根据每个像素的深度确认是否应该出现在图像上
        a = max_x - min_x + 1, b = max_y - min_y + 1;
        for(int i = 0; i < a; i++)
        {
            int x = min_x + i;
            for (int j = 0; j < b; j++) 
            {
                int y = min_y + j;
                if(!insideTriangle(x, y, t.v) && !insideTriangle(x + 1, y + 1, t.v) && !insideTriangle(x + 1, y, t.v) && !insideTriangle(x, y + 1, t.v)) continue;
                
                // 计算当前像素的深度
                auto[alpha, beta, gamma] = computeBarycentric2D((float)x + 0.5, (float)y + 0.5, t.v);
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;

                // 如果深度小于当前深度，则显示这个像素
                if (depth_buf[get_index(x, y)] > z_interpolated) 
                {
                    Vector3f color = ResultReal[int(x - min_x)][int(y- min_y)];
                    // for(int try_print = 0; try_print < 3; try_print++)
                    //     cout << ResultReal[int(x - min_x)][int(y- min_y)][try_print] << " ";
                    // cout << endl;
                    Vector3f point(3);
                    point << (float)x, (float)y, z_interpolated;
                    depth_buf[get_index(x, y)] = z_interpolated;
                    set_pixel(point, color);
                }
            }
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
	// 如果buff中包含了颜色，那就将颜色清零
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
	// 如果buff中包含了深度，那就将深度清零（其实是变成无穷远）
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

// 构造函数，设置宽度和高度，设置图像缓冲区以及深度缓冲区两个大小
rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on