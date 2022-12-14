// clang-format off
#include <iostream>
#include <opencv2/opencv.hpp>
#include "rasterizer.hpp"
#include "global.hpp"
#include "Triangle.hpp"

using namespace std;

constexpr double MY_PI = 3.1415926;

double sum_time = 0;

// 获得视角矩阵。参数为眼睛当前的位置。将眼睛移到原点的位置。
Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();  // 矩阵初始化。4*4矩阵。

    Eigen::Matrix4f translate;  // 转换矩阵
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

// 将模型矩阵初始化为单位矩阵
Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    // 将模型初始化为单位矩阵
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    return model;
}

// 使用给定的参数逐个元素地构建透视投影矩阵并返回该矩阵
Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();  // 初始化投影矩阵
    Eigen::Matrix4f P2O = Eigen::Matrix4f::Identity();  // 将透视投影转换为正交投影的矩阵
    // 进行透视投影转化为正交投影的矩阵
    P2O << zNear, 0, 0, 0,
            0, zNear, 0, 0,
            0, 0, zNear + zFar, -zFar * zNear,
            0, 0, 1, 0;
    
    float halfEyeAngelRadian = eye_fov / 2.0 / 180.0 * MY_PI;
    float t = zNear * std::tan(halfEyeAngelRadian);  // top y轴的最高点
    float r = t * aspect_ratio;  // right x轴的最大值
    float l = -r;  // left x轴最小值
    float b = -t;  // bottom y轴的最大值

    Eigen::Matrix4f ortho1 = Eigen::Matrix4f::Identity();
    // 进行一定的缩放使之成为一个标准的长度为2的正方体
    ortho1 << 2 / (r - l), 0, 0, 0,
            0, 2 / (t - b), 0, 0,
            0, 0, 2 / (zNear - zFar), 0,
            0, 0, 0, 1;
    
    Eigen::Matrix4f ortho2 = Eigen::Matrix4f::Identity();
    // 把一个长方体的中心移动到原点
    ortho2 << 1, 0, 0, -(r + l) / 2,
            0, 1, 0, -(t + b) / 2,
            0, 0, 1, -(zNear + zFar) / 2,
            0, 0, 0, 1;
    
    Eigen::Matrix4f Matrix_ortho = ortho1 * ortho2;
    projection = Matrix_ortho * P2O;

    return projection;
}

int main(int argc, char *argv[])
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    // 创建一个rasterizer类对象
    rst::rasterizer r(700, 700);
    // 创建一个长度为3的向量，元素类型为float。眼睛的位置为{0, 0, 5}
    Eigen::Vector3f eye_pos = {0,0,5};

    // 创建一个vector容器，包含的元素为维度为3的向量。两个三角形顶点的位置。
    std::vector<Eigen::Vector3f> pos
            {
                    {2, 0, -2},
                    {0, 2, -2},
                    {-2, 0, -2},
                    {3.5, -1, -5},
                    {2.5, 1.5, -5},
                    {-1, 0.5, -5}
            };
    // 创建一个vector容器，包含的元素为维度为3的向量。索引。
    std::vector<Eigen::Vector3i> ind
            {
                    {0, 1, 2},
                    {3, 4, 5}
            };
    // 创建一个vector容器，包含的元素为维度为3的向量。每个顶点的颜色。
    std::vector<Eigen::Vector3f> cols
            {
                    {217.0, 238.0, 185.0},
                    {217.0, 238.0, 185.0},
                    {217.0, 238.0, 185.0},
                    {185.0, 217.0, 238.0},
                    {185.0, 217.0, 238.0},
                    {185.0, 217.0, 238.0}
            };

    // 在对应位置插入数据
    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);
    auto col_id = r.load_colors(cols);

    int key = 0;
    int frame_count = 0;

    while(1)
    {
        // 将颜色和深度都清零
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        // 设置模型、视角以及投影
        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if(frame_count == 50)
        {
            cout << "time: " << sum_time / 50 << endl;
            break;
        }
    }

    return 0;
}
// clang-format on