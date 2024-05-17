#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include </usr/include/c++/7/bits/stl_numeric.h>
#include <unordered_map>
#include "paddle_api.h"
#include <sys/time.h>
#include <time.h>
#include <cmath>
#include <string>
#include <dirent.h>
#include <mqtt/async_client.h>
#include <chrono>
#include <ctime>

using namespace paddle::lite_api;

int IMAGE = 0;
int VIDEO = 1;
int CAM = 2;
bool FALL_FLAGE = 0;
bool LAST_FLAGE = 0;
bool FALL_DETECTED = 0;


// 定义类别到类名的映射
std::unordered_map<int, std::string> classes = {
    {0, "Fall"}
};

// 定义置信度阈值和IoU阈值
float confidence_thres = 0.50;
float iou_thres = 0.5;
const std::vector<int64_t> INPUT_SHAPE = {1, 3, 640, 640};

// 生成随机颜色
cv::RNG rng;
std::vector<cv::Scalar> color_palette;
void create_color_palette() {
    for (int i = 0; i < 10; i++)
    {
        int r = rng.uniform(0, 256); // 生成0到255的随机数作为红色通道值
        int g = rng.uniform(0, 256); // 生成0到255的随机数作为绿色通道值
        int b = rng.uniform(0, 256); // 生成0到255的随机数作为蓝色通道值

        cv::Scalar color(b, g, r); // 创建颜色值（BGR顺序）

        color_palette.push_back(color); // 将颜色值添加到颜色向量中
    }
}

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

//mqtt服务配置
const std::string SERVER_ADDRESS("tcp://8.134.150.174:1883");
const std::string CLIENT_ID("falldetect_publisher");

class callback : public mqtt::callback {
  void connection_lost(const std::string& cause) override {
    std::cout << "Connection lost: " << cause << std::endl;
  }

  void delivery_complete(mqtt::delivery_token_ptr delivery_token) override {
    std::cout << "Message delivery complete" << std::endl;
  }
};

/**
 * 计算给定边界框与一组其他边界框之间的交并比（IoU）。
 *
 * @param box 单个边界框，格式为 [x1, y1, width, height]。
 * @param other_boxes 其他边界框的数组，每个边界框的格式也为 [x1, y1, width, height]。
 * @return 一个数组，包含给定边界框与每个其他边界框的IoU值。
 */
std::vector<float> calculate_iou(std::vector<float> box, 
                                 std::vector<std::vector<float>> other_boxes) {
    std::vector<float> iou;

    float box_x1 = box[0];
    float box_y1 = box[1];
    float box_width = box[2];
    float box_height = box[3];
    float box_x2 = box_x1 + box_width;
    float box_y2 = box_y1 + box_height;
    float box_area = box_width * box_height;

    for (const auto& other_box : other_boxes) {
        float other_box_x1 = other_box[0];
        float other_box_y1 = other_box[1];
        float other_box_width = other_box[2];
        float other_box_height = other_box[3];
        float other_box_x2 = other_box_x1 + other_box_width;
        float other_box_y2 = other_box_y1 + other_box_height;
        float other_box_area = other_box_width * other_box_height;

        float intersection_x1 = std::max(box_x1, other_box_x1);
        float intersection_y1 = std::max(box_y1, other_box_y1);
        float intersection_x2 = std::min(box_x2, other_box_x2);
        float intersection_y2 = std::min(box_y2, other_box_y2);

        float intersection_area = std::max(0.0f, intersection_x2 - intersection_x1) * std::max(0.0f, intersection_y2 - intersection_y1);

        float iou_value = intersection_area / (box_area + other_box_area - intersection_area);
        iou.push_back(iou_value);
    }

    return iou;
}

/**
 * 使用自定义的非最大抑制（NMS）算法选择具有高置信度且不重叠的边界框。
 *
 * @param boxes 边界框的数组，每个边界框的格式为 [x1, y1, width, height]。
 * @param scores 边界框的置信度得分数组。
 * @param confidence_threshold 置信度阈值，低于该阈值的边界框将被过滤。
 * @param iou_threshold IoU阈值，用于确定边界框是否重叠。
 * @return 一个数组，包含选择的边界框的索引。
 */
std::vector<int> custom_NMSBoxes(std::vector<std::vector<float>> boxes, 
                                 std::vector<float> scores, 
                                 float confidence_threshold, 
                                 float iou_threshold) {
    std::vector<int> indices;

    // 如果没有边界框，则直接返回空列表
    if (boxes.empty()) {
        return indices;
    }

    // 根据置信度阈值过滤边界框
    std::vector<float> filtered_scores;
    std::vector<std::vector<float>> filtered_boxes;

    for (size_t i = 0; i < scores.size(); i++) {
        if (scores[i] > confidence_threshold) {
            filtered_scores.push_back(scores[i]);
            filtered_boxes.push_back(boxes[i]);
        }
    }

    // 如果过滤后没有边界框，则返回空列表
    if (filtered_boxes.empty()) {
        return indices;
    }

    // 根据置信度得分对边界框进行排序
    std::vector<int> sorted_indices(filtered_scores.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) {
        return filtered_scores[a] > filtered_scores[b];
    });

    // 当还有未处理的边界框时，循环继续
    while (!sorted_indices.empty()) {
        // 选择得分最高的边界框索引
        int current_index = sorted_indices[0];
        indices.push_back(current_index);

        // 如果只剩一个边界框，则结束循环
        if (sorted_indices.size() == 1) {
            break;
        }

        // 获取当前边界框和其他边界框
        std::vector<float> current_box = filtered_boxes[current_index];
        std::vector<std::vector<float>> other_boxes;

        for (size_t i = 1; i < sorted_indices.size(); i++) {
            other_boxes.push_back(filtered_boxes[sorted_indices[i]]);
        }

        // 计算当前边界框与其他边界框的IoU
        std::vector<float> iou = calculate_iou(current_box, other_boxes);

        // 找到IoU低于阈值的边界框，即与当前边界框不重叠的边界框
        std::vector<int> non_overlapping_indices;

        for (size_t i = 0; i < iou.size(); i++) {
            if (iou[i] <= iou_threshold) {
                non_overlapping_indices.push_back(i);
            }
        }

        // 更新sorted_indices以仅包含不重叠的边界框
        std::vector<int> new_sorted_indices(non_overlapping_indices.size());

        for (size_t i = 0; i < non_overlapping_indices.size(); i++) {
            new_sorted_indices[i] = sorted_indices[non_overlapping_indices[i] + 1];
        }

        sorted_indices = new_sorted_indices;
    }

    // 返回选择的边界框索引
    return indices;
}

/**
 * 在输出图像上绘制检测结果的边界框和标签文本。
 * 
 * @param output_image 输出图像，将在其上绘制边界框和标签文本
 * @param box 边界框的坐标 [x1, y1, w, h]
 * @param score 检测结果的得分
 */
void draw_detections(cv::Mat &output_image, std::vector<float> box, float score) {
    // 提取边界框的坐标
    float x1 = box[0];
    float y1 = box[1];
    float w = box[2];
    float h = box[3];

    // 根据类别ID检索颜色
    cv::Scalar color = color_palette[0];

    // 在图像上绘制边界框
    cv::rectangle(output_image, cv::Point(x1, y1), cv::Point(x1 + w, y1 + h), color, 2);

    // 创建标签文本，包括类名和得分
    std::string label = classes[0] + ": " + std::to_string(score);

    // 计算标签文本的尺寸
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);

    // 计算标签文本的位置
    int label_x = x1;
    int label_y = y1 - 10 > label_size.height ? y1 - 10 : y1 + 10;

    // 绘制填充的矩形作为标签文本的背景
    cv::rectangle(output_image, cv::Point(label_x, label_y - label_size.height), cv::Point(label_x + label_size.width, label_y + label_size.height), color, cv::FILLED);

    // 在图像上绘制标签文本
    cv::putText(output_image, label, cv::Point(label_x, label_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

/**
 * 将 NHWC3 格式的浮点数图像数据转换为 NC3HW 格式，并进行均值归一化和标准化处理。
 * 
 * @param src 源图像数据指针，NHWC3 格式，按 [channel, height, width] 排列
 * @param dst 目标图像数据指针，NC3HW 格式，按 [channel, height, width] 排列
 * @param mean 均值数组指针，包含三个通道的均值值，默认为 nullptr
 * @param std 标准差数组指针，包含三个通道的标准差值，默认为 nullptr
 * @param width 图像宽度
 * @param height 图像高度
 */
void NHWC3ToNC3HW(const float *src, float *dst, const float *mean,
                  const float *std, int width, int height) {
  int size = height * width;
  float32x4_t vmean0 = vdupq_n_f32(mean ? mean[0] : 0.0f);
  float32x4_t vmean1 = vdupq_n_f32(mean ? mean[1] : 0.0f);
  float32x4_t vmean2 = vdupq_n_f32(mean ? mean[2] : 0.0f);
  float scale0 = std ? (1.0f / std[0]) : 1.0f;
  float scale1 = std ? (1.0f / std[1]) : 1.0f;
  float scale2 = std ? (1.0f / std[2]) : 1.0f;
  float32x4_t vscale0 = vdupq_n_f32(scale0);
  float32x4_t vscale1 = vdupq_n_f32(scale1);
  float32x4_t vscale2 = vdupq_n_f32(scale2);
  float *dst_c0 = dst;
  float *dst_c1 = dst + size;
  float *dst_c2 = dst + size * 2;
  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(src);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dst_c0, vs0);
    vst1q_f32(dst_c1, vs1);
    vst1q_f32(dst_c2, vs2);
    src += 12;
    dst_c0 += 4;
    dst_c1 += 4;
    dst_c2 += 4;
  }
  for (; i < size; i++) {
    *(dst_c0++) = (*(src++) - mean[0]) * scale0;
    *(dst_c1++) = (*(src++) - mean[1]) * scale1;
    *(dst_c2++) = (*(src++) - mean[2]) * scale2;
  }
}

/**
 * 调整图像大小并进行缩放操作。
 * 
 * @param image 输入图像
 * @param size 目标图像尺寸
 * @param letterboxImage 是否进行letterbox缩放
 * @return 调整大小后的图像
 */
cv::Mat resizeImage(const cv::Mat& image, 
                    const cv::Size& size, 
                    bool letterboxImage)
{
    int ih = image.rows;
    int iw = image.cols;
    int h = size.height;
    int w = size.width;

    cv::Mat resizedImage;
    if (letterboxImage) {
        double scale = std::min(static_cast<double>(w) / iw, static_cast<double>(h) / ih);
        int nw = static_cast<int>(iw * scale);
        int nh = static_cast<int>(ih * scale);
        cv::resize(image, resizedImage, cv::Size(nw, nh), cv::INTER_LINEAR);

        cv::Mat imageBack(h, w, CV_8UC3, cv::Scalar(128, 128, 128));
        cv::Rect roi((w - nw) / 2, (h - nh) / 2, nw, nh);
        resizedImage.copyTo(imageBack(roi));
        return imageBack;
    } else {
        cv::resize(image, resizedImage, size, cv::INTER_LINEAR);
        return resizedImage;
    }
}


/**
 * 对图像进行预处理，包括颜色空间转换、调整大小、归一化等操作。
 * 
 * @param img 输入图像
 * @param input_width 输入图像的目标宽度
 * @param input_height 输入图像的目标高度
 * @return 预处理后的图像数据
 */
cv::Mat preprocess(const cv::Mat img, 
                   int input_width, 
                   int input_height) {

    // 将图像颜色空间从BGR转换为RGB
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);

    // 使用letterbox将图像大小调整为匹配输入形状
    cv::Mat resized_img;
    resized_img = resizeImage(rgb_img,cv::Size(input_width,input_height),true);

    // 通过除以255.0来归一化图像数据
    cv::Mat normalized_img;
    resized_img.convertTo(normalized_img, CV_32F, 1.0 / 255.0);

    // 返回预处理后的图像数据
    return normalized_img;
}

/**
 * 后处理函数，用于解析模型输出并生成检测结果。
 * 
 * @param input_image 输入图像
 * @param predictor Paddle预测器
 * @param input_width 输入图像的宽度
 * @param input_height 输入图像的高度
 * @return 包含检测边界框和得分的向量对
 */
std::pair<std::vector<std::vector<float>>, std::vector<float>> postprocess(const cv::Mat input_image, 
                                                                           std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor, 
                                                                           int input_width, 
                                                                           int input_height) {
    int img_height = input_image.rows;
    int img_width = input_image.cols;
    auto outputTensor = predictor->GetOutput(0);
    auto outputData = outputTensor->data<float>();
    auto outputShape = outputTensor->shape();
    auto outputRow = outputShape[2];
    // 用于存储检测的边界框、得分和类别ID的向量
    std::vector<std::vector<float>> boxes;
    std::vector<float> scores;
    // 计算边界框坐标的缩放因子
    float x_factor = static_cast<float>(img_width) / input_width;
    float y_factor = static_cast<float>(img_height) / input_height;
    // 遍历输出数组的每一行
    for (int i = 0; i < outputRow; i += 1) {
            
        // 提取当前行得分
        auto score = outputData[i + outputRow*4];
        if (score < confidence_thres)
            continue;
        // 从当前行提取边界框坐标
        float x = outputData[i];
        float y = outputData[i + outputRow];
        float w = outputData[i + outputRow*2];
        float h = outputData[i + outputRow*3];
        // 计算边界框的缩放坐标
        float left = static_cast<float>((x - w / 2) * x_factor);
        float top = static_cast<float>((y - h / 2) * y_factor);
        float width = static_cast<float>(w * x_factor);
        float height = static_cast<float>(h * y_factor);
        scores.push_back(score);
        boxes.push_back({ left, top, width, height });
    }
    return std::make_pair(boxes, scores);
}

/**
 * 绘制检测结果到输出图像上。
 * 
 * @param output_image 输出图像，绘制检测结果后的图像
 * @param results 包含检测边界框和得分的向量对
 */
void draw_resualt(cv::Mat& output_image, std::pair<std::vector<std::vector<float>>, std::vector<float>> resualts){
    std::vector<std::vector<float>> boxes = resualts.first;
    std::vector<float> scores = resualts.second;
    // 应用非最大抑制过滤重叠的边界框
    std::vector<int> indices = custom_NMSBoxes(boxes, scores, confidence_thres, iou_thres);
    // 遍历非最大抑制后的选定索引
    for (const auto& index : indices) {
        // 根据索引获取框、得分和类别ID
        const std::vector<float>& box = boxes[index];
        float score = scores[index];
        // 在输入图像上绘制检测结果
        draw_detections(output_image, box, score);
    }
}

/**
 * 进行图像处理和模型推断，生成检测结果并绘制到输出图像上。
 * 
 * @param input_image 输入图像
 * @param output_image 输出图像，绘制检测结果后的图像
 * @param predictor Paddle预测器
 */
void process(const cv::Mat& input_image, 
             cv::Mat& output_image,
             std::shared_ptr<PaddlePredictor> predictor){
    int input_width = INPUT_SHAPE[3];
    int input_height = INPUT_SHAPE[2];
    
    //设置输入张量大小并设置值
    auto inputTensor = predictor->GetInput(0);
    std::vector<int64_t> inputShape = {1, 3, input_height, input_width};
    inputTensor->Resize(inputShape);
    auto inputData = inputTensor->mutable_data<float>();

    cv::Mat preproc_image = preprocess(input_image,input_width,input_height);

    //对张量值inputData进行设置
    cv::imwrite("./output_img/preproc_img.jpg",preproc_image);
    NHWC3ToNC3HW(reinterpret_cast<const float *>(preproc_image.data), inputData,
               NULL, NULL, input_width, input_height);
    
    //Run predictor
    auto start = GetCurrentUS();
    predictor->Run();
    auto duration = (GetCurrentUS() - start) / 1000.0;
    std::cout << "process time duration:" << duration << std::endl;

    auto results = postprocess(input_image, predictor,input_width,input_height);
    std::vector<float> scores = results.second;
    if (!scores.empty()) {
        std::cout << "scores: " << scores[0] << std::endl;
        FALL_FLAGE = true;
    }
    output_image = input_image.clone();
    draw_resualt(output_image,results);
}

/**
 * 保存图像到指定路径。
 * 
 * @param input_image_path 输入图像的路径
 * @param output_image 要保存的图像
 */
void save_image(std::string input_image_path, const cv::Mat& output_image) {
    int start = input_image_path.find_last_of("/");
    int end = input_image_path.find_last_of(".");
    std::string img_name = input_image_path.substr(start + 1, end - start - 1);
    std::string result_name =
        "output/"+img_name + "_yolov8n_lite_falldetect.jpg";
    cv::imwrite(result_name, output_image);
}

int mqtt_publisher(mqtt::async_client& client,mqtt::connect_options& conn_opts){

    try {
        client.connect(conn_opts)->wait();

        std::string message = "Fall detected";

        // 获取当前时间
        auto now = std::chrono::system_clock::now();
        std::time_t timestamp = std::chrono::system_clock::to_time_t(now);

        // 将时间戳附加到消息文本中
        message += " : ";
        message += std::ctime(&timestamp);
        message.pop_back(); // 移除末尾的换行符
        std::cout << "发送mqtt消息：" << message << std::endl;

        mqtt::message_ptr pubmsg = mqtt::make_message("topic_fall", message);
        pubmsg->set_qos(1);
        client.publish(pubmsg)->wait();

        client.disconnect()->wait();
    } catch (const mqtt::exception& exc) {
        std::cerr << "Error: " << exc.what() << std::endl;
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    std::string srcPath;  //test file文件路径
    std::string img_mode = "image_test";
    std::string video_mode = "video_test";
    int mode;
    //检测输入参数设置检测模式
    if (argc  == 2) {
        mode = CAM;
        srcPath = argv[1];
    } else if (argc == 3) {
        std::string arg1 = argv[1];
        srcPath = argv[2];
        if (!arg1.compare(img_mode)) mode = IMAGE;
        else if (!arg1.compare(video_mode)) mode = VIDEO;
        else {
            printf("Usage: \n"
                "./yolov8_lite_arm_cpu.cc image_test <imageDirectory/>  :if you want test the image fall detection. \n"
                "./yolov8_lite_arm_cpu.cc video_test <videoFile>        :if you want test the video fall detection. \n"
                "./yolov8_lite_arm_cpu.cc </dev/video*>                 :if you want use the usb cam to detect. \n");
            return -1;
        }
    } else {
        printf("Usage: \n"
                "./yolov8_lite_arm_cpu.cc image_test <imageDirectory/>  :if you want test the image fall detection. \n"
                "./yolov8_lite_arm_cpu.cc video_test <videoFile>        :if you want test the video fall detection. \n"
                "./yolov8_lite_arm_cpu.cc </dev/video*>                 :if you want use the usb cam to detect. \n");
            return -1;
    }
    //配置mqtt服务
    mqtt::async_client client(SERVER_ADDRESS, CLIENT_ID);

    callback cb;
    client.set_callback(cb);

    mqtt::connect_options conn_opts;
    conn_opts.set_keep_alive_interval(20);

    //配置paddle 模型
    std::string model_path = "v8deploy_armopt.nb";
    create_color_palette();

    //1. set MobileConfig
    MobileConfig config;
    config.set_model_from_file(model_path);

    config.set_power_mode(LITE_POWER_NO_BIND);
    config.set_threads(2);

    // 2. Create PaddlePredictor by MobileConfig
    std::shared_ptr<PaddlePredictor> predictor =
        CreatePaddlePredictor<MobileConfig>(config);

    // 3. Read input source
    if (mode == IMAGE) {
        std::cout << "\nRun in Image mode, Process images in test_img/..." << std::endl;
        //读取文件夹中的图片
        std::cout << "\n======= benchmark summary =======\n" << std::endl;
        
        DIR* dir;
        struct dirent* entry;
        
        // 打开目录
        dir = opendir(srcPath.c_str());
        if (dir == nullptr) {
            std::cout << "Cant open the directory..." << std::endl;
            return 1;
        }

        // 遍历目录中的文件
        while ((entry = readdir(dir)) != nullptr) {
            std::string fileName = entry->d_name;
            std::string filePath = filePath + "/" + fileName;

            // 仅处理图像文件
            if (fileName.find(".jpg") != std::string::npos) {
                // 读取图像文件
                cv::Mat input_image = cv::imread(filePath);
                cv::Mat output_image = cv::Mat::zeros(input_image.size(), input_image.type());
                // 检查图像是否成功读取
                if (!input_image.empty()) {
                    // 调用处理函数处理图像
                    process(input_image, output_image, predictor);
                    save_image(filePath, output_image);
                }
            }
        }
        // 关闭目录
        closedir(dir);
        std::cout << "\nResult has been saved to output_img/: " << std::endl; 
    } else if(mode == VIDEO) {
        std::cout << "\nRun in Video mode..." << std::endl;
        cv::VideoCapture inputVideo(srcPath);
        if (!inputVideo.isOpened()){
            std::cout << "\n[ERROR]Could not open video\n" << std::endl;
            return -1;
        }

        // 获取输入视频的参数
        int frameWidth = inputVideo.get(cv::CAP_PROP_FRAME_WIDTH);
        int frameHeight = inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = inputVideo.get(cv::CAP_PROP_FPS);
        cv::VideoWriter outputVideo("output/falldet_output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frameWidth, frameHeight));

        // 逐帧检测
        cv::Mat frame;
        while(inputVideo.read(frame)){
            
            cv::Mat output_frame = cv::Mat::zeros(frame.size(), frame.type());
            process(frame, output_frame, predictor);
            outputVideo.write(output_frame);
        }
        std::cout << "Result has been saved to ./output/falldet_output.mp4 " << std::endl; 

        //释放资源
        inputVideo.release();
        outputVideo.release();

    } else if (mode == CAM){
        std::cout << "\nRun in USB Cam mode..." << std::endl;
        int fall_detecte_count = 0;
        cv::VideoCapture cap(srcPath);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640);
        if (!cap.isOpened()){
            std::cout << "\n[ERROR]Could not open camera\n" << std::endl;
            return -1;
        }

        while(1) {
            cv::Mat input_image;
            cap >> input_image;
            cv::Mat output_image = cv::Mat::zeros(input_image.size(), input_image.type());
            process(input_image, output_image, predictor);
            save_image("video", output_image);
            if (FALL_FLAGE && LAST_FLAGE) {
                fall_detecte_count ++;
                std::cout << "Detect fall times: " << fall_detecte_count << std::endl;
            }
            else fall_detecte_count = 0;
            LAST_FLAGE = FALL_FLAGE;
            FALL_FLAGE = 0;
            if (fall_detecte_count>=8) {
                FALL_DETECTED = true;
                std::cout << "\n======= !!!Fall Detected!!! =======\n" << std::endl;
                mqtt_publisher(client,conn_opts);
                FALL_DETECTED = false;
                fall_detecte_count = 0;
            }
            if (cv::waitKey(1) == char('q')) {
                break;
            }
        }
    }

    return 0;
}