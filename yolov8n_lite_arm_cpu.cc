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
#include <json/json.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>


using namespace paddle::lite_api;

int IMAGE = 0;
int VIDEO = 1;
int CAM = 2;
bool FALL_FLAGE = 0;
bool LAST_FLAGE = 0;


// 定义类别到类名的映射
std::unordered_map<int, std::string> classes = {
    {0, "Fall"},{1, "Dog"},{2, "People"}
};

// 定义置信度阈值和IoU阈值
float confidence_thres = 0.70;
float iou_thres = 0.5;
//配置paddle 模型
std::string model_path = "yolov8_falldet_3cls_arm_opt.nb";
//配置mqtt服务器地址及话题
std::string server_address = "tcp://8.134.150.174:1883";
std::string mqtt_topic = "820_cmd";
//配置房间号
std::string location = "apartment:1 floor:2 room:3";
//mac 用于mqtt clint唯一标识，防冲突
std::string mac = "xxxx-xxxx-xxxx";
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

class callback : public mqtt::callback {
  void connection_lost(const std::string& cause) override {
    std::cout << "Connection lost: " << cause << std::endl;
  }

  void delivery_complete(mqtt::delivery_token_ptr delivery_token) override {
    std::cout << "Message delivery complete" << std::endl;
  }
};

void readJSON(const std::string& filename) {
    // 从文件中读取 JSON 数据
    std::ifstream ifs(filename);
    Json::Reader reader;
    Json::Value root;

    if (!reader.parse(ifs, root, false)) {
        std::cerr << "Failed to parse JSON" << std::endl;
        return;
    }

    // 读取字段
    confidence_thres = root["confidence_thres"].asFloat();
    model_path = root["model_path"].asString();
    server_address = root["SERVER_ADDRESS"].asString();
    mqtt_topic = root["mqtt_topic"].asString();
    location = root["location"].asString();

    // 打印读取的字段
    std::cout << "::::==========配置参数============::::" << std::endl;
    std::cout << "::::=============================::::" << std::endl;
    std::cout << "| confidence_thres:   " << confidence_thres << std::endl;
    std::cout << "| model_path:         " << model_path << std::endl;
    std::cout << "| SERVER_ADDRESS:     " << server_address << std::endl;
    std::cout << "| mqtt_topic:         " << mqtt_topic << std::endl;
    std::cout << "| location:           " << location << std::endl;
    std::cout << "::::=============================::::" << std::endl;
}

std::string getMacAddress() {
    std::ifstream file("/sys/class/net/eth0/address");
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file");
    }

    std::string macAddress;
    std::getline(file, macAddress);
    file.close();

    return macAddress;
}

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
void draw_detections(cv::Mat &output_image,int class_id, std::vector<float> box, float score) {
    // 提取边界框的坐标
    float x1 = box[0];
    float y1 = box[1];
    float w = box[2];
    float h = box[3];

    // 根据类别ID检索颜色
    cv::Scalar color = color_palette[class_id];

    // 在图像上绘制边界框
    cv::rectangle(output_image, cv::Point(x1, y1), cv::Point(x1 + w, y1 + h), color, 2);

    // 创建标签文本，包括类名和得分
    std::string label = classes[class_id] + ": " + std::to_string(score);

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
 * @brief 将NHWC格式的3通道图像数据转换为NC3HW格式
 * 
 * @param src 指向输入图像数据的指针，格式为NHWC
 * @param dst 指向输出图像数据的指针，格式为NC3HW
 * @param width 图像的宽度
 * @param height 图像的高度
 */
void NHWC3ToNC3HW(const float *src, float *dst, int width, int height) {
  int size = height * width;
  float *dst_c0 = dst;
  float *dst_c1 = dst + size;
  float *dst_c2 = dst + size * 2;
  int i = 0;
  for (; i < size; i++) {
    *(dst_c0++) = *(src++);
    *(dst_c1++) = *(src++);
    *(dst_c2++) = *(src++);
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

    //将图像颜色空间从BGR转换为RGB
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);

    // 使用letterbox将图像大小调整为匹配输入形状
    cv::Mat resized_img;
    resized_img = resizeImage(rgb_img,cv::Size(input_width,input_height),true);
    cv::imwrite("./output_img/preproc_img.jpg",resized_img);

    // 通过除以255.0来归一化图像数据
    cv::Mat normalized_img;
    resized_img.convertTo(normalized_img, CV_32F, 1.0 / 255.0);

    // 返回预处理后的图像数据
    return normalized_img;
}

/**
 * 后处理函数，用于解析模型输出并生成检测结果。
 * 
 * @param input_image 原始图像
 * @param predictor Paddle预测器
 * @param input_width 预处理后图像的宽度
 * @param input_height 预处理后图像的高度
 * @return 包含检测边界框和得分的tuple
 */
std::tuple<std::vector<int>, std::vector<std::vector<float>>, std::vector<float>> postprocess(const cv::Mat input_image, 
                                                                           std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor, 
                                                                           int input_width, 
                                                                           int input_height) {
    int img_height = input_image.rows;
    int img_width = input_image.cols;
    auto outputTensor = predictor->GetOutput(0);
    auto outputData = outputTensor->data<float>();
    auto outputShape = outputTensor->shape();
    auto outputRow = outputShape[2];
    auto outputCol = outputShape[1];
    // 用于存储检测的边界框、得分和类别ID的向量
    std::vector<std::vector<float>> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    // 计算边界框坐标的缩放因子
    float x_factor = static_cast<float>(img_width) / input_width;
    float y_factor = static_cast<float>(img_height) / input_height;
    int class_num = outputCol - 4;
    // 遍历输出数组的每一行
    for (int i = 0; i < outputRow; i += 1) {

        std::vector<float> score_of_class;
        // 提取当前行所有类得分，并得出最高分
        for (int j = 0; j < class_num; j += 1){
            score_of_class.push_back(outputData[i + outputRow*(4+j)]);
        }
        auto max_score_itr = std::max_element(score_of_class.begin(),score_of_class.end());
        float max_score = *max_score_itr;
        if (max_score < confidence_thres)
            continue;
        // 获取最高分索引
        int max_score_index = std::distance(score_of_class.begin(), max_score_itr);
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
        class_ids.push_back(max_score_index);
        scores.push_back(max_score);
        boxes.push_back({ left, top, width, height });
    }
    return std::make_tuple(class_ids, boxes, scores);
}

/**
 * 绘制检测结果到输出图像上。
 * 
 * @param output_image 输出图像，绘制检测结果后的图像
 * @param results 包含检测边界框和得分的向量对
 */
void draw_resualt(cv::Mat& output_image, std::vector<int> class_ids, std::vector<std::vector<float>> boxes, std::vector<float> scores ){
    // 应用非最大抑制过滤重叠的边界框
    std::vector<int> indices = custom_NMSBoxes(boxes, scores, confidence_thres, iou_thres);
    // 遍历非最大抑制后的选定索引
    for (const auto& index : indices) {
        // 根据索引获取框、得分和类别ID
        int class_id = class_ids[index];
        const std::vector<float>& box = boxes[index];
        float score = scores[index];
        // 在输入图像上绘制检测结果
        draw_detections(output_image, class_id, box, score);
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
    NHWC3ToNC3HW(reinterpret_cast<const float *>(preproc_image.data), inputData, input_width, input_height);
    
    //Run predictor
    auto start = GetCurrentUS();
    predictor->Run();
    auto duration = (GetCurrentUS() - start) / 1000.0;
    std::cout << "process time duration:" << duration << std::endl;

    auto [class_ids, boxes, scores] = postprocess(input_image, predictor,input_width, input_height);
    auto it = std::find(class_ids.begin(),class_ids.end(), 0);
    if ( it != class_ids.end()) {
        int index = std::distance(class_ids.begin(),it);
        std::cout << "scores: " << scores[index] << std::endl;
        FALL_FLAGE = true;
    }
    output_image = input_image.clone();
    draw_resualt(output_image, class_ids, boxes, scores);
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

        //json OBJ
        Json::Value json_msg;
        json_msg["type"] = "Fall";

        // 获取当前时间
        auto now = std::chrono::system_clock::now();
        std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
        char time_str[100];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", std::localtime(&timestamp));
        json_msg["time"] = time_str;

        // 当前房间
        json_msg["location"] = location;
        //imei
        json_msg["imei"] = "";
        //cmd
        json_msg["cmd"] = "help";
        /*完整mqtt接口
        {
            "time": "2024-06-11 15:18:41",
            "cmd": "help",
            "imei": "",
            "location": "apartment:1 floor:2 room:3",
            "type": "Fall"
        }*/

        //json to string
        Json::StreamWriterBuilder wbuilder;
        std::string message = Json::writeString(wbuilder, json_msg);

        std::cout << "发送mqtt消息：" << message << std::endl;

        mqtt::message_ptr pubmsg = mqtt::make_message("820_cmd", message);
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
    readJSON("config.json");
    try {
        mac = getMacAddress();
        std::cout << "MAC address: " << mac << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    //配置mqtt服务
    //mqtt服务配置
    const std::string SERVER_ADDRESS(server_address);
    const std::string CLIENT_ID(mac);
    mqtt::async_client client(SERVER_ADDRESS, CLIENT_ID);

    callback cb;
    client.set_callback(cb);

    mqtt::connect_options conn_opts;
    conn_opts.set_keep_alive_interval(20);

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
            std::string filePath = srcPath + "/" + fileName;

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
        cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
        while(1) {
            cv::Mat input_image;
            cap >> input_image;
            cv::Mat output_image = cv::Mat::zeros(input_image.size(), input_image.type());
            process(input_image, output_image, predictor);
            //save_image("Camfram", output_image);
            // try
            // {
            //     /* code */
            //     // cv::imshow("Output", output_image); // 显示处理后的图像

            // }
            // catch(const std::exception& e)
            // {
            //     std::cerr << e.what() << '\n';
            // }
            
            if (FALL_FLAGE && LAST_FLAGE) {
                fall_detecte_count ++;
                std::cout << "Detect fall times: " << fall_detecte_count << std::endl;
            }
            else fall_detecte_count = 0;
            LAST_FLAGE = FALL_FLAGE;
            FALL_FLAGE = 0;
            if (fall_detecte_count>=8) {
                std::cout << "\n======= !!!Fall Detected!!! =======\n" << std::endl;
                mqtt_publisher(client,conn_opts);
                fall_detecte_count = 0;
            }
            if (cv::waitKey(1) == char('q')) {
                break;
            }
        }
        cv::destroyAllWindows();
    }

    return 0;
}
