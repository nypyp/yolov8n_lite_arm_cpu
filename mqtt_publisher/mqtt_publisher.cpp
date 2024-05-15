#include <iostream>
#include <mqtt/async_client.h>
#include <chrono>
#include <ctime>

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

int main() {
  mqtt::async_client client(SERVER_ADDRESS, CLIENT_ID);

  callback cb;
  client.set_callback(cb);

  mqtt::connect_options conn_opts;
  conn_opts.set_keep_alive_interval(20);

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