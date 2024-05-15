#include<SoftwareSerial.h>
#include <ESP8266WiFi.h>
#include "ESP8266MQTT.h"
// #include <ESP8266MQTT.h>
#include "NTPClient.h"
#include <WiFiUdp.h>
#include <ArduinoJson.h>   

WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "ntp.aliyun.com"); //npt服务器，可修改
void onConnectionEstablished();

String content = "";
String dong = "";
String floor1 = "";
String room = "";

int rec_data_size = 100;
byte rec_data[100];
int bytesRead=0;
String time2= "";

ESP8266MQTT client(
  "new_huawei",                 // Wifi ssid
  "rennicai102",             // Wifi password
  "8.134.150.174",           // MQTT broker ip
  1883,                      // MQTT broker port
  "admin",              // MQTT username
  "admin",         // MQTT password
  WiFi.macAddress().c_str(),            // Client name,设置唯一,MAC，防冲突
  onConnectionEstablished, // Connection established callback
  true,               // Enable web updater
  true                // Enable debug messages
);

void setup() 
{
  // put your setup code here, to run once:
  Serial.begin(115200, SERIAL_8N1); //8个数据位，无奇偶校验，1个停止位
  timeClient.begin();
  timeClient.setTimeOffset(28800); //+1区，偏移3600，+8区，偏移3600*8
}

void onConnectionEstablished()
{
  // 订阅主题并且将该主题内收到的消息通过串口发送
  //主题名可更改，lab102
  client.subscribe("lab102", [](const String &payload) {
  // Serial.println(payload);//此处可以编写一个函数来代替
  });
  
    // 向某个主题发送消息，主题"820_cmd"
    //client.publish("t/air820", "bbbbbbbbbbbbbbbb");
    uint8_t macAddr[6]; // 建立保存mac地址的数组。用于以下语句
    WiFi.macAddress(macAddr);  
     
    //Serial.printf("通过转存数组获取MAC地址: %02x:%02x:%02x:%02x:%02x:%02x\n", macAddr[0], macAddr[1], macAddr[2], macAddr[3], macAddr[4], macAddr[5]);
    // 无参数调用macAddress时，ESP8266的mac地址将以字符串形式返回
    //Serial.printf("字符串获取MAC地址: %s\n", WiFi.macAddress().c_str());
    
}


void loop() 
{
  // DynamicJsonDocument data(256);
   client.loop();//mqtt连接  心跳连接
  
  // put your main code here, to run repeatedly:
  if(Serial.available()>0)
  {
    while(Serial.available()>0)  
    {
      bytesRead = Serial.readBytes(rec_data, rec_data_size);
        // 打印读取到的字节数，并换行
      // Serial.println(bytesRead);
//      rec += char(Serial.read());
      delay(2);
    }
    serial_text2() ;
    // content=" 呼救: ";
    // rec_disopse();
    
  //  client.publish("t/air820", content);//向主题发送消息
  }

  time_set();
  // Serial.print(7);
  // delay(1000);
  // content=" 呼救: ";
  // client.publish("t/air820", content);//向主题发送消息
  // delay(1000);

}

void serial_text2()
{
  String macstr = WiFi.macAddress().c_str(); 

  DynamicJsonDocument data(256);
 //序列化
  // data["mac"]= macstr;
  data["t"]=time2;
  data["cmd"]="help";
  data["imei"]=macstr;

  if(rec_data[0]==0x01)
    {
      // content +=" 呼救: depar:1 floor:2 room:3 ";
      // client.publish("t/air820", content);//向主题发送消息
      data["location"]=" apartment:1 floor:2 room:3 ";
    }

    if(rec_data[0]==0x02)
    {
      // content +=" 呼救: depar:4 floor:5 room:6 ";
      // client.publish("t/air820", content);//向主题发送消息
      data["location"]=" apartment:4 floor:5 room:6 ";
    }

    if(rec_data[0]==0x03)
    {
      // content +=" 呼救: depar:7 floor:8 room:9 ";
      // client.publish("t/air820", content);//向主题发送消息
      data["location"]=" apartment:7 floor:8 room:9 ";
    }
  
  char json_string[256];
  serializeJson(data,json_string);

  client.publish("t/air820", json_string);//向主题发送消息
}



void time_set()
{
  //更新时间
  int timeupdate = 0;

  if(timeupdate%100 == 0)
  {
      timeClient.update();//更新时间,时间更新太频繁似乎会导致不准
  }
  if(timeupdate>2147483640)
  {
    timeupdate = 0;
  }
  
  unsigned long epochTime = timeClient.getEpochTime();//epochTime,NTP时间
  time_t mytime = (int)epochTime;
  struct tm* timeinfo;
  char buffer[128];
  timeinfo = localtime(&mytime);
  strftime(buffer,sizeof(buffer),"%Y-%m-%d %H:%M:%S",timeinfo);//转化为年月日时分秒格式
  // Serial.println(buffer);
  time2 = (String) buffer;//转化格式
  // Serial.println(time2);
//   String macstr = WiFi.macAddress().c_str(); 

//   DynamicJsonDocument data(256);


//  //序列化
//   // data["mac"]= macstr;
//   data["t"]=time2;
//   data["cmd"]="help";
//   data["imei"]=macstr;
//   char json_string[256];
//   serializeJson(data,json_string);
  //String t = timeClient.getFormattedTime();//时分秒时间，无年月日
  
  
  
//   while (Serial.available() > 0)  
//    {
//        content += char(Serial.read());  //使用这个直接循环加起来
//        delay(2);
//    }

   
      // client.publish("t/air820", json_string);//向主题发送消息
      // String content = "";
     
    delay(100);
  // Serial.println(json_string);

}

void serial_text()
{
  
  // String time2 = (String) buffer;//转化格式
  // content += buffer;
  String macstr = WiFi.macAddress().c_str();

  Serial.println(content);

  if(rec_data[0]==0x01)
    {
      content +=" 呼救: depar:1 floor:2 room:3 ";
      client.publish("t/air820", content);//向主题发送消息
    }

    if(rec_data[0]==0x02)
    {
      content +=" 呼救: depar:4 floor:5 room:6 ";
      client.publish("t/air820", content);//向主题发送消息
    }

    if(rec_data[0]==0x03)
    {
      content +=" 呼救: depar:7 floor:8 room:9 ";
      client.publish("t/air820", content);//向主题发送消息
    }

  // Serial.println(content);
  content +=" MAC: ";
  content += macstr;
  content="";
}

void rec_disopse()
{
    //  client.publish("t/air820", content);//向主题发送消息
  // for(int i=0;i<bytesRead;i++)
  //   {
  //     Serial.println(rec_data[i],HEX);
  //   }
    if(rec_data[0]==0x10&&rec_data[1]==0x12)
    {
      Serial.println("收到的数据为：00");
      // content += "dong:";
      switch(rec_data[3]) //楼栋数
      {
        case  1:
          Serial.println("dong1");
          content += "dong:1 ";
          break;
        case  2:
          Serial.println("dong2");
          content += "dong:2 ";
          break;
        case  3:
          Serial.println("dong3");
          content += "dong:3 ";
          break;
        case  4:
          Serial.println("dong4");
          content += "dong:4 ";
          break;
        case  5:
          Serial.println("dong5");
          content += "dong:5 ";
          break;
        case  6:
          Serial.println("dong6");
          content += "dong:6 ";
          break;
        case  7:
          Serial.println("dong6");
          content += "dong:7 ";
          break;
      }

      // content += " floor ";
      switch(rec_data[4]) //层数
      {
        case  1:
          Serial.println("floor1");
          content += " floor:1 ";
          break;
        case  2:
          Serial.println("floor2");
          content += " floor:2 ";
          break;
        case  3:
          Serial.println("floor3");
          content += " floor:3 ";
          break;
        case  4:
          Serial.println("floor4");
          content += " floor:4 ";
          break;
        case  5:
          Serial.println("floor5");
          content += " floor:5 ";
          break;
        case  6:
          Serial.println("floor6");
          content += " floor:6 ";
          break;
        case  7:
          Serial.println("floor7");
          content += " floor:7 ";
          break;
      }

      // content += " room ";
      switch(rec_data[5]) //房间号
      {
        case  1:
          Serial.println("room1");
          content += " room:1 ";
          break;
        case  2:
          Serial.println("room2");
          content += " room:2 ";
          break;
        case  3:
          Serial.println("room3");
          content += " room:3 ";
          break;
        case  4:
          Serial.println("room4");
          content += " room:4 ";
          break;
        case  5:
          Serial.println("room5");
          content += " room:5 ";
          break;
        case  6:
          Serial.println("room6");
          content += " room:6 ";
          break;
        case  7:
          Serial.println("room7");
          content += " room:7 ";
          break;
      }

      client.publish("t/air820", content);//向主题发送消息
      Serial.println(content);
    }
    else
    {
      Serial.println("zhentouerror1");
      Serial.println(01,HEX);
    }
}


//    char rec_shuzhu[rec.length() + 1];  
//    strcpy(rec_shuzhu, rec.c_str());
//    Serial.println(rec，HEX);
//      a= Serial.available();
//      Serial.println(Serial.available());
