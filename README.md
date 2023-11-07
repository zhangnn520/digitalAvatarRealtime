# DigtalAvatarRealtime

数字人的实时推流和视频服务

## 安装

演示在Ubuntu22.04上。

requirements.txt中的torch系列库的安装最好到pytorch官网查询具体cuda编译版本的安装命令。

```shell
sudo apt-get install ffmpeg
pip install -r requirements.txt
```

## 配置文件

configuration/development_config.py

## 启动

```shell
nohup python main.py &
```

## 客户端 HTTP API文档

[HTTP API.md](HTTP%20API.md)