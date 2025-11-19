---
title: Kafka安装部署
date: 2025-04-07
permalink: /kafka/Kafka安装部署.html
tags:
  - Kafka
categories:
  - Kafka
---

## 环境准备

### 前置条件
- JDK 1.8+
- ZooKeeper 3.4.6+
- 操作系统：Linux（推荐CentOS 7+）
- 内存：最少4GB RAM
- 磁盘：根据数据量规划，建议50GB以上

### 下载安装包
1. 访问[Kafka官网](https://kafka.apache.org/downloads)下载最新稳定版本
2. 选择二进制版本，例如：kafka_2.13-3.9.0.tgz

## 单机版安装

### 安装步骤
1. 解压安装包：
```bash
tar -xzf kafka_2.13-3.9.0.tgz
cd kafka_2.13-3.9.0
```

2. 启动ZooKeeper服务：
```bash
bin/zookeeper-server-start.sh config/zookeeper.properties
```

3. 启动Kafka服务：
```bash
bin/kafka-server-start.sh config/server.properties
```

## 集群版安装

### 集群规划
- 建议3台及以上服务器
- 每台服务器配置相同
- 网络互通，时钟同步

### 安装步骤
1. 在每台服务器上解压安装包
2. 修改配置文件config/server.properties：
```properties
# broker.id需要唯一
broker.id=0

# 监听地址
listeners=PLAINTEXT://your.host.name:9092

# ZooKeeper连接地址
zookeeper.connect=zk1:2181,zk2:2181,zk3:2181
```

3. 依次启动各节点服务

## 配置说明

### 重要参数
```properties
# 数据存储目录
log.dirs=/path/to/kafka-logs

# 默认分区数
num.partitions=3

# 数据保留时间
log.retention.hours=168

# 单个日志文件大小
log.segment.bytes=1073741824
```

### 性能调优
- 增加分区数提高并行度
- 适当调整replica.lag.time.max.ms
- 优化JVM参数
- 使用多磁盘存储

## 安装验证

### 创建测试主题
```bash
bin/kafka-topics.sh --create --topic test --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
```

### 发送测试消息
```bash
bin/kafka-console-producer.sh --topic test --bootstrap-server localhost:9092
```

### 消费测试消息
```bash
bin/kafka-console-consumer.sh --topic test --from-beginning --bootstrap-server localhost:9092
```

## 常见问题

### 启动失败
- 检查JDK版本和环境变量
- 确认ZooKeeper服务状态
- 查看日志文件排查错误

### 连接超时
- 检查防火墙设置
- 验证网络连通性
- 确认配置文件中的监听地址

### 性能问题
- 检查磁盘IO
- 监控JVM内存使用
- 优化网络配置

## 运维建议

### 监控指标
- 消息积压量
- 消费延迟
- 磁盘使用率
- JVM堆内存使用

### 备份策略
- 定期备份配置文件
- 重要主题数据备份
- 制定灾备方案



