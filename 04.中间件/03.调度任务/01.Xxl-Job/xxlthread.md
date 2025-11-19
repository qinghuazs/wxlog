---
title: xxl-Job中的线程
date: 2024/10/22
---

在 xxl-job 中有很多自建的线程，如清理日志文件的线程类 JobLogFileCleanThread ，本文针对 xxl-job 自建线程进行整理，希望能为后面的开发提供一些参考。

## JobLogFileCleanThread

### 单例对象

日志文件清理线程类，主要作用是每天清理 3 天前的日志文件。

第一个设计点，单例模式，保证 JobLogFileCleanThread 对象在实例中唯一，避免出现重复创建线程的作用。

这里的单例模式是用静态对象来实现的。

```java 
private static JobLogFileCleanThread instance = new JobLogFileCleanThread();
public static JobLogFileCleanThread getInstance(){
	return instance;
}
```

### 属性

JobLogFileCleanThread 中有 2 个属性：`localThread` 和 `toStop`。

localThread 是一个线程对象，负责进行日志文件清理。

toStop 是 boolean 对象，默认 false，线程中断标识，当 toStop 为 true 时，线程中断执行。

### start 方法

Start() 方法用于在执行器启动时初始化 JobLogFileCleanThread ，该方法的作用主要是创建并启动 localThread 线程，定期清理 logRetentionDays 天前的日志文件。

localThread 设置成后台线程，保证当服务停止时，日志文件清理线程也能同步停止。

while循环保证清理日志文件的动作能一直执行。

toStop 保证能跳出循环，终止线程。

TimeUnit.DAYS.sleep(1) 的目的是为了达到每天执行一次的效果，防止线程一直占用 CPU 资源。

我们在开发一些后端定时任务的时候，也可以参考这种实现方式，但是要保证线程能终止，而不是无限循环，且一直占用 CPU 资源。

start 方法在容器启动时调用。

```java 
public void start(final long logRetentionDays){
    //限制最小值为 3
    if (logRetentionDays < 3 ) {
        return;
    }

    localThread = new Thread(new Runnable() {
        @Override
        public void run() {
            //while 保证重复执行，toStop 保证能跳出循环，终止线程
            while (!toStop) {
                try {
                    // 清理日志文件逻辑，不是本文的重点，所以不放了       
                } catch (Exception e) {
                    if (!toStop) {
                        logger.error(e.getMessage(), e);
                    }
                }
                try {
                    //清理完成后，沉睡 1 天，达到每天执行一次的效果
                    TimeUnit.DAYS.sleep(1);
                } catch (InterruptedException e) {
                    if (!toStop) {
                        logger.error(e.getMessage(), e);
                    }
                }
            }
            logger.info(">>>>>>>>>>> xxl-job, executor JobLogFileCleanThread thread destroy.");
        }
    });
    //设置成后台线程，保证当服务停止时，日志文件清理线程也能同步停止。
    localThread.setDaemon(true);
    //设置线程名称是为了看日志方便
    localThread.setName("xxl-job, executor JobLogFileCleanThread");
    //启动线程
    localThread.start();
}
```

### stop 方法

该方法的作用是终止 localThread 的执行。

```java
public void toStop() {
    //toStop 设置为 true，while 循环终止
    toStop = true;

    if (localThread == null) {
        return;
    }

    // interrupt会请求线程终止，将线程的中断状态设置为 true，该方法不会强制停止线程执行
    localThread.interrupt();
    try {
        //等待线程终止
        localThread.join();
    } catch (InterruptedException e) {
        logger.error(e.getMessage(), e);
    }
}
```

localThread.join() 在这里的作用是为了阻塞调用方，保证 localThread 完成它的工作之后，再进行其他的操作。

stop 方法在容器销毁时调用。







