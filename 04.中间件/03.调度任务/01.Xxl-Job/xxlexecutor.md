---
title: 执行器
date: 2024/04/18
---

XXL-Job 是一个分布式任务调度平台，而 XXL-Job 的执行器则是用于执行调度任务的组件。执行器负责从调度中心获取任务，并按照任务定义的调度策略执行任务。XXL-Job 的执行器支持分布式部署，可以在不同的服务器上部署多个执行器实例，以实现任务的并行执行和高可用性。

XXL-Job 的执行器通过配置和启动执行器实例，将其加入到 XXL-Job 的调度中心中，以便统一管理和调度任务的执行。

执行器的核心类是 `XxlJobExecutor`。这个类负责执行器的初始化、任务的注册和执行等操作。在执行器启动时，会创建一个 `XxlJobExecutor` 实例，并通过配置文件等方式配置执行器的参数，然后启动执行器，使其开始监听调度中心下发的任务并执行。

执行器的主要功能：

1. **初始化**：执行器启动时，进行一些初始化操作，如加载配置、初始化线程池等。
2. **注册任务**：将执行器注册到调度中心，以接收调度中心下发的任务。
3. **执行任务**：根据调度中心下发的任务信息，执行具体的任务逻辑。
4. **通知任务状态**：执行器执行完任务后，将任务执行结果反馈给调度中心，以便调度中心进行任务执行情况的监控和统计。
5. **关闭**：执行器停止时，进行资源释放等清理操作。

## XxlJobExecutor

### 属性

`XxlJobExecutor` 的属性，这些属性可以在执行器的工程的application.properties文件中配置。

```java
 	/**
     * 调度中心地址
     * xxl.job.admin.addresses
     */
    private String adminAddresses;
    /**
     * 访问token
     * xxl.job.accessToken
     */
    private String accessToken;
    /**
     * 应用名称
     * xxl.job.executor.appname
     */
    private String appname;
    /**
     * 执行器地址，如果配置了address则忽略IP和port的配置
     * xxl.job.executor.address
     * 如果未配置，则取本级地址
     */
    private String address;
    /**
     * IP地址
     * xxl.job.executor.ip
     */
    private String ip;
    /**
     * 端口
     * xxl.job.executor.port
     */
    private int port;
    /**
     * 日志路径
     * xxl.job.executor.logpath
     */
    private String logPath;
    /**
     * 日志保留天数
     * xxl.job.executor.logretentiondays
     */
    private int logRetentionDays;

```

#### logPath

`logPath` 主要用于在 `XxlJobFileAppender` 中初始化日志路径，如果没有配置 logPath，则使用 logBasePath 作为日志路径，logBasePath 的值默认为 `/data/applogs/xxl-job/jobhandler` 。

```java
private static String logBasePath = "/data/applogs/xxl-job/jobhandler";
private static String glueSrcPath = logBasePath.concat("/gluesource");
public static void initLogPath(String logPath){
    // init
    if (logPath!=null && logPath.trim().length()>0) {
        logBasePath = logPath;
    }
    // mk base dir
    File logPathDir = new File(logBasePath);
    if (!logPathDir.exists()) {
        logPathDir.mkdirs();
    }
    logBasePath = logPathDir.getPath();

    // mk glue dir
    File glueBaseDir = new File(logPathDir, "gluesource");
    if (!glueBaseDir.exists()) {
        glueBaseDir.mkdirs();
    }
    glueSrcPath = glueBaseDir.getPath();
}
```

#### adminAddresses 和 accessToken

adminAddresses 和 accessToken 的作用是初始化 AdminBiz 对象，并加入到 adminBizList 列表中。AdminBiz 主要用于调用调度中心的 RPC 接口。AdminBiz 中封装了 registry 注册接口、registryRemove 注销接口和 callback 回调接口。

```java
private static List<AdminBiz> adminBizList;
private void initAdminBizList(String adminAddresses, String accessToken) throws Exception {
    if (adminAddresses!=null && adminAddresses.trim().length()>0) {
        for (String address: adminAddresses.trim().split(",")) {
            if (address!=null && address.trim().length()>0) {
                AdminBiz adminBiz = new AdminBizClient(address.trim(), accessToken);
                if (adminBizList == null) {
                    adminBizList = new ArrayList<AdminBiz>();
                }
                adminBizList.add(adminBiz);
            }
        }
    }
}
public static List<AdminBiz> getAdminBizList(){
    return adminBizList;
}
```

#### logRetentionDays

logRetentionDays 日志保留天数，JobLogFileCleanThread 中启用一个守护线程 localThread 根据该字段去定期清理日志文件。目前限定了 logRetentionDays 不能小于3，localThread 线程每天执行一次。

#### embedServer

根据 address、ip、port、appname和accessToken 来初始化 embedServer 。address 有值时会忽略 ip 和 port ，如果 address 未设置，则根据 ip 和  port 组装 address。

`EmbedServer` 是嵌入式调度中心的服务器类，用于启动和管理嵌入式调度中心。嵌入式调度中心是 XXL-JOB 的一种运行模式，它将调度中心集成到应用程序中，无需单独部署调度中心服务器，方便快捷地实现任务调度。

```java
private EmbedServer embedServer = null;
private void initEmbedServer(String address, String ip, int port, String appname, String accessToken) throws Exception {
    // generate address
    if (address==null || address.trim().length()==0) {
        // fill ip port
        ip = (ip!=null&&ip.trim().length()>0)?ip: IpUtil.getIp();
        port = port>0?port: NetUtil.findAvailablePort(9999);

        String ip_port_address = IpUtil.getIpPort(ip, port);   // registry-address：default use address to registry , otherwise use ip:port if address is null
        address = "http://{ip_port}/".replace("{ip_port}", ip_port_address);
    }

    // accessToken
    if (accessToken==null || accessToken.trim().length()==0) {
        logger.warn(">>>>>>>>>>> xxl-job accessToken is empty. To ensure system security, please set the accessToken.");
    }

    // start
    embedServer = new EmbedServer();
    embedServer.start(address, port, appname, accessToken);
}
```

### start() 方法

start() 方法主要是基于执行器的属性进行初始化操作，具体的初始化内容在上面都涉及到了，这里就贴一下代码。

```java
public void start() throws Exception {
    //初始化日志路径
    XxlJobFileAppender.initLogPath(logPath);

    //初始化调度中心信息
    initAdminBizList(adminAddresses, accessToken);

    //初始化并启动日志清理线程
    JobLogFileCleanThread.getInstance().start(logRetentionDays);

    //初始化并启动触发器回调线程
    TriggerCallbackThread.getInstance().start();

    //初始化嵌入式执行器服务信息
    initEmbedServer(address, ip, port, appname, accessToken);
}
```

XxlJobSpringExecutor 实现了 SmartInitializingSingleton，在 afterSingletonsInstantiated() 方法会调用 start() 方法。

`afterSingletonsInstantiated` 是 Spring 中的一个回调方法，用于在单例对象实例化完成后进行一些额外的初始化操作。在 Spring 容器启动时，会首先实例化所有的单例对象，并在所有单例对象实例化完成后调用 `afterSingletonsInstantiated` 方法。

TriggerCallbackThread 特殊一点，没有依赖 XxlJobExecutor 的属性进行创建，单独说一下。

TriggerCallbackThread 用于触发任务执行后的回调处理，主要封装了2个回调线程 triggerCallbackThread 和 triggerRetryCallbackThread。当调度中心触发任务执行后，会通过回调的方式通知执行器任务的执行结果，`TriggerCallbackThread` 就是负责处理这个回调通知的线程类。回调的内容会存入到 callBackQueue 阻塞队列中，triggerCallbackThread 会定期从 callBackQueue 中获取数据，然后调用 doCallback() 进行处理。doCallback() 方法中会向所有的调度中心发送回调接口，直到有一个调度中心响应成功，同时写入日志。如果所有的调度中心都没有回调成功，则表示回调失败，将回调信息通过appendFailCallbackFile() 方法写入到回调失败文件中，供 triggerRetryCallbackThread 线程重试使用。

```java
// callback
triggerCallbackThread = new Thread(new Runnable() {
    @Override
    public void run() {
        // normal callback
        while(!toStop){
            try {
                HandleCallbackParam callback = getInstance().callBackQueue.take();
                if (callback != null) {

                    // callback list param
                    List<HandleCallbackParam> callbackParamList = new ArrayList<HandleCallbackParam>();
                    int drainToNum = getInstance().callBackQueue.drainTo(callbackParamList);
                    callbackParamList.add(callback);

                    // callback, will retry if error
                    if (callbackParamList!=null && callbackParamList.size()>0) {
                        doCallback(callbackParamList);
                    }
                }
            } catch (Exception e) {
                if (!toStop) {
                    logger.error(e.getMessage(), e);
                }
            }
        }

        // last callback 这里是防止线程终止时callBackQueue中还有数据没有处理
        try {
            List<HandleCallbackParam> callbackParamList = new ArrayList<HandleCallbackParam>();
            int drainToNum = getInstance().callBackQueue.drainTo(callbackParamList);
            if (callbackParamList!=null && callbackParamList.size()>0) {
                doCallback(callbackParamList);
            }
        } catch (Exception e) {
            if (!toStop) {
                logger.error(e.getMessage(), e);
            }
        }
        logger.info(">>>>>>>>>>> xxl-job, executor callback thread destroy.");

    }
});
```

triggerRetryCallbackThread 的作用主要是进行回调失败的补偿机制，从回调失败日志文件中读取文件内容，重新执行 doCallback() 方法，每30秒执行一次（RegistryConfig.BEAT_TIMEOUT 默认配置是 30 秒）。

```java
triggerRetryCallbackThread = new Thread(new Runnable() {
    @Override
    public void run() {
        while(!toStop){
            try {
                retryFailCallbackFile();
            } catch (Exception e) {
                if (!toStop) {
                    logger.error(e.getMessage(), e);
                }
            }
            try {
                TimeUnit.SECONDS.sleep(RegistryConfig.BEAT_TIMEOUT);
            } catch (InterruptedException e) {
                if (!toStop) {
                    logger.error(e.getMessage(), e);
                }
            }
        }
        logger.info(">>>>>>>>>>> xxl-job, executor retry callback thread destroy.");
    }
});
private void retryFailCallbackFile(){
    // valid   获取回调失败的日志文件路径
    File callbackLogPath = new File(failCallbackFilePath);
    if (!callbackLogPath.exists()) {
        return;
    }
    if (callbackLogPath.isFile()) {
        callbackLogPath.delete();
    }
    if (!(callbackLogPath.isDirectory() && callbackLogPath.list()!=null && callbackLogPath.list().length>0)) {
        return;
    }

    // load and clear file, retry
    for (File callbaclLogFile: callbackLogPath.listFiles()) {
        //读取文件内容
        byte[] callbackParamList_bytes = FileUtil.readFileContent(callbaclLogFile);
        // avoid empty file
        if(callbackParamList_bytes == null || callbackParamList_bytes.length < 1){
            callbaclLogFile.delete();
            continue;
        }
        //反序列化到list中再进行回调处理
        List<HandleCallbackParam> callbackParamList = (List<HandleCallbackParam>) JdkSerializeTool.deserialize(callbackParamList_bytes, List.class);
        callbaclLogFile.delete();
        doCallback(callbackParamList);
    }
}
```



### destroy() 方法

destroy() 方法主要是在执行器销毁时清理对象、停止线程等。

```java
public void destroy(){
    //销毁嵌入式执行器
    stopEmbedServer();

    // 销毁 jobThreadRepository
    if (jobThreadRepository.size() > 0) {
        for (Map.Entry<Integer, JobThread> item: jobThreadRepository.entrySet()) {
            JobThread oldJobThread = removeJobThread(item.getKey(), "web container destroy and kill the job.");
            // wait for job thread push result to callback queue
            if (oldJobThread != null) {
                try {
                    oldJobThread.join();
                } catch (InterruptedException e) {
                    logger.error(">>>>>>>>>>> xxl-job, JobThread destroy(join) error, jobId:{}", item.getKey(), e);
                }
            }
        }
        jobThreadRepository.clear();
    }
    jobHandlerRepository.clear();


    // 销毁 JobLogFileCleanThread
    JobLogFileCleanThread.getInstance().toStop();

    // 销毁 TriggerCallbackThread
    TriggerCallbackThread.getInstance().toStop();
}
private void stopEmbedServer() {
    // stop provider factory
    if (embedServer != null) {
        try {
            embedServer.stop();
        } catch (Exception e) {
            logger.error(e.getMessage(), e);
        }
    }
}
```

销毁线程的方法都大同小异，不外乎设置 toStop为true，再进行 interrupt() 和  join()。

```java
public void toStop() {
    toStop = true;
    // interrupt and wait
    if (registryThread != null) {
        registryThread.interrupt();
        try {
            registryThread.join();
        } catch (InterruptedException e) {
            logger.error(e.getMessage(), e);
        }
    }
}
```

在 XxlJobSpringExecutor 中，实现了 DisposableBean 接口，重写了 DisposableBean 的 destroy() 方法，在该方法中直接引用了 XxlJobExecutor 的  destroy() 方法。

```java
public class XxlJobSpringExecutor extends XxlJobExecutor implements ApplicationContextAware, SmartInitializingSingleton, DisposableBean {
    // destroy
    @Override
    public void destroy() {
        super.destroy();
    }
}
```

