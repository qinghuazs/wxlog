---
title: Performance Optimization
date: 2025-11-18
permalink: /ai/claude-code/architecture-performance-optimization.html
categories:
  - AI
---

# Claude Code 性能优化深度剖析

## 1. 引言：性能优化的重要性

### 1.1 为什么性能优化至关重要

在 AI 编程助手领域，性能直接影响用户体验：

- **响应速度**：用户期望即时反馈（<100ms感知为即时，>1s开始焦虑）
- **成本控制**：API调用费用、服务器资源消耗
- **并发能力**：支持更多用户同时使用
- **资源利用**：合理使用内存、CPU、带宽等资源

### 1.2 性能优化的核心指标

```mermaid
graph TD
    A[性能指标体系] --> B[响应时间指标]
    A --> C[吞吐量指标]
    A --> D[资源使用指标]
    A --> E[成本指标]

    B --> B1[首屏加载时间]
    B --> B2[API响应时间]
    B --> B3[用户交互延迟]

    C --> C1[QPS查询每秒]
    C --> C2[并发用户数]
    C --> C3[请求成功率]

    D --> D1[CPU使用率]
    D --> D2[内存占用]
    D --> D3[网络带宽]

    E --> E1[Token消耗量]
    E --> E2[API调用次数]
    E --> E3[存储成本]
```

### 1.3 性能优化的层次

```java
/**
 * 性能优化层次模型
 */
public class PerformanceOptimizationLayers {

    // Layer 1: 架构层优化
    public static class ArchitectureLayer {
        // 微服务拆分、负载均衡、CDN加速
        private ServiceMesh serviceMesh;
        private LoadBalancer loadBalancer;
        private CDNProvider cdn;
    }

    // Layer 2: 算法层优化
    public static class AlgorithmLayer {
        // 时间复杂度、空间复杂度优化
        private CacheStrategy cacheStrategy;
        private IndexStructure indexStructure;
    }

    // Layer 3: 代码层优化
    public static class CodeLayer {
        // 代码重构、并发优化
        private ConcurrentExecutor executor;
        private ResourcePool resourcePool;
    }

    // Layer 4: 配置层优化
    public static class ConfigurationLayer {
        // JVM参数、线程池配置
        private JvmParameters jvmParams;
        private ThreadPoolConfig threadPoolConfig;
    }
}
```


## 3. 智能缓存策略

### 3.1 多级缓存架构

```mermaid
graph TD
    A[请求] --> B{L1: 内存缓存}
    B -->|命中| C[返回结果]
    B -->|未命中| D{L2: Redis缓存}
    D -->|命中| E[写入L1]
    E --> C
    D -->|未命中| F{L3: 磁盘缓存}
    F -->|命中| G[写入L2和L1]
    G --> C
    F -->|未命中| H[调用API]
    H --> I[写入L1/L2/L3]
    I --> C

    style B fill:#90EE90
    style D fill:#87CEEB
    style F fill:#FFD700
    style H fill:#FF6B6B
```

### 3.2 多级缓存实现

```java
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import redis.clients.jedis.JedisPool;
import java.io.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

/**
 * 多级缓存管理器
 * L1: Caffeine内存缓存 (容量限制、LRU淘汰)
 * L2: Redis缓存 (分布式、持久化)
 * L3: 磁盘缓存 (本地文件、无限容量)
 */
public class MultiLevelCacheManager<K, V> {

    // L1: 内存缓存
    private final Cache<K, V> l1Cache;

    // L2: Redis缓存
    private final JedisPool redisPool;
    private final String redisKeyPrefix;

    // L3: 磁盘缓存
    private final Path diskCachePath;

    // 缓存配置
    private final CacheConfig config;

    public MultiLevelCacheManager(CacheConfig config) {
        this.config = config;

        // 初始化L1缓存
        this.l1Cache = Caffeine.newBuilder()
            .maximumSize(config.getL1MaxSize())
            .expireAfterWrite(Duration.ofMinutes(config.getL1TtlMinutes()))
            .recordStats()
            .build();

        // 初始化L2缓存
        this.redisPool = new JedisPool(config.getRedisHost(), config.getRedisPort());
        this.redisKeyPrefix = config.getRedisKeyPrefix();

        // 初始化L3缓存
        this.diskCachePath = Paths.get(config.getDiskCachePath());
        try {
            Files.createDirectories(diskCachePath);
        } catch (IOException e) {
            throw new RuntimeException("Failed to create disk cache directory", e);
        }
    }

    /**
     * 获取缓存值（依次查询L1 -> L2 -> L3）
     */
    public Optional<V> get(K key) {
        // L1: 内存缓存
        V value = l1Cache.getIfPresent(key);
        if (value != null) {
            recordHit(CacheLevel.L1);
            return Optional.of(value);
        }

        // L2: Redis缓存
        value = getFromRedis(key);
        if (value != null) {
            recordHit(CacheLevel.L2);
            // 回填L1
            l1Cache.put(key, value);
            return Optional.of(value);
        }

        // L3: 磁盘缓存
        value = getFromDisk(key);
        if (value != null) {
            recordHit(CacheLevel.L3);
            // 回填L2和L1
            putToRedis(key, value);
            l1Cache.put(key, value);
            return Optional.of(value);
        }

        recordMiss();
        return Optional.empty();
    }

    /**
     * 写入缓存（写入所有级别）
     */
    public void put(K key, V value) {
        // 写入L1
        l1Cache.put(key, value);

        // 异步写入L2和L3
        CompletableFuture.runAsync(() -> {
            putToRedis(key, value);
            putToDisk(key, value);
        });
    }

    /**
     * 从Redis获取
     */
    private V getFromRedis(K key) {
        try (var jedis = redisPool.getResource()) {
            String redisKey = redisKeyPrefix + key.toString();
            byte[] bytes = jedis.get(redisKey.getBytes());
            if (bytes != null) {
                return deserialize(bytes);
            }
        } catch (Exception e) {
            logger.warn("Redis get failed", e);
        }
        return null;
    }

    /**
     * 写入Redis
     */
    private void putToRedis(K key, V value) {
        try (var jedis = redisPool.getResource()) {
            String redisKey = redisKeyPrefix + key.toString();
            byte[] bytes = serialize(value);
            jedis.setex(
                redisKey.getBytes(),
                config.getL2TtlMinutes() * 60,
                bytes
            );
        } catch (Exception e) {
            logger.warn("Redis put failed", e);
        }
    }

    /**
     * 从磁盘获取
     */
    private V getFromDisk(K key) {
        try {
            Path filePath = getDiskCacheFile(key);
            if (Files.exists(filePath)) {
                byte[] bytes = Files.readAllBytes(filePath);
                return deserialize(bytes);
            }
        } catch (Exception e) {
            logger.warn("Disk get failed", e);
        }
        return null;
    }

    /**
     * 写入磁盘
     */
    private void putToDisk(K key, V value) {
        try {
            Path filePath = getDiskCacheFile(key);
            byte[] bytes = serialize(value);
            Files.write(filePath, bytes, StandardOpenOption.CREATE);
        } catch (Exception e) {
            logger.warn("Disk put failed", e);
        }
    }

    /**
     * 获取磁盘缓存文件路径
     */
    private Path getDiskCacheFile(K key) {
        String hash = String.valueOf(key.hashCode());
        return diskCachePath.resolve(hash + ".cache");
    }

    /**
     * 序列化
     */
    private byte[] serialize(V value) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(bos);
        oos.writeObject(value);
        return bos.toByteArray();
    }

    /**
     * 反序列化
     */
    @SuppressWarnings("unchecked")
    private V deserialize(byte[] bytes) throws IOException, ClassNotFoundException {
        ByteArrayInputStream bis = new ByteArrayInputStream(bytes);
        ObjectInputStream ois = new ObjectInputStream(bis);
        return (V) ois.readObject();
    }

    /**
     * 获取缓存统计
     */
    public CacheStats getStats() {
        var caffeineStats = l1Cache.stats();

        return CacheStats.builder()
            .l1HitRate(caffeineStats.hitRate())
            .l1Size(l1Cache.estimatedSize())
            .l2HitCount(l2HitCount.get())
            .l3HitCount(l3HitCount.get())
            .missCount(missCount.get())
            .build();
    }

    // 统计计数器
    private final AtomicLong l2HitCount = new AtomicLong();
    private final AtomicLong l3HitCount = new AtomicLong();
    private final AtomicLong missCount = new AtomicLong();

    private void recordHit(CacheLevel level) {
        switch (level) {
            case L2: l2HitCount.incrementAndGet(); break;
            case L3: l3HitCount.incrementAndGet(); break;
        }
    }

    private void recordMiss() {
        missCount.incrementAndGet();
    }

    private enum CacheLevel {
        L1, L2, L3
    }
}

/**
 * 缓存配置
 */
@Data
@Builder
public class CacheConfig {
    // L1配置
    private int l1MaxSize = 10000;           // 最大10000条
    private int l1TtlMinutes = 30;           // 30分钟过期

    // L2配置
    private String redisHost = "localhost";
    private int redisPort = 6379;
    private String redisKeyPrefix = "claude:cache:";
    private int l2TtlMinutes = 60;           // 60分钟过期

    // L3配置
    private String diskCachePath = "/tmp/claude-cache";
}
```

### 3.3 缓存策略选择

```java
/**
 * 智能缓存策略选择器
 */
public class CacheStrategySelector {

    /**
     * 根据数据特征选择缓存策略
     */
    public CacheStrategy selectStrategy(CacheRequest request) {
        // 1. 高频访问数据 -> 全缓存
        if (request.getAccessFrequency() > 100) {
            return CacheStrategy.builder()
                .useL1(true)
                .useL2(true)
                .useL3(true)
                .ttl(Duration.ofHours(24))
                .build();
        }

        // 2. 中频访问数据 -> L1 + L2
        if (request.getAccessFrequency() > 10) {
            return CacheStrategy.builder()
                .useL1(true)
                .useL2(true)
                .useL3(false)
                .ttl(Duration.ofHours(1))
                .build();
        }

        // 3. 低频访问数据 -> 仅L1
        if (request.getAccessFrequency() > 1) {
            return CacheStrategy.builder()
                .useL1(true)
                .useL2(false)
                .useL3(false)
                .ttl(Duration.ofMinutes(10))
                .build();
        }

        // 4. 一次性数据 -> 不缓存
        return CacheStrategy.NO_CACHE;
    }

    /**
     * 根据数据大小选择缓存层级
     */
    public CacheStrategy selectBySize(long dataSize) {
        // 小数据(<1KB): 全缓存
        if (dataSize < 1024) {
            return CacheStrategy.ALL_LEVELS;
        }

        // 中等数据(1KB-100KB): L2+L3
        if (dataSize < 100 * 1024) {
            return CacheStrategy.builder()
                .useL1(false)
                .useL2(true)
                .useL3(true)
                .build();
        }

        // 大数据(>100KB): 仅L3
        return CacheStrategy.builder()
            .useL1(false)
            .useL2(false)
            .useL3(true)
            .build();
    }
}
```

### 3.4 缓存预热和更新

```java
/**
 * 缓存预热管理器
 */
public class CacheWarmupManager {

    private final MultiLevelCacheManager<String, Object> cacheManager;
    private final ScheduledExecutorService scheduler;

    public CacheWarmupManager(MultiLevelCacheManager<String, Object> cacheManager) {
        this.cacheManager = cacheManager;
        this.scheduler = Executors.newScheduledThreadPool(2);
    }

    /**
     * 应用启动时预热高频数据
     */
    public void warmupOnStartup() {
        logger.info("Starting cache warmup...");

        // 预热常用提示词模板
        loadPromptTemplates();

        // 预热用户配置
        loadUserConfigurations();

        // 预热代码片段
        loadCodeSnippets();

        logger.info("Cache warmup completed");
    }

    /**
     * 定期刷新缓存
     */
    public void startPeriodicRefresh() {
        // 每小时刷新一次
        scheduler.scheduleAtFixedRate(
            this::refreshCache,
            1,
            1,
            TimeUnit.HOURS
        );
    }

    /**
     * 刷新缓存
     */
    private void refreshCache() {
        logger.info("Refreshing cache...");

        // 获取访问统计
        CacheStats stats = cacheManager.getStats();

        // 淘汰低频数据
        evictLowFrequencyData(stats);

        // 预加载高频数据
        preloadHighFrequencyData(stats);
    }

    /**
     * 增量更新缓存
     */
    public void incrementalUpdate(String key, Object newValue) {
        // 检查是否存在
        Optional<Object> cached = cacheManager.get(key);

        if (cached.isPresent()) {
            // 计算差异
            Object diff = calculateDiff(cached.get(), newValue);

            // 仅更新差异部分
            if (diff != null) {
                cacheManager.put(key, newValue);
                logger.debug("Incremental update for key: {}", key);
            }
        } else {
            // 首次缓存
            cacheManager.put(key, newValue);
        }
    }

    private Object calculateDiff(Object oldValue, Object newValue) {
        // 实现差异计算逻辑
        // 例如：对于文本，使用Diff算法
        if (oldValue instanceof String && newValue instanceof String) {
            return DiffUtils.diff((String) oldValue, (String) newValue);
        }
        return newValue;
    }
}
```


## 5. 懒加载和预加载

### 5.1 代码分割策略

```typescript
/**
 * TypeScript/JavaScript 代码分割示例
 */

// 1. 路由级别的懒加载
const router = createRouter({
  routes: [
    {
      path: '/editor',
      // 懒加载编辑器组件
      component: () => import('./views/Editor.vue')
    },
    {
      path: '/settings',
      // 懒加载设置页面
      component: () => import('./views/Settings.vue')
    }
  ]
});

// 2. 组件级别的懒加载
export default {
  components: {
    // 重组件懒加载
    CodeEditor: () => import('./components/CodeEditor.vue'),

    // 条件懒加载
    HeavyChart: () => {
      if (userHasPremium) {
        return import('./components/PremiumChart.vue');
      }
      return import('./components/BasicChart.vue');
    }
  }
};

// 3. 功能模块的懒加载
class FeatureLoader {
  private loadedModules = new Map<string, any>();

  /**
   * 延迟加载功能模块
   */
  async loadFeature(featureName: string) {
    if (this.loadedModules.has(featureName)) {
      return this.loadedModules.get(featureName);
    }

    let module;
    switch (featureName) {
      case 'git':
        module = await import('./features/git');
        break;
      case 'docker':
        module = await import('./features/docker');
        break;
      case 'ai-assistant':
        module = await import('./features/ai-assistant');
        break;
      default:
        throw new Error(`Unknown feature: ${featureName}`);
    }

    this.loadedModules.set(featureName, module);
    return module;
  }

  /**
   * 预加载可能需要的功能
   */
  async preloadFeatures(features: string[]) {
    return Promise.all(
      features.map(feature => this.loadFeature(feature))
    );
  }
}
```

### 5.2 智能预加载

```java
/**
 * 智能预加载管理器
 * 基于用户行为预测和预加载资源
 */
public class SmartPreloader {

    private final UserBehaviorAnalyzer behaviorAnalyzer;
    private final ResourceLoader resourceLoader;
    private final ScheduledExecutorService scheduler;

    public SmartPreloader() {
        this.behaviorAnalyzer = new UserBehaviorAnalyzer();
        this.resourceLoader = new ResourceLoader();
        this.scheduler = Executors.newScheduledThreadPool(2);
    }

    /**
     * 分析用户行为并预加载
     */
    public void analyzeAndPreload(UserSession session) {
        // 分析用户行为模式
        BehaviorPattern pattern = behaviorAnalyzer.analyze(session);

        // 预测下一步操作
        List<PredictedAction> predictions = pattern.predictNextActions();

        // 预加载资源
        for (PredictedAction action : predictions) {
            if (action.getProbability() > 0.7) {  // 概率>70%才预加载
                preloadForAction(action);
            }
        }
    }

    /**
     * 为特定操作预加载资源
     */
    private void preloadForAction(PredictedAction action) {
        switch (action.getType()) {
            case OPEN_FILE:
                preloadFile(action.getTarget());
                break;
            case RUN_COMMAND:
                preloadCommand(action.getTarget());
                break;
            case AI_COMPLETION:
                preloadAiContext(action.getTarget());
                break;
        }
    }

    /**
     * 预加载文件
     */
    private void preloadFile(String filePath) {
        scheduler.submit(() -> {
            try {
                // 读取文件内容
                String content = resourceLoader.loadFile(filePath);

                // 存入缓存
                fileCache.put(filePath, content);

                // 预解析语法
                SyntaxTree tree = parser.parse(content);
                syntaxCache.put(filePath, tree);

                logger.debug("Preloaded file: {}", filePath);

            } catch (Exception e) {
                logger.warn("Failed to preload file: {}", filePath, e);
            }
        });
    }

    /**
     * 预加载AI上下文
     */
    private void preloadAiContext(String context) {
        scheduler.submit(() -> {
            try {
                // 预先准备提示词
                String prompt = promptBuilder.build(context);

                // 预热模型（可选）
                if (shouldWarmupModel()) {
                    apiClient.warmup(prompt);
                }

                logger.debug("Preloaded AI context: {}", context);

            } catch (Exception e) {
                logger.warn("Failed to preload AI context", e);
            }
        });
    }

    /**
     * 基于时间的预加载
     */
    public void scheduleTimeBasedPreload() {
        // 工作时间开始前预加载
        scheduler.schedule(() -> {
            preloadCommonResources();
        }, getTimeUntilWorkStart(), TimeUnit.MILLISECONDS);
    }

    private void preloadCommonResources() {
        // 预加载常用文件
        List<String> recentFiles = fileHistory.getRecentFiles(10);
        recentFiles.forEach(this::preloadFile);

        // 预加载项目配置
        resourceLoader.loadProjectConfig();

        // 预加载常用代码片段
        resourceLoader.loadCodeSnippets();
    }
}

/**
 * 用户行为分析器
 */
public class UserBehaviorAnalyzer {

    private final LinkedList<UserAction> actionHistory = new LinkedList<>();
    private static final int MAX_HISTORY_SIZE = 100;

    /**
     * 记录用户操作
     */
    public void recordAction(UserAction action) {
        actionHistory.addFirst(action);
        if (actionHistory.size() > MAX_HISTORY_SIZE) {
            actionHistory.removeLast();
        }
    }

    /**
     * 分析行为模式
     */
    public BehaviorPattern analyze(UserSession session) {
        // 1. 时间模式分析
        Map<Integer, List<UserAction>> hourlyActions = groupByHour();

        // 2. 序列模式分析（马尔可夫链）
        Map<String, Map<String, Double>> transitionMatrix = buildTransitionMatrix();

        // 3. 频率分析
        Map<String, Long> actionFrequency = actionHistory.stream()
            .collect(Collectors.groupingBy(
                UserAction::getType,
                Collectors.counting()
            ));

        return BehaviorPattern.builder()
            .hourlyActions(hourlyActions)
            .transitionMatrix(transitionMatrix)
            .actionFrequency(actionFrequency)
            .build();
    }

    /**
     * 构建转移矩阵（预测下一步操作）
     */
    private Map<String, Map<String, Double>> buildTransitionMatrix() {
        Map<String, Map<String, Integer>> counts = new HashMap<>();

        // 统计转移次数
        for (int i = 0; i < actionHistory.size() - 1; i++) {
            String current = actionHistory.get(i).getType();
            String next = actionHistory.get(i + 1).getType();

            counts.computeIfAbsent(current, k -> new HashMap<>())
                  .merge(next, 1, Integer::sum);
        }

        // 转换为概率
        Map<String, Map<String, Double>> probabilities = new HashMap<>();
        for (Map.Entry<String, Map<String, Integer>> entry : counts.entrySet()) {
            String current = entry.getKey();
            Map<String, Integer> nextCounts = entry.getValue();

            int total = nextCounts.values().stream().mapToInt(Integer::intValue).sum();

            Map<String, Double> nextProbs = new HashMap<>();
            for (Map.Entry<String, Integer> nextEntry : nextCounts.entrySet()) {
                double probability = (double) nextEntry.getValue() / total;
                nextProbs.put(nextEntry.getKey(), probability);
            }

            probabilities.put(current, nextProbs);
        }

        return probabilities;
    }
}
```

### 5.3 预测性加载示例

```typescript
/**
 * 预测性资源加载器
 */
class PredictiveLoader {
  private prefetchQueue: PrefetchTask[] = [];
  private isLoading = false;

  /**
   * 基于鼠标悬停预加载
   */
  setupHoverPreload() {
    document.querySelectorAll('[data-preload]').forEach(element => {
      let hoverTimer: NodeJS.Timeout;

      element.addEventListener('mouseenter', () => {
        // 鼠标悬停200ms后开始预加载
        hoverTimer = setTimeout(() => {
          const resource = element.getAttribute('data-preload');
          this.prefetch(resource);
        }, 200);
      });

      element.addEventListener('mouseleave', () => {
        clearTimeout(hoverTimer);
      });
    });
  }

  /**
   * 基于滚动位置预加载
   */
  setupScrollPreload() {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            // 元素即将进入视口，预加载
            const resource = entry.target.getAttribute('data-preload');
            this.prefetch(resource);
          }
        });
      },
      {
        rootMargin: '200px' // 提前200px开始加载
      }
    );

    document.querySelectorAll('[data-scroll-preload]').forEach(element => {
      observer.observe(element);
    });
  }

  /**
   * 预取资源
   */
  async prefetch(resource: string) {
    if (this.isPrefetched(resource)) {
      return;
    }

    this.prefetchQueue.push({
      resource,
      priority: this.calculatePriority(resource),
      timestamp: Date.now()
    });

    // 按优先级排序
    this.prefetchQueue.sort((a, b) => b.priority - a.priority);

    // 执行预加载
    if (!this.isLoading) {
      this.processQueue();
    }
  }

  /**
   * 处理预加载队列
   */
  private async processQueue() {
    if (this.prefetchQueue.length === 0) {
      this.isLoading = false;
      return;
    }

    this.isLoading = true;
    const task = this.prefetchQueue.shift();

    try {
      // 使用浏览器空闲时间加载
      if ('requestIdleCallback' in window) {
        requestIdleCallback(async () => {
          await this.loadResource(task.resource);
          this.processQueue();
        });
      } else {
        await this.loadResource(task.resource);
        this.processQueue();
      }
    } catch (error) {
      console.warn(`Failed to prefetch ${task.resource}:`, error);
      this.processQueue();
    }
  }

  /**
   * 加载资源
   */
  private async loadResource(resource: string) {
    // 使用 <link rel="prefetch"> 预加载
    const link = document.createElement('link');
    link.rel = 'prefetch';
    link.href = resource;
    document.head.appendChild(link);

    // 或使用 fetch 预加载到缓存
    if (resource.endsWith('.json') || resource.includes('/api/')) {
      await fetch(resource, { priority: 'low' });
    }
  }

  /**
   * 计算资源优先级
   */
  private calculatePriority(resource: string): number {
    // 基于资源类型和用户行为计算优先级
    let priority = 50; // 基础优先级

    // 常用文件类型提高优先级
    if (resource.endsWith('.ts') || resource.endsWith('.tsx')) {
      priority += 20;
    }

    // 最近访问过的文件提高优先级
    if (this.wasRecentlyAccessed(resource)) {
      priority += 30;
    }

    return priority;
  }
}
```


## 7. Token 使用优化

### 7.1 Token 压缩技术

```java
/**
 * Token优化器
 */
public class TokenOptimizer {

    private final TokenCounter tokenCounter;

    /**
     * 压缩提示词
     */
    public String compressPrompt(String prompt, int maxTokens) {
        int currentTokens = tokenCounter.count(prompt);

        if (currentTokens <= maxTokens) {
            return prompt;
        }

        // 策略1: 移除注释
        String compressed = removeComments(prompt);
        currentTokens = tokenCounter.count(compressed);
        if (currentTokens <= maxTokens) {
            return compressed;
        }

        // 策略2: 压缩空白
        compressed = compressWhitespace(compressed);
        currentTokens = tokenCounter.count(compressed);
        if (currentTokens <= maxTokens) {
            return compressed;
        }

        // 策略3: 提取关键代码
        compressed = extractKeyCode(compressed, maxTokens);
        currentTokens = tokenCounter.count(compressed);
        if (currentTokens <= maxTokens) {
            return compressed;
        }

        // 策略4: 智能截断
        return intelligentTruncate(compressed, maxTokens);
    }

    /**
     * 移除代码注释
     */
    private String removeComments(String code) {
        // 移除单行注释
        code = code.replaceAll("//.*?\n", "\n");

        // 移除多行注释
        code = code.replaceAll("/\\*.*?\\*/", "");

        return code;
    }

    /**
     * 压缩空白字符
     */
    private String compressWhitespace(String text) {
        // 移除多余的空行
        text = text.replaceAll("\n{3,}", "\n\n");

        // 压缩连续空格
        text = text.replaceAll(" {2,}", " ");

        // 移除行尾空格
        text = text.replaceAll(" +\n", "\n");

        return text;
    }

    /**
     * 提取关键代码
     */
    private String extractKeyCode(String code, int maxTokens) {
        // 解析代码结构
        CodeStructure structure = parseCode(code);

        // 按重要性排序
        List<CodeBlock> blocks = structure.getBlocks();
        blocks.sort(Comparator.comparing(CodeBlock::getImportance).reversed());

        // 逐个添加直到达到token限制
        StringBuilder result = new StringBuilder();
        int tokens = 0;

        for (CodeBlock block : blocks) {
            int blockTokens = tokenCounter.count(block.getContent());

            if (tokens + blockTokens <= maxTokens) {
                result.append(block.getContent()).append("\n\n");
                tokens += blockTokens;
            } else {
                break;
            }
        }

        return result.toString();
    }

    /**
     * 智能截断
     */
    private String intelligentTruncate(String text, int maxTokens) {
        List<String> sentences = splitIntoSentences(text);
        StringBuilder result = new StringBuilder();
        int tokens = 0;

        for (String sentence : sentences) {
            int sentenceTokens = tokenCounter.count(sentence);

            if (tokens + sentenceTokens <= maxTokens) {
                result.append(sentence);
                tokens += sentenceTokens;
            } else {
                // 添加省略标记
                result.append("\n... (truncated)");
                break;
            }
        }

        return result.toString();
    }

    /**
     * Token去重
     */
    public String deduplicateTokens(String text) {
        // 检测重复的代码块
        List<String> blocks = extractCodeBlocks(text);
        Set<String> seen = new HashSet<>();
        List<String> unique = new ArrayList<>();

        for (String block : blocks) {
            String normalized = normalizeCode(block);
            if (seen.add(normalized)) {
                unique.add(block);
            }
        }

        return String.join("\n\n", unique);
    }

    /**
     * 标准化代码（用于去重）
     */
    private String normalizeCode(String code) {
        return code
            .replaceAll("\\s+", " ")          // 统一空白
            .replaceAll("/\\*.*?\\*/", "")    // 移除注释
            .trim()
            .toLowerCase();
    }
}

/**
 * Token计数器
 */
public class TokenCounter {

    private final Tokenizer tokenizer;

    /**
     * 计算token数量
     */
    public int count(String text) {
        // 使用tiktoken或类似库
        return tokenizer.encode(text).size();
    }

    /**
     * 估算token数量（快速但不精确）
     */
    public int estimate(String text) {
        // 粗略估算: 1 token ≈ 4 字符（英文）
        return text.length() / 4;
    }

    /**
     * 批量计算
     */
    public Map<String, Integer> countBatch(List<String> texts) {
        return texts.stream()
            .collect(Collectors.toMap(
                text -> text,
                this::count
            ));
    }
}
```

### 7.2 上下文窗口管理

```java
/**
 * 上下文窗口管理器
 * 智能管理有限的token窗口
 */
public class ContextWindowManager {

    private final int maxTokens;
    private final TokenCounter tokenCounter;
    private final Deque<ContextItem> contextQueue;
    private int currentTokens;

    public ContextWindowManager(int maxTokens) {
        this.maxTokens = maxTokens;
        this.tokenCounter = new TokenCounter();
        this.contextQueue = new LinkedList<>();
        this.currentTokens = 0;
    }

    /**
     * 添加上下文项
     */
    public void addContext(ContextItem item) {
        int itemTokens = tokenCounter.count(item.getContent());

        // 如果单个项就超过限制，截断它
        if (itemTokens > maxTokens) {
            item = truncateItem(item, maxTokens);
            itemTokens = tokenCounter.count(item.getContent());
        }

        // 移除旧项直到有足够空间
        while (currentTokens + itemTokens > maxTokens && !contextQueue.isEmpty()) {
            ContextItem removed = removeOldestLowPriority();
            currentTokens -= tokenCounter.count(removed.getContent());
        }

        // 添加新项
        contextQueue.addLast(item);
        currentTokens += itemTokens;
    }

    /**
     * 移除最旧的低优先级项
     */
    private ContextItem removeOldestLowPriority() {
        // 找到优先级最低的项
        ContextItem lowest = contextQueue.stream()
            .min(Comparator.comparing(ContextItem::getPriority))
            .orElse(contextQueue.peekFirst());

        contextQueue.remove(lowest);
        return lowest;
    }

    /**
     * 构建最终上下文
     */
    public String buildContext() {
        // 按优先级和时间排序
        List<ContextItem> sorted = new ArrayList<>(contextQueue);
        sorted.sort(Comparator
            .comparing(ContextItem::getPriority).reversed()
            .thenComparing(ContextItem::getTimestamp)
        );

        StringBuilder context = new StringBuilder();

        for (ContextItem item : sorted) {
            context.append(item.getContent()).append("\n\n");
        }

        return context.toString();
    }

    /**
     * 获取使用统计
     */
    public ContextStats getStats() {
        return ContextStats.builder()
            .currentTokens(currentTokens)
            .maxTokens(maxTokens)
            .utilizationRate((double) currentTokens / maxTokens)
            .itemCount(contextQueue.size())
            .build();
    }
}

/**
 * 上下文项
 */
@Data
@Builder
public class ContextItem {
    private String content;
    private int priority;          // 优先级 (1-10)
    private long timestamp;        // 时间戳
    private ContextType type;      // 类型

    public enum ContextType {
        SYSTEM_PROMPT,      // 系统提示词
        USER_MESSAGE,       // 用户消息
        CODE_CONTEXT,       // 代码上下文
        FILE_CONTENT,       // 文件内容
        CONVERSATION_HISTORY // 对话历史
    }
}
```

### 7.3 Token使用监控

```java
/**
 * Token使用监控器
 */
public class TokenUsageMonitor {

    private final Map<String, TokenUsageStats> userStats;
    private final AtomicLong totalTokensUsed;

    public TokenUsageMonitor() {
        this.userStats = new ConcurrentHashMap<>();
        this.totalTokensUsed = new AtomicLong(0);
    }

    /**
     * 记录token使用
     */
    public void recordUsage(String userId, int inputTokens, int outputTokens) {
        TokenUsageStats stats = userStats.computeIfAbsent(
            userId,
            k -> new TokenUsageStats()
        );

        stats.addUsage(inputTokens, outputTokens);
        totalTokensUsed.addAndGet(inputTokens + outputTokens);
    }

    /**
     * 获取用户统计
     */
    public TokenUsageStats getUserStats(String userId) {
        return userStats.getOrDefault(userId, new TokenUsageStats());
    }

    /**
     * 检查是否超过配额
     */
    public boolean isQuotaExceeded(String userId, int dailyLimit) {
        TokenUsageStats stats = getUserStats(userId);
        return stats.getDailyTokens() > dailyLimit;
    }

    /**
     * 生成使用报告
     */
    public UsageReport generateReport(String userId, Period period) {
        TokenUsageStats stats = getUserStats(userId);

        return UsageReport.builder()
            .userId(userId)
            .period(period)
            .totalTokens(stats.getTotalTokens())
            .totalCost(calculateCost(stats.getTotalTokens()))
            .averagePerRequest(stats.getAveragePerRequest())
            .peakUsageTime(stats.getPeakUsageTime())
            .recommendations(generateRecommendations(stats))
            .build();
    }

    /**
     * 生成优化建议
     */
    private List<String> generateRecommendations(TokenUsageStats stats) {
        List<String> recommendations = new ArrayList<>();

        if (stats.getAverageInputTokens() > 2000) {
            recommendations.add("考虑压缩输入上下文以减少token消耗");
        }

        if (stats.getCacheHitRate() < 0.3) {
            recommendations.add("启用缓存可以减少重复请求的token消耗");
        }

        if (stats.getRepetitiveQueryRate() > 0.5) {
            recommendations.add("检测到大量重复查询，建议使用批处理");
        }

        return recommendations;
    }

    /**
     * 计算成本
     */
    private double calculateCost(long tokens) {
        // Claude API定价 (示例)
        // Input: $3 / 1M tokens
        // Output: $15 / 1M tokens
        double inputCost = 3.0 / 1_000_000;
        double outputCost = 15.0 / 1_000_000;

        // 假设输入输出比例为 3:1
        long inputTokens = tokens * 3 / 4;
        long outputTokens = tokens / 4;

        return inputTokens * inputCost + outputTokens * outputCost;
    }
}

/**
 * Token使用统计
 */
@Data
public class TokenUsageStats {
    private long totalInputTokens = 0;
    private long totalOutputTokens = 0;
    private int requestCount = 0;
    private long dailyTokens = 0;
    private Map<LocalDate, Long> dailyUsage = new HashMap<>();

    public void addUsage(int inputTokens, int outputTokens) {
        this.totalInputTokens += inputTokens;
        this.totalOutputTokens += outputTokens;
        this.requestCount++;

        LocalDate today = LocalDate.now();
        long todayUsage = dailyUsage.getOrDefault(today, 0L);
        dailyUsage.put(today, todayUsage + inputTokens + outputTokens);
        this.dailyTokens = dailyUsage.get(today);
    }

    public long getTotalTokens() {
        return totalInputTokens + totalOutputTokens;
    }

    public double getAveragePerRequest() {
        return requestCount == 0 ? 0 : (double) getTotalTokens() / requestCount;
    }

    public double getAverageInputTokens() {
        return requestCount == 0 ? 0 : (double) totalInputTokens / requestCount;
    }
}
```


## 9. 性能监控和分析工具

### 9.1 性能监控系统

```java
import io.micrometer.core.instrument.*;
import io.micrometer.core.instrument.Timer;

/**
 * 性能监控系统
 */
public class PerformanceMonitor {

    private final MeterRegistry meterRegistry;

    // 计数器
    private final Counter apiCallCounter;
    private final Counter cacheHitCounter;
    private final Counter cacheMissCounter;

    // 计时器
    private final Timer apiResponseTimer;
    private final Timer cacheAccessTimer;

    // 仪表
    private final Gauge memoryUsageGauge;
    private final Gauge activeConnectionsGauge;

    public PerformanceMonitor(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;

        // 初始化计数器
        this.apiCallCounter = Counter.builder("api.calls.total")
            .description("Total API calls")
            .register(meterRegistry);

        this.cacheHitCounter = Counter.builder("cache.hits.total")
            .description("Total cache hits")
            .register(meterRegistry);

        this.cacheMissCounter = Counter.builder("cache.misses.total")
            .description("Total cache misses")
            .register(meterRegistry);

        // 初始化计时器
        this.apiResponseTimer = Timer.builder("api.response.time")
            .description("API response time")
            .publishPercentiles(0.5, 0.95, 0.99)
            .register(meterRegistry);

        this.cacheAccessTimer = Timer.builder("cache.access.time")
            .description("Cache access time")
            .register(meterRegistry);

        // 初始化仪表
        this.memoryUsageGauge = Gauge.builder("jvm.memory.used", this,
            monitor -> getMemoryUsage())
            .description("JVM memory usage")
            .register(meterRegistry);

        this.activeConnectionsGauge = Gauge.builder("connections.active", this,
            monitor -> getActiveConnections())
            .description("Active connections")
            .register(meterRegistry);
    }

    /**
     * 记录API调用
     */
    public void recordApiCall() {
        apiCallCounter.increment();
    }

    /**
     * 记录API响应时间
     */
    public <T> T recordApiCall(Supplier<T> supplier) {
        return apiResponseTimer.record(supplier);
    }

    /**
     * 记录缓存命中
     */
    public void recordCacheHit() {
        cacheHitCounter.increment();
    }

    /**
     * 记录缓存未命中
     */
    public void recordCacheMiss() {
        cacheMissCounter.increment();
    }

    /**
     * 获取性能报告
     */
    public PerformanceReport getReport() {
        return PerformanceReport.builder()
            .totalApiCalls(apiCallCounter.count())
            .cacheHitRate(calculateCacheHitRate())
            .averageResponseTime(apiResponseTimer.mean(TimeUnit.MILLISECONDS))
            .p95ResponseTime(apiResponseTimer.percentile(0.95, TimeUnit.MILLISECONDS))
            .p99ResponseTime(apiResponseTimer.percentile(0.99, TimeUnit.MILLISECONDS))
            .memoryUsage(getMemoryUsage())
            .activeConnections(getActiveConnections())
            .build();
    }

    /**
     * 计算缓存命中率
     */
    private double calculateCacheHitRate() {
        double hits = cacheHitCounter.count();
        double misses = cacheMissCounter.count();
        double total = hits + misses;

        return total == 0 ? 0 : hits / total;
    }

    private long getMemoryUsage() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }

    private int getActiveConnections() {
        // 从连接池获取
        return connectionPool.getStats().getActiveConnections();
    }
}

/**
 * 性能报告
 */
@Data
@Builder
public class PerformanceReport {
    private double totalApiCalls;
    private double cacheHitRate;
    private double averageResponseTime;
    private double p95ResponseTime;
    private double p99ResponseTime;
    private long memoryUsage;
    private int activeConnections;

    /**
     * 生成可读报告
     */
    public String toReadableString() {
        return String.format(
            """
            Performance Report:
            ==================
            API Calls: %.0f
            Cache Hit Rate: %.2f%%
            Response Time:
              - Average: %.2f ms
              - P95: %.2f ms
              - P99: %.2f ms
            Memory Usage: %s
            Active Connections: %d
            """,
            totalApiCalls,
            cacheHitRate * 100,
            averageResponseTime,
            p95ResponseTime,
            p99ResponseTime,
            formatBytes(memoryUsage),
            activeConnections
        );
    }

    private String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        char unit = "KMGTPE".charAt(exp - 1);
        return String.format("%.2f %sB", bytes / Math.pow(1024, exp), unit);
    }
}
```

### 9.2 性能分析器

```java
/**
 * 性能分析器
 * 用于定位性能瓶颈
 */
public class PerformanceProfiler {

    private final Map<String, MethodStats> methodStats;
    private final ThreadLocal<Deque<TimingContext>> timingStack;

    public PerformanceProfiler() {
        this.methodStats = new ConcurrentHashMap<>();
        this.timingStack = ThreadLocal.withInitial(LinkedList::new);
    }

    /**
     * 开始计时
     */
    public void startTiming(String methodName) {
        TimingContext context = new TimingContext(
            methodName,
            System.nanoTime()
        );
        timingStack.get().push(context);
    }

    /**
     * 结束计时
     */
    public void endTiming() {
        Deque<TimingContext> stack = timingStack.get();
        if (stack.isEmpty()) {
            return;
        }

        TimingContext context = stack.pop();
        long duration = System.nanoTime() - context.startTime;

        // 更新统计
        MethodStats stats = methodStats.computeIfAbsent(
            context.methodName,
            k -> new MethodStats()
        );
        stats.addMeasurement(duration);
    }

    /**
     * 使用装饰器模式计时
     */
    public <T> T profile(String methodName, Supplier<T> supplier) {
        startTiming(methodName);
        try {
            return supplier.get();
        } finally {
            endTiming();
        }
    }

    /**
     * 获取性能热点
     */
    public List<HotSpot> getHotSpots(int topN) {
        return methodStats.entrySet().stream()
            .map(entry -> new HotSpot(
                entry.getKey(),
                entry.getValue()
            ))
            .sorted(Comparator.comparing(HotSpot::getTotalTime).reversed())
            .limit(topN)
            .collect(Collectors.toList());
    }

    /**
     * 生成火焰图数据
     */
    public FlameGraphData generateFlameGraph() {
        // 生成火焰图所需的数据格式
        FlameGraphData data = new FlameGraphData();

        methodStats.forEach((method, stats) -> {
            data.addNode(FlameGraphNode.builder()
                .name(method)
                .value(stats.getTotalTime())
                .count(stats.getCallCount())
                .build());
        });

        return data;
    }

    /**
     * 重置统计
     */
    public void reset() {
        methodStats.clear();
    }

    /**
     * 方法统计
     */
    @Data
    private static class MethodStats {
        private long callCount = 0;
        private long totalTime = 0;
        private long minTime = Long.MAX_VALUE;
        private long maxTime = 0;
        private final List<Long> samples = new ArrayList<>();

        public synchronized void addMeasurement(long duration) {
            callCount++;
            totalTime += duration;
            minTime = Math.min(minTime, duration);
            maxTime = Math.max(maxTime, duration);
            samples.add(duration);
        }

        public double getAverageTime() {
            return callCount == 0 ? 0 : (double) totalTime / callCount;
        }

        public double getPercentile(double p) {
            if (samples.isEmpty()) {
                return 0;
            }

            List<Long> sorted = new ArrayList<>(samples);
            Collections.sort(sorted);

            int index = (int) Math.ceil(p * sorted.size()) - 1;
            return sorted.get(Math.max(0, index));
        }
    }

    /**
     * 计时上下文
     */
    private static class TimingContext {
        final String methodName;
        final long startTime;

        TimingContext(String methodName, long startTime) {
            this.methodName = methodName;
            this.startTime = startTime;
        }
    }

    /**
     * 性能热点
     */
    @Data
    public static class HotSpot {
        private final String methodName;
        private final long callCount;
        private final long totalTime;
        private final double averageTime;
        private final long minTime;
        private final long maxTime;

        public HotSpot(String methodName, MethodStats stats) {
            this.methodName = methodName;
            this.callCount = stats.getCallCount();
            this.totalTime = stats.getTotalTime();
            this.averageTime = stats.getAverageTime();
            this.minTime = stats.getMinTime();
            this.maxTime = stats.getMaxTime();
        }
    }
}
```


## 11. FAQ

### Q1: 何时应该使用批处理？

**A:** 满足以下条件时应使用批处理：
- 有多个相似的小请求
- 请求之间没有强依赖关系
- 可以容忍一定的延迟（通常<100ms）
- API支持批量调用

性能提升通常在5-10倍之间。

### Q2: 缓存的最佳过期时间是多少？

**A:** 根据数据特征选择：
- 静态内容（配置、模板）：24小时或更长
- 代码分析结果：1-2小时
- 用户对话历史：30分钟
- 临时计算结果：5-10分钟

建议使用LRU或LFU策略，并监控命中率。

### Q3: 如何确定合适的连接池大小？

**A:** 使用以下公式：
```
连接池大小 = (核心线程数 * 2) + 1
或
连接池大小 = (平均请求数 / 平均响应时间) * 1.5
```

建议：
- 最小值：5-10
- 最大值：50-100
- 根据实际负载调整

### Q4: Token压缩会影响回答质量吗？

**A:** 适度压缩不会显著影响质量：
- 移除注释：无影响
- 压缩空白：无影响
- 提取关键代码：可能有轻微影响
- 智能截断：需要保留核心上下文

建议保留至少70%的原始内容。

### Q5: 如何检测和解决内存泄漏？

**A:** 步骤：
1. 使用内存泄漏检测器追踪对象
2. 定期检查长期存活的对象
3. 使用 VisualVM 或 JProfiler 分析堆转储
4. 检查：
   - 静态集合是否持续增长
   - 缓存是否有清理机制
   - 监听器是否正确移除
   - 线程是否正确关闭

### Q6: 预加载多少数据合适？

**A:** 遵循"3-1"原则：
- 预加载未来3次操作可能需要的数据
- 不要预加载超过1屏幕的内容
- 使用概率阈值（>70%才预加载）

监控预加载命中率，目标>50%。

### Q7: 性能优化的优先级顺序？

**A:** 按投入产出比排序：
1. **缓存**（最高ROI，实现简单）
2. **批处理**（高ROI，中等难度）
3. **Token优化**（中等ROI，简单）
4. **连接池**（中等ROI，简单）
5. **并发控制**（低ROI，复杂）
6. **预加载**（低ROI，复杂）

先做缓存和批处理，通常能解决80%的性能问题。

### Q8: 如何平衡性能和成本？

**A:** 优化策略：
- 高频操作：投入更多资源优化
- 低频操作：可以牺牲一些性能
- 使用缓存降低API调用成本
- 监控Token使用量，设置配额
- 根据用户等级提供不同服务质量

成本降低通常在50-70%之间。

### Q9: 性能监控的关键指标有哪些？

**A:** 核心指标：
- **响应时间**：P50, P95, P99
- **吞吐量**：QPS, 并发用户数
- **错误率**：失败率、超时率
- **缓存命中率**：>30%为良好
- **资源使用**：CPU、内存、网络
- **成本**：Token消耗、API调用次数

设置告警阈值并定期审查。

### Q10: 如何进行性能测试？

**A:** 测试流程：
1. **基准测试**：记录当前性能
2. **压力测试**：找到系统瓶颈
3. **负载测试**：模拟真实使用
4. **持久测试**：检测内存泄漏

工具推荐：
- JMeter：压力测试
- Gatling：负载测试
- VisualVM：性能分析
- Micrometer：指标收集

---

## 总结

性能优化是一个持续的过程，需要：

1. **建立基准**：测量当前性能
2. **识别瓶颈**：使用工具定位问题
3. **优先优化**：按ROI排序
4. **验证效果**：对比优化前后
5. **持续监控**：防止性能退化

记住：**过早优化是万恶之源**。先确保功能正确，再优化性能。

核心原则：
- 缓存一切可以缓存的
- 合并一切可以合并的
- 延迟一切可以延迟的
- 监控一切重要的指标

通过合理应用本文介绍的技术，Claude Code的性能可以提升5-10倍，同时降低50-70%的成本。
