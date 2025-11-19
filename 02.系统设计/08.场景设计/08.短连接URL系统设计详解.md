---
title: 短连接URL系统设计详解
date: 2025-09-24
categories:
  - Architecture
  - System Design
---

## 1. 系统需求分析

### 1.1 业务需求

**核心功能：**
- 将长URL转换为短URL
- 通过短URL重定向到原始长URL
- 自定义短链接（可选）
- 链接访问统计
- 链接过期时间设置

**非功能性需求：**
- 高可用性：99.9%以上
- 低延迟：<100ms响应时间
- 高并发：支持百万级QPS
- 数据安全：防止链接被恶意猜测
- 扩展性：支持业务快速增长

### 1.2 容量估算

**业务量估算：**
- 日均创建短链接：1000万
- 日均访问次数：1亿次
- 读写比例：100:1
- 数据保存时间：2年
- 短链接长度：7位字符

**存储容量估算：**
```
总短链接数 = 1000万/天 × 365天 × 2年 = 73亿
存储空间 = 73亿 × (8字节ID + 500字节URL + 其他字段) ≈ 4TB
```

**QPS估算：**
```
写QPS = 1000万/(24×3600) ≈ 116 QPS
读QPS = 1亿/(24×3600) ≈ 1157 QPS
峰值QPS = 平均QPS × 10 ≈ 11570 QPS
```

## 2. 第一阶段：单机版本（万级数据）

### 2.1 基础架构

```
[用户] → [Spring Boot应用] → [MySQL数据库]
```

### 2.2 数据库设计

```sql
-- 短链接表
CREATE TABLE short_url (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    short_code VARCHAR(10) NOT NULL UNIQUE COMMENT '短链接编码',
    original_url TEXT NOT NULL COMMENT '原始URL',
    user_id BIGINT COMMENT '用户ID',
    click_count BIGINT DEFAULT 0 COMMENT '点击次数',
    expire_time DATETIME COMMENT '过期时间',
    created_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_short_code (short_code),
    INDEX idx_user_id (user_id),
    INDEX idx_created_time (created_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 访问记录表
CREATE TABLE access_log (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    short_code VARCHAR(10) NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    referer TEXT,
    access_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_short_code (short_code),
    INDEX idx_access_time (access_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 2.3 核心代码实现

#### 2.3.1 实体类定义

```java
@Entity
@Table(name = "short_url")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ShortUrl {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "short_code", unique = true, nullable = false)
    private String shortCode;

    @Column(name = "original_url", nullable = false, columnDefinition = "TEXT")
    private String originalUrl;

    @Column(name = "user_id")
    private Long userId;

    @Column(name = "click_count")
    private Long clickCount = 0L;

    @Column(name = "expire_time")
    private LocalDateTime expireTime;

    @Column(name = "created_time")
    private LocalDateTime createdTime;

    @Column(name = "updated_time")
    private LocalDateTime updatedTime;

    @PrePersist
    public void prePersist() {
        this.createdTime = LocalDateTime.now();
        this.updatedTime = LocalDateTime.now();
    }

    @PreUpdate
    public void preUpdate() {
        this.updatedTime = LocalDateTime.now();
    }
}

@Entity
@Table(name = "access_log")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AccessLog {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "short_code", nullable = false)
    private String shortCode;

    @Column(name = "ip_address")
    private String ipAddress;

    @Column(name = "user_agent", columnDefinition = "TEXT")
    private String userAgent;

    @Column(name = "referer", columnDefinition = "TEXT")
    private String referer;

    @Column(name = "access_time")
    private LocalDateTime accessTime;

    @PrePersist
    public void prePersist() {
        this.accessTime = LocalDateTime.now();
    }
}
```

#### 2.3.2 短码生成器

```java
@Component
public class ShortCodeGenerator {

    private static final String ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    private static final int BASE = ALPHABET.length();
    private static final int SHORT_CODE_LENGTH = 7;
    private final Random random = new Random();

    /**
     * 基于数据库ID生成短码
     */
    public String generateByBase62(Long id) {
        if (id == null || id <= 0) {
            throw new IllegalArgumentException("ID must be positive");
        }

        StringBuilder shortCode = new StringBuilder();
        while (id > 0) {
            shortCode.append(ALPHABET.charAt((int) (id % BASE)));
            id /= BASE;
        }

        // 补足长度
        while (shortCode.length() < SHORT_CODE_LENGTH) {
            shortCode.append(ALPHABET.charAt(0));
        }

        return shortCode.reverse().toString();
    }

    /**
     * 生成随机短码
     */
    public String generateRandom() {
        StringBuilder shortCode = new StringBuilder();
        for (int i = 0; i < SHORT_CODE_LENGTH; i++) {
            shortCode.append(ALPHABET.charAt(random.nextInt(BASE)));
        }
        return shortCode.toString();
    }

    /**
     * 基于URL哈希生成短码
     */
    public String generateByHash(String url) {
        long hash = url.hashCode() & 0x7FFFFFFFL; // 确保为正数
        return generateByBase62(hash);
    }

    /**
     * 解码短码为数字
     */
    public Long decodeToId(String shortCode) {
        long id = 0;
        for (char c : shortCode.toCharArray()) {
            id = id * BASE + ALPHABET.indexOf(c);
        }
        return id;
    }
}
```

#### 2.3.3 服务层实现

```java
@Service
@Transactional
@Slf4j
public class ShortUrlService {

    @Autowired
    private ShortUrlRepository shortUrlRepository;

    @Autowired
    private AccessLogRepository accessLogRepository;

    @Autowired
    private ShortCodeGenerator shortCodeGenerator;

    /**
     * 创建短链接
     */
    public String createShortUrl(CreateShortUrlRequest request) {
        log.info("Creating short URL for: {}", request.getOriginalUrl());

        // 验证URL格式
        validateUrl(request.getOriginalUrl());

        // 检查是否已存在
        if (request.isCheckExisting()) {
            ShortUrl existing = shortUrlRepository.findByOriginalUrlAndUserId(
                request.getOriginalUrl(), request.getUserId());
            if (existing != null && !isExpired(existing)) {
                log.info("Found existing short URL: {}", existing.getShortCode());
                return existing.getShortCode();
            }
        }

        // 创建新的短链接
        ShortUrl shortUrl = ShortUrl.builder()
                .originalUrl(request.getOriginalUrl())
                .userId(request.getUserId())
                .expireTime(request.getExpireTime())
                .build();

        // 生成短码
        String shortCode;
        if (StringUtils.hasText(request.getCustomCode())) {
            // 自定义短码
            shortCode = request.getCustomCode();
            if (shortUrlRepository.existsByShortCode(shortCode)) {
                throw new BusinessException("Custom short code already exists");
            }
        } else {
            // 自动生成短码
            shortCode = generateUniqueShortCode(request.getOriginalUrl());
        }

        shortUrl.setShortCode(shortCode);
        shortUrlRepository.save(shortUrl);

        log.info("Created short URL: {} -> {}", shortCode, request.getOriginalUrl());
        return shortCode;
    }

    /**
     * 获取原始URL并记录访问
     */
    public String getOriginalUrl(String shortCode, HttpServletRequest request) {
        log.debug("Accessing short code: {}", shortCode);

        ShortUrl shortUrl = shortUrlRepository.findByShortCode(shortCode);
        if (shortUrl == null) {
            throw new NotFoundException("Short URL not found");
        }

        // 检查是否过期
        if (isExpired(shortUrl)) {
            throw new ExpiredException("Short URL has expired");
        }

        // 记录访问日志（异步）
        recordAccessAsync(shortCode, request);

        // 更新点击次数
        shortUrlRepository.incrementClickCount(shortCode);

        return shortUrl.getOriginalUrl();
    }

    /**
     * 生成唯一短码
     */
    private String generateUniqueShortCode(String originalUrl) {
        String shortCode;
        int attempts = 0;
        int maxAttempts = 5;

        do {
            if (attempts < 3) {
                // 前3次尝试基于URL哈希生成
                shortCode = shortCodeGenerator.generateByHash(originalUrl + attempts);
            } else {
                // 后续尝试随机生成
                shortCode = shortCodeGenerator.generateRandom();
            }
            attempts++;
        } while (shortUrlRepository.existsByShortCode(shortCode) && attempts < maxAttempts);

        if (attempts >= maxAttempts) {
            throw new BusinessException("Unable to generate unique short code");
        }

        return shortCode;
    }

    /**
     * 异步记录访问日志
     */
    @Async
    public void recordAccessAsync(String shortCode, HttpServletRequest request) {
        try {
            AccessLog accessLog = AccessLog.builder()
                    .shortCode(shortCode)
                    .ipAddress(getClientIpAddress(request))
                    .userAgent(request.getHeader("User-Agent"))
                    .referer(request.getHeader("Referer"))
                    .build();

            accessLogRepository.save(accessLog);
        } catch (Exception e) {
            log.error("Failed to record access log for short code: {}", shortCode, e);
        }
    }

    /**
     * 验证URL格式
     */
    private void validateUrl(String url) {
        if (!StringUtils.hasText(url)) {
            throw new IllegalArgumentException("URL cannot be empty");
        }

        try {
            new URL(url);
        } catch (MalformedURLException e) {
            throw new IllegalArgumentException("Invalid URL format");
        }

        if (url.length() > 2000) {
            throw new IllegalArgumentException("URL too long");
        }
    }

    /**
     * 检查短链接是否过期
     */
    private boolean isExpired(ShortUrl shortUrl) {
        return shortUrl.getExpireTime() != null &&
               shortUrl.getExpireTime().isBefore(LocalDateTime.now());
    }

    /**
     * 获取客户端IP地址
     */
    private String getClientIpAddress(HttpServletRequest request) {
        String[] headers = {
            "X-Forwarded-For",
            "X-Real-IP",
            "Proxy-Client-IP",
            "WL-Proxy-Client-IP",
            "HTTP_CLIENT_IP",
            "HTTP_X_FORWARDED_FOR"
        };

        for (String header : headers) {
            String ip = request.getHeader(header);
            if (StringUtils.hasText(ip) && !"unknown".equalsIgnoreCase(ip)) {
                return ip.split(",")[0].trim();
            }
        }

        return request.getRemoteAddr();
    }
}
```

#### 2.3.4 控制器实现

```java
@RestController
@RequestMapping("/api/v1/shorturl")
@Slf4j
@Validated
public class ShortUrlController {

    @Autowired
    private ShortUrlService shortUrlService;

    /**
     * 创建短链接
     */
    @PostMapping("/create")
    public ApiResponse<CreateShortUrlResponse> createShortUrl(
            @Valid @RequestBody CreateShortUrlRequest request) {

        String shortCode = shortUrlService.createShortUrl(request);

        CreateShortUrlResponse response = CreateShortUrlResponse.builder()
                .shortCode(shortCode)
                .shortUrl("https://short.ly/" + shortCode)
                .originalUrl(request.getOriginalUrl())
                .build();

        return ApiResponse.success(response);
    }

    /**
     * 短链接重定向
     */
    @GetMapping("/{shortCode}")
    public ResponseEntity<Void> redirect(@PathVariable String shortCode,
                                       HttpServletRequest request) {
        try {
            String originalUrl = shortUrlService.getOriginalUrl(shortCode, request);

            return ResponseEntity.status(HttpStatus.MOVED_PERMANENTLY)
                    .location(URI.create(originalUrl))
                    .build();

        } catch (NotFoundException e) {
            return ResponseEntity.notFound().build();
        } catch (ExpiredException e) {
            return ResponseEntity.status(HttpStatus.GONE).build();
        }
    }

    /**
     * 获取短链接信息
     */
    @GetMapping("/{shortCode}/info")
    public ApiResponse<ShortUrlInfo> getShortUrlInfo(@PathVariable String shortCode) {
        ShortUrlInfo info = shortUrlService.getShortUrlInfo(shortCode);
        return ApiResponse.success(info);
    }

    /**
     * 获取访问统计
     */
    @GetMapping("/{shortCode}/stats")
    public ApiResponse<ShortUrlStats> getStats(@PathVariable String shortCode,
                                             @RequestParam(defaultValue = "7") int days) {
        ShortUrlStats stats = shortUrlService.getStats(shortCode, days);
        return ApiResponse.success(stats);
    }
}
```

### 2.4 配置和性能优化

#### 2.4.1 数据库配置

```yaml
spring:
  datasource:
    url: jdbc:mysql://localhost:3306/shorturl?useUnicode=true&characterEncoding=utf8&useSSL=false&serverTimezone=Asia/Shanghai
    username: root
    password: password
    driver-class-name: com.mysql.cj.jdbc.Driver

    # 连接池配置
    hikari:
      minimum-idle: 5
      maximum-pool-size: 20
      auto-commit: true
      idle-timeout: 30000
      pool-name: DateSourceHikariCP
      max-lifetime: 900000
      connection-timeout: 30000

  jpa:
    hibernate:
      ddl-auto: none
    show-sql: false
    properties:
      hibernate:
        dialect: org.hibernate.dialect.MySQL8Dialect
        format_sql: true
        jdbc:
          batch_size: 20
        order_inserts: true
        order_updates: true
```

## 3. 第二阶段：缓存优化（十万级数据）

### 3.1 架构升级

```
[用户] → [Spring Boot应用] → [Redis缓存] → [MySQL数据库]
```

### 3.2 Redis缓存设计

#### 3.2.1 缓存策略

```java
@Service
@Slf4j
public class CachedShortUrlService extends ShortUrlService {

    @Autowired
    private RedisTemplate<String, Object> redisTemplate;

    private static final String SHORT_URL_CACHE_PREFIX = "shorturl:";
    private static final String CLICK_COUNT_CACHE_PREFIX = "clicks:";
    private static final Duration CACHE_TTL = Duration.ofHours(24);

    /**
     * 带缓存的获取原始URL
     */
    @Override
    public String getOriginalUrl(String shortCode, HttpServletRequest request) {
        String cacheKey = SHORT_URL_CACHE_PREFIX + shortCode;

        // 先从缓存获取
        Object cached = redisTemplate.opsForValue().get(cacheKey);
        if (cached != null) {
            if (cached instanceof String) {
                String originalUrl = (String) cached;

                // 异步更新点击次数
                incrementClickCountAsync(shortCode);

                // 记录访问日志
                recordAccessAsync(shortCode, request);

                return originalUrl;
            } else if ("NOT_FOUND".equals(cached)) {
                throw new NotFoundException("Short URL not found");
            }
        }

        // 缓存未命中，从数据库获取
        try {
            String originalUrl = super.getOriginalUrl(shortCode, request);

            // 缓存结果
            redisTemplate.opsForValue().set(cacheKey, originalUrl, CACHE_TTL);

            return originalUrl;

        } catch (NotFoundException e) {
            // 缓存空结果，防止缓存穿透
            redisTemplate.opsForValue().set(cacheKey, "NOT_FOUND", Duration.ofMinutes(5));
            throw e;
        }
    }

    /**
     * 异步更新点击次数
     */
    @Async
    public void incrementClickCountAsync(String shortCode) {
        String clickCountKey = CLICK_COUNT_CACHE_PREFIX + shortCode;

        try {
            // Redis中累计点击次数
            Long newCount = redisTemplate.opsForValue().increment(clickCountKey);

            // 每100次点击或每小时同步一次到数据库
            if (newCount % 100 == 0 || newCount == 1) {
                syncClickCountToDatabase(shortCode, newCount);
            }

        } catch (Exception e) {
            log.error("Failed to increment click count for: {}", shortCode, e);
            // 降级到数据库直接更新
            super.shortUrlRepository.incrementClickCount(shortCode);
        }
    }

    /**
     * 同步点击次数到数据库
     */
    private void syncClickCountToDatabase(String shortCode, Long clickCount) {
        try {
            shortUrlRepository.updateClickCount(shortCode, clickCount);
            log.debug("Synced click count to database: {} -> {}", shortCode, clickCount);
        } catch (Exception e) {
            log.error("Failed to sync click count to database: {}", shortCode, e);
        }
    }

    /**
     * 定时同步点击次数
     */
    @Scheduled(fixedDelay = 3600000) // 每小时执行一次
    public void syncClickCounts() {
        try {
            Set<String> keys = redisTemplate.keys(CLICK_COUNT_CACHE_PREFIX + "*");
            if (keys != null && !keys.isEmpty()) {
                for (String key : keys) {
                    String shortCode = key.replace(CLICK_COUNT_CACHE_PREFIX, "");
                    Long clickCount = (Long) redisTemplate.opsForValue().get(key);
                    if (clickCount != null) {
                        syncClickCountToDatabase(shortCode, clickCount);
                    }
                }
            }
        } catch (Exception e) {
            log.error("Failed to sync click counts", e);
        }
    }
}
```

#### 3.2.2 Redis配置

```yaml
spring:
  redis:
    host: localhost
    port: 6379
    password:
    timeout: 3000ms
    jedis:
      pool:
        max-active: 20
        max-wait: -1ms
        max-idle: 10
        min-idle: 5

# Redis配置类
@Configuration
@EnableCaching
public class RedisConfig {

    @Bean
    public RedisTemplate<String, Object> redisTemplate(RedisConnectionFactory connectionFactory) {
        RedisTemplate<String, Object> template = new RedisTemplate<>();
        template.setConnectionFactory(connectionFactory);

        // 使用Jackson2JsonRedisSerializer来序列化和反序列化Redis的value值
        Jackson2JsonRedisSerializer<Object> serializer = new Jackson2JsonRedisSerializer<>(Object.class);

        ObjectMapper om = new ObjectMapper();
        om.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.ANY);
        om.activateDefaultTyping(LaissezFaireSubTypeValidator.instance,
                                ObjectMapper.DefaultTyping.NON_FINAL);
        serializer.setObjectMapper(om);

        template.setValueSerializer(serializer);
        template.setHashValueSerializer(serializer);
        template.setKeySerializer(new StringRedisSerializer());
        template.setHashKeySerializer(new StringRedisSerializer());

        template.afterPropertiesSet();
        return template;
    }

    @Bean
    public CacheManager cacheManager(RedisConnectionFactory factory) {
        RedisCacheConfiguration config = RedisCacheConfiguration.defaultCacheConfig()
                .entryTtl(Duration.ofHours(1))
                .serializeKeysWith(RedisSerializationContext.SerializationPair
                        .fromSerializer(new StringRedisSerializer()))
                .serializeValuesWith(RedisSerializationContext.SerializationPair
                        .fromSerializer(new Jackson2JsonRedisSerializer<>(Object.class)))
                .disableCachingNullValues();

        return RedisCacheManager.builder(factory)
                .cacheDefaults(config)
                .build();
    }
}
```

### 3.3 布隆过滤器防缓存穿透

```java
@Component
public class BloomFilterService {

    private final RedisTemplate<String, Object> redisTemplate;
    private static final String BLOOM_FILTER_KEY = "shorturl:bloomfilter";
    private static final int EXPECTED_INSERTIONS = 10_000_000;
    private static final double FALSE_POSITIVE_RATE = 0.001;

    public BloomFilterService(RedisTemplate<String, Object> redisTemplate) {
        this.redisTemplate = redisTemplate;
        initBloomFilter();
    }

    /**
     * 初始化布隆过滤器
     */
    private void initBloomFilter() {
        // 使用Redisson实现分布式布隆过滤器
        String script = """
            local bloomFilter = redis.call('BF.RESERVE', KEYS[1], ARGV[1], ARGV[2])
            return bloomFilter
            """;

        try {
            redisTemplate.execute((RedisCallback<Object>) connection -> {
                // 如果过滤器不存在则创建
                if (!connection.exists(BLOOM_FILTER_KEY.getBytes())) {
                    connection.eval(script.getBytes(),
                                  ReturnType.VALUE,
                                  1,
                                  BLOOM_FILTER_KEY.getBytes(),
                                  String.valueOf(FALSE_POSITIVE_RATE).getBytes(),
                                  String.valueOf(EXPECTED_INSERTIONS).getBytes());
                }
                return null;
            });
        } catch (Exception e) {
            log.warn("Failed to initialize Bloom Filter, falling back to Redis SET");
        }
    }

    /**
     * 添加短码到布隆过滤器
     */
    public void addShortCode(String shortCode) {
        try {
            String script = "return redis.call('BF.ADD', KEYS[1], ARGV[1])";
            redisTemplate.execute((RedisCallback<Object>) connection ->
                connection.eval(script.getBytes(),
                              ReturnType.VALUE,
                              1,
                              BLOOM_FILTER_KEY.getBytes(),
                              shortCode.getBytes()));
        } catch (Exception e) {
            log.error("Failed to add to Bloom Filter: {}", shortCode, e);
            // 降级方案：使用Redis SET
            redisTemplate.opsForSet().add("shorturl:codes", shortCode);
        }
    }

    /**
     * 检查短码是否可能存在
     */
    public boolean mightContain(String shortCode) {
        try {
            String script = "return redis.call('BF.EXISTS', KEYS[1], ARGV[1])";
            Long result = redisTemplate.execute((RedisCallback<Long>) connection ->
                (Long) connection.eval(script.getBytes(),
                                     ReturnType.INTEGER,
                                     1,
                                     BLOOM_FILTER_KEY.getBytes(),
                                     shortCode.getBytes()));
            return result != null && result == 1;
        } catch (Exception e) {
            log.error("Failed to check Bloom Filter: {}", shortCode, e);
            // 降级方案：使用Redis SET
            return Boolean.TRUE.equals(redisTemplate.opsForSet().isMember("shorturl:codes", shortCode));
        }
    }
}
```

## 4. 第三阶段：读写分离（百万级数据）

### 4.1 架构升级

```
[用户] → [Nginx负载均衡] → [应用集群] → [Redis集群] → [MySQL主从]
                                              ↓
                                          [从库只读]
```

### 4.2 数据库读写分离

#### 4.2.1 主从配置

```yaml
spring:
  datasource:
    master:
      url: jdbc:mysql://master-db:3306/shorturl
      username: root
      password: password
      driver-class-name: com.mysql.cj.jdbc.Driver
      hikari:
        maximum-pool-size: 20
        minimum-idle: 5

    slave:
      url: jdbc:mysql://slave-db:3306/shorturl
      username: readonly
      password: password
      driver-class-name: com.mysql.cj.jdbc.Driver
      hikari:
        maximum-pool-size: 30
        minimum-idle: 10
```

#### 4.2.2 动态数据源配置

```java
@Configuration
public class DataSourceConfig {

    @Bean
    @ConfigurationProperties("spring.datasource.master")
    public DataSource masterDataSource() {
        return DataSourceBuilder.create().build();
    }

    @Bean
    @ConfigurationProperties("spring.datasource.slave")
    public DataSource slaveDataSource() {
        return DataSourceBuilder.create().build();
    }

    @Bean
    @Primary
    public DataSource dynamicDataSource() {
        DynamicDataSource dynamicDataSource = new DynamicDataSource();

        Map<Object, Object> dataSourceMap = new HashMap<>();
        dataSourceMap.put(DataSourceType.MASTER, masterDataSource());
        dataSourceMap.put(DataSourceType.SLAVE, slaveDataSource());

        dynamicDataSource.setTargetDataSources(dataSourceMap);
        dynamicDataSource.setDefaultTargetDataSource(masterDataSource());

        return dynamicDataSource;
    }
}

public class DynamicDataSource extends AbstractRoutingDataSource {

    @Override
    protected Object determineCurrentLookupKey() {
        return DataSourceContextHolder.getDataSourceType();
    }
}

public class DataSourceContextHolder {

    private static final ThreadLocal<DataSourceType> CONTEXT_HOLDER = new ThreadLocal<>();

    public static void setDataSourceType(DataSourceType type) {
        CONTEXT_HOLDER.set(type);
    }

    public static DataSourceType getDataSourceType() {
        return CONTEXT_HOLDER.get();
    }

    public static void clearDataSourceType() {
        CONTEXT_HOLDER.remove();
    }
}

public enum DataSourceType {
    MASTER, SLAVE
}
```

#### 4.2.3 读写分离注解

```java
@Target({ElementType.METHOD, ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Documented
public @interface DataSource {
    DataSourceType value() default DataSourceType.MASTER;
}

@Aspect
@Component
@Order(1)
public class DataSourceAspect {

    @Pointcut("@annotation(com.shorturl.annotation.DataSource)")
    public void dataSourcePointCut() {}

    @Around("dataSourcePointCut()")
    public Object around(ProceedingJoinPoint point) throws Throwable {
        MethodSignature signature = (MethodSignature) point.getSignature();
        DataSource dataSource = signature.getMethod().getAnnotation(DataSource.class);

        if (dataSource != null) {
            DataSourceContextHolder.setDataSourceType(dataSource.value());
        }

        try {
            return point.proceed();
        } finally {
            DataSourceContextHolder.clearDataSourceType();
        }
    }
}
```

#### 4.2.4 优化后的服务层

```java
@Service
@Transactional
@Slf4j
public class OptimizedShortUrlService {

    @Autowired
    private ShortUrlRepository shortUrlRepository;

    @Autowired
    private BloomFilterService bloomFilterService;

    @Autowired
    private RedisTemplate<String, Object> redisTemplate;

    /**
     * 创建短链接（写操作，使用主库）
     */
    @DataSource(DataSourceType.MASTER)
    public String createShortUrl(CreateShortUrlRequest request) {
        // 创建逻辑保持不变
        String shortCode = doCreateShortUrl(request);

        // 添加到布隆过滤器
        bloomFilterService.addShortCode(shortCode);

        return shortCode;
    }

    /**
     * 获取原始URL（读操作，优先使用缓存和从库）
     */
    @DataSource(DataSourceType.SLAVE)
    @Transactional(readOnly = true)
    public String getOriginalUrl(String shortCode, HttpServletRequest request) {
        // 1. 布隆过滤器预检查
        if (!bloomFilterService.mightContain(shortCode)) {
            throw new NotFoundException("Short URL not found");
        }

        // 2. Redis缓存检查
        String cacheKey = "shorturl:" + shortCode;
        Object cached = redisTemplate.opsForValue().get(cacheKey);
        if (cached != null) {
            if (cached instanceof String) {
                // 异步更新点击次数
                incrementClickCountAsync(shortCode);
                recordAccessAsync(shortCode, request);
                return (String) cached;
            } else if ("NOT_FOUND".equals(cached)) {
                throw new NotFoundException("Short URL not found");
            }
        }

        // 3. 从数据库查询（从库）
        ShortUrl shortUrl = shortUrlRepository.findByShortCodeForRead(shortCode);
        if (shortUrl == null) {
            // 缓存空结果
            redisTemplate.opsForValue().set(cacheKey, "NOT_FOUND", Duration.ofMinutes(5));
            throw new NotFoundException("Short URL not found");
        }

        // 4. 检查过期
        if (isExpired(shortUrl)) {
            throw new ExpiredException("Short URL has expired");
        }

        // 5. 缓存结果
        redisTemplate.opsForValue().set(cacheKey, shortUrl.getOriginalUrl(), Duration.ofHours(24));

        // 6. 异步操作
        incrementClickCountAsync(shortCode);
        recordAccessAsync(shortCode, request);

        return shortUrl.getOriginalUrl();
    }

    /**
     * 获取统计信息（读操作，使用从库）
     */
    @DataSource(DataSourceType.SLAVE)
    @Transactional(readOnly = true)
    public ShortUrlStats getStats(String shortCode, int days) {
        // 从从库读取统计信息
        return buildStatsFromDatabase(shortCode, days);
    }
}
```

### 4.3 应用集群部署

#### 4.3.1 Nginx负载均衡配置

```nginx
upstream shorturl_backend {
    least_conn;
    server 192.168.1.10:8080 weight=3 max_fails=3 fail_timeout=30s;
    server 192.168.1.11:8080 weight=3 max_fails=3 fail_timeout=30s;
    server 192.168.1.12:8080 weight=2 max_fails=3 fail_timeout=30s;

    keepalive 32;
}

server {
    listen 80;
    server_name short.ly;

    location /api/ {
        proxy_pass http://shorturl_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;

        # 启用HTTP/1.1长连接
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }

    # 短链接重定向（无需API前缀）
    location ~ ^/([a-zA-Z0-9]{7})$ {
        proxy_pass http://shorturl_backend/api/v1/shorturl/$1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # 静态资源缓存
    location ~* \.(css|js|jpg|jpeg|png|gif|ico|svg)$ {
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

## 5. 第四阶段：分库分表（千万级数据）

### 5.1 架构升级

```
[CDN] → [Nginx] → [应用集群] → [Redis集群] → [分库分表MySQL]
                      ↓
                 [消息队列MQ]
```

### 5.2 分库分表策略

#### 5.2.1 Sharding-JDBC配置

```yaml
spring:
  shardingsphere:
    datasource:
      names: master0,master1,slave0,slave1

      master0:
        type: com.zaxxer.hikari.HikariDataSource
        driver-class-name: com.mysql.cj.jdbc.Driver
        jdbc-url: jdbc:mysql://192.168.1.10:3306/shorturl_0
        username: root
        password: password

      master1:
        type: com.zaxxer.hikari.HikariDataSource
        driver-class-name: com.mysql.cj.jdbc.Driver
        jdbc-url: jdbc:mysql://192.168.1.11:3306/shorturl_1
        username: root
        password: password

      slave0:
        type: com.zaxxer.hikari.HikariDataSource
        driver-class-name: com.mysql.cj.jdbc.Driver
        jdbc-url: jdbc:mysql://192.168.1.12:3306/shorturl_0
        username: readonly
        password: password

      slave1:
        type: com.zaxxer.hikari.HikariDataSource
        driver-class-name: com.mysql.cj.jdbc.Driver
        jdbc-url: jdbc:mysql://192.168.1.13:3306/shorturl_1
        username: readonly
        password: password

    rules:
      sharding:
        default-database-strategy:
          standard:
            sharding-column: short_code
            sharding-algorithm-name: database_inline

        tables:
          short_url:
            actual-data-nodes: master$->{0..1}.short_url_$->{0..7}
            table-strategy:
              standard:
                sharding-column: short_code
                sharding-algorithm-name: table_inline
            key-generate-strategy:
              column: id
              key-generator-name: snowflake

          access_log:
            actual-data-nodes: master$->{0..1}.access_log_$->{0..7}
            table-strategy:
              standard:
                sharding-column: short_code
                sharding-algorithm-name: table_inline

        sharding-algorithms:
          database_inline:
            type: INLINE
            props:
              algorithm-expression: master$->{short_code.hashCode() % 2}

          table_inline:
            type: INLINE
            props:
              algorithm-expression: short_url_$->{short_code.hashCode() % 8}

        key-generators:
          snowflake:
            type: SNOWFLAKE
            props:
              worker-id: 1

      readwrite-splitting:
        data-sources:
          master0_slave0:
            static-strategy:
              write-data-source-name: master0
              read-data-source-names: slave0
          master1_slave1:
            static-strategy:
              write-data-source-name: master1
              read-data-source-names: slave1

    props:
      sql-show: false
```

#### 5.2.2 分片键优化

```java
@Component
public class ShardingKeyOptimizer {

    /**
     * 根据短码计算分片键
     */
    public int calculateShardingKey(String shortCode) {
        // 使用短码的哈希值进行分片
        return Math.abs(shortCode.hashCode());
    }

    /**
     * 根据用户ID和时间戳生成分布式ID
     */
    public long generateDistributedId(Long userId) {
        // 时间戳(42位) + 机器ID(10位) + 序列号(12位)
        long timestamp = System.currentTimeMillis() - 1609459200000L; // 2021-01-01基准时间
        long machineId = getMachineId();
        long sequence = getSequence();

        return (timestamp << 22) | (machineId << 12) | sequence;
    }

    private long getMachineId() {
        // 简化实现，实际应该从配置获取
        return 1L;
    }

    private long getSequence() {
        // 简化实现，实际应该使用原子计数器
        return System.nanoTime() % 4096;
    }
}
```

### 5.3 消息队列异步处理

#### 5.3.1 RocketMQ配置

```yaml
rocketmq:
  name-server: 192.168.1.20:9876;192.168.1.21:9876
  producer:
    group: shorturl_producer
    send-message-timeout: 3000
    retry-times-when-send-failed: 2
    max-message-size: 4096
  consumer:
    group: shorturl_consumer
    consume-thread-min: 5
    consume-thread-max: 20
```

#### 5.3.2 异步消息处理

```java
@Component
@RocketMQMessageListener(topic = "shorturl-topic", consumerGroup = "shorturl_consumer")
public class ShortUrlMessageListener implements RocketMQListener<String> {

    @Autowired
    private AccessLogService accessLogService;

    @Autowired
    private ClickCountService clickCountService;

    @Override
    public void onMessage(String message) {
        try {
            ShortUrlMessage msg = JSON.parseObject(message, ShortUrlMessage.class);

            switch (msg.getType()) {
                case ACCESS_LOG:
                    handleAccessLog(msg);
                    break;
                case CLICK_COUNT:
                    handleClickCount(msg);
                    break;
                case STATS_UPDATE:
                    handleStatsUpdate(msg);
                    break;
                default:
                    log.warn("Unknown message type: {}", msg.getType());
            }

        } catch (Exception e) {
            log.error("Failed to process message: {}", message, e);
            throw e; // 触发重试
        }
    }

    private void handleAccessLog(ShortUrlMessage message) {
        AccessLogData data = message.getData(AccessLogData.class);
        accessLogService.saveAccessLog(data);
    }

    private void handleClickCount(ShortUrlMessage message) {
        ClickCountData data = message.getData(ClickCountData.class);
        clickCountService.updateClickCount(data.getShortCode(), data.getIncrement());
    }

    private void handleStatsUpdate(ShortUrlMessage message) {
        StatsData data = message.getData(StatsData.class);
        // 更新统计信息
    }
}

@Service
public class MessageProducerService {

    @Autowired
    private RocketMQTemplate rocketMQTemplate;

    public void sendAccessLog(String shortCode, HttpServletRequest request) {
        AccessLogData data = AccessLogData.builder()
                .shortCode(shortCode)
                .ipAddress(getClientIpAddress(request))
                .userAgent(request.getHeader("User-Agent"))
                .referer(request.getHeader("Referer"))
                .timestamp(System.currentTimeMillis())
                .build();

        ShortUrlMessage message = ShortUrlMessage.builder()
                .type(MessageType.ACCESS_LOG)
                .data(data)
                .build();

        rocketMQTemplate.asyncSend("shorturl-topic", JSON.toJSONString(message),
                                 new SendCallback() {
            @Override
            public void onSuccess(SendResult sendResult) {
                log.debug("Access log message sent successfully: {}", shortCode);
            }

            @Override
            public void onException(Throwable e) {
                log.error("Failed to send access log message: {}", shortCode, e);
            }
        });
    }
}
```

## 6. 第五阶段：微服务架构（亿级数据）

### 6.1 微服务拆分

```
[网关Gateway] → [短链生成服务] → [数据库集群]
              → [短链解析服务] → [缓存集群]
              → [统计服务]     → [消息队列]
              → [用户服务]     → [文件存储]
```

### 6.2 服务划分

#### 6.2.1 短链生成服务

```java
@RestController
@RequestMapping("/api/v1/generate")
public class ShortUrlGenerateController {

    @Autowired
    private ShortUrlGenerateService generateService;

    @PostMapping
    public ApiResponse<GenerateResponse> generate(@Valid @RequestBody GenerateRequest request) {
        GenerateResponse response = generateService.generateShortUrl(request);
        return ApiResponse.success(response);
    }
}

@Service
public class ShortUrlGenerateService {

    @Autowired
    private ShortCodeGenerator shortCodeGenerator;

    @Autowired
    private ShortUrlRepository shortUrlRepository;

    @Autowired
    private DistributedLockService lockService;

    public GenerateResponse generateShortUrl(GenerateRequest request) {
        String lockKey = "generate:" + request.getOriginalUrl().hashCode();

        return lockService.executeWithLock(lockKey, Duration.ofSeconds(10), () -> {
            // 检查是否已存在
            if (request.isCheckExisting()) {
                ShortUrl existing = shortUrlRepository.findByOriginalUrl(request.getOriginalUrl());
                if (existing != null && !isExpired(existing)) {
                    return buildResponse(existing);
                }
            }

            // 生成新的短码
            String shortCode = generateUniqueShortCode(request.getOriginalUrl());

            // 保存到数据库
            ShortUrl shortUrl = ShortUrl.builder()
                    .shortCode(shortCode)
                    .originalUrl(request.getOriginalUrl())
                    .userId(request.getUserId())
                    .expireTime(request.getExpireTime())
                    .build();

            shortUrlRepository.save(shortUrl);

            // 异步更新缓存和布隆过滤器
            updateCacheAsync(shortUrl);

            return buildResponse(shortUrl);
        });
    }
}
```

#### 6.2.2 短链解析服务

```java
@RestController
@RequestMapping("/api/v1/resolve")
public class ShortUrlResolveController {

    @Autowired
    private ShortUrlResolveService resolveService;

    @GetMapping("/{shortCode}")
    public ResponseEntity<Void> resolve(@PathVariable String shortCode,
                                      HttpServletRequest request) {
        ResolveResult result = resolveService.resolveShortUrl(shortCode, request);

        return ResponseEntity.status(HttpStatus.MOVED_PERMANENTLY)
                .location(URI.create(result.getOriginalUrl()))
                .build();
    }
}

@Service
public class ShortUrlResolveService {

    @Autowired
    private MultiLevelCacheService cacheService;

    @Autowired
    private MessageProducerService messageProducer;

    public ResolveResult resolveShortUrl(String shortCode, HttpServletRequest request) {
        // 多级缓存查询
        String originalUrl = cacheService.getOriginalUrl(shortCode);

        if (originalUrl != null) {
            // 异步记录访问日志
            messageProducer.sendAccessLog(shortCode, request);

            // 异步更新点击次数
            messageProducer.sendClickCountIncrement(shortCode);

            return ResolveResult.success(originalUrl);
        }

        throw new NotFoundException("Short URL not found");
    }
}
```

### 6.3 多级缓存架构

#### 6.3.1 本地缓存 + Redis + 数据库

```java
@Service
public class MultiLevelCacheService {

    @Autowired
    private RedisTemplate<String, Object> redisTemplate;

    private final Cache<String, String> localCache;
    private final Cache<String, String> negativeCache; // 负缓存

    public MultiLevelCacheService() {
        this.localCache = Caffeine.newBuilder()
                .maximumSize(100_000)
                .expireAfterWrite(Duration.ofMinutes(10))
                .recordStats()
                .build();

        this.negativeCache = Caffeine.newBuilder()
                .maximumSize(10_000)
                .expireAfterWrite(Duration.ofMinutes(1))
                .build();
    }

    public String getOriginalUrl(String shortCode) {
        // L1: 本地缓存
        String cached = localCache.getIfPresent(shortCode);
        if (cached != null) {
            return cached;
        }

        // 检查负缓存
        if (negativeCache.getIfPresent(shortCode) != null) {
            return null;
        }

        // L2: Redis缓存
        String redisKey = "shorturl:" + shortCode;
        Object redisValue = redisTemplate.opsForValue().get(redisKey);

        if (redisValue != null) {
            if (redisValue instanceof String) {
                String originalUrl = (String) redisValue;
                localCache.put(shortCode, originalUrl);
                return originalUrl;
            } else if ("NOT_FOUND".equals(redisValue)) {
                negativeCache.put(shortCode, "");
                return null;
            }
        }

        // L3: 数据库查询
        ShortUrl shortUrl = shortUrlRepository.findByShortCode(shortCode);

        if (shortUrl != null && !isExpired(shortUrl)) {
            String originalUrl = shortUrl.getOriginalUrl();

            // 更新各级缓存
            localCache.put(shortCode, originalUrl);
            redisTemplate.opsForValue().set(redisKey, originalUrl, Duration.ofHours(24));

            return originalUrl;
        } else {
            // 缓存空结果
            negativeCache.put(shortCode, "");
            redisTemplate.opsForValue().set(redisKey, "NOT_FOUND", Duration.ofMinutes(5));

            return null;
        }
    }

    @EventListener
    public void handleCacheInvalidation(CacheInvalidationEvent event) {
        String shortCode = event.getShortCode();

        // 清除本地缓存
        localCache.invalidate(shortCode);
        negativeCache.invalidate(shortCode);

        // 清除Redis缓存
        redisTemplate.delete("shorturl:" + shortCode);
    }
}
```

### 6.4 分布式ID生成

#### 6.4.1 Snowflake算法实现

```java
@Component
public class SnowflakeIdGenerator {

    private final long twepoch = 1609459200000L; // 2021-01-01 基准时间
    private final long workerIdBits = 5L;
    private final long datacenterIdBits = 5L;
    private final long maxWorkerId = -1L ^ (-1L << workerIdBits);
    private final long maxDatacenterId = -1L ^ (-1L << datacenterIdBits);
    private final long sequenceBits = 12L;
    private final long workerIdShift = sequenceBits;
    private final long datacenterIdShift = sequenceBits + workerIdBits;
    private final long timestampLeftShift = sequenceBits + workerIdBits + datacenterIdBits;
    private final long sequenceMask = -1L ^ (-1L << sequenceBits);

    private long workerId;
    private long datacenterId;
    private long sequence = 0L;
    private long lastTimestamp = -1L;

    public SnowflakeIdGenerator(@Value("${snowflake.worker-id:1}") long workerId,
                               @Value("${snowflake.datacenter-id:1}") long datacenterId) {
        if (workerId > maxWorkerId || workerId < 0) {
            throw new IllegalArgumentException("worker Id can't be greater than " + maxWorkerId + " or less than 0");
        }
        if (datacenterId > maxDatacenterId || datacenterId < 0) {
            throw new IllegalArgumentException("datacenter Id can't be greater than " + maxDatacenterId + " or less than 0");
        }

        this.workerId = workerId;
        this.datacenterId = datacenterId;
    }

    public synchronized long nextId() {
        long timestamp = timeGen();

        if (timestamp < lastTimestamp) {
            throw new RuntimeException("Clock moved backwards. Refusing to generate id for " +
                                     (lastTimestamp - timestamp) + " milliseconds");
        }

        if (lastTimestamp == timestamp) {
            sequence = (sequence + 1) & sequenceMask;
            if (sequence == 0) {
                timestamp = tilNextMillis(lastTimestamp);
            }
        } else {
            sequence = 0L;
        }

        lastTimestamp = timestamp;

        return ((timestamp - twepoch) << timestampLeftShift) |
               (datacenterId << datacenterIdShift) |
               (workerId << workerIdShift) |
               sequence;
    }

    protected long tilNextMillis(long lastTimestamp) {
        long timestamp = timeGen();
        while (timestamp <= lastTimestamp) {
            timestamp = timeGen();
        }
        return timestamp;
    }

    protected long timeGen() {
        return System.currentTimeMillis();
    }
}
```

### 6.5 服务治理

#### 6.5.1 Spring Cloud Gateway配置

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: shorturl-generate
          uri: lb://shorturl-generate-service
          predicates:
            - Path=/api/v1/generate/**
          filters:
            - name: RequestRateLimiter
              args:
                redis-rate-limiter.replenishRate: 1000
                redis-rate-limiter.burstCapacity: 2000
                key-resolver: "#{@ipKeyResolver}"
            - name: CircuitBreaker
              args:
                name: generate-circuit-breaker
                fallbackUri: forward:/fallback/generate

        - id: shorturl-resolve
          uri: lb://shorturl-resolve-service
          predicates:
            - Path=/api/v1/resolve/**,/{shortCode}
          filters:
            - name: RequestRateLimiter
              args:
                redis-rate-limiter.replenishRate: 10000
                redis-rate-limiter.burstCapacity: 20000
            - name: CircuitBreaker
              args:
                name: resolve-circuit-breaker
                fallbackUri: forward:/fallback/resolve

  redis:
    host: redis-cluster
    port: 6379
    cluster:
      nodes:
        - redis-1:6379
        - redis-2:6379
        - redis-3:6379

resilience4j:
  circuitbreaker:
    instances:
      generate-circuit-breaker:
        failure-rate-threshold: 50
        wait-duration-in-open-state: 30s
        sliding-window-size: 10
        minimum-number-of-calls: 5
      resolve-circuit-breaker:
        failure-rate-threshold: 60
        wait-duration-in-open-state: 20s
        sliding-window-size: 20
        minimum-number-of-calls: 10
```

## 7. 第六阶段：全球化部署（十亿级数据）

### 7.1 多地域部署架构

```
[全球CDN] → [各地域Gateway] → [本地服务集群] → [本地存储] → [全局数据同步]
```

### 7.2 地理位置路由

#### 7.2.1 GeoDNS配置

```yaml
# DNS配置示例
short.ly:
  - type: A
    value: 1.2.3.4  # 美国东部
    geo:
      continent: NA
      country: US

  - type: A
    value: 5.6.7.8  # 欧洲
    geo:
      continent: EU

  - type: A
    value: 9.10.11.12 # 亚太
    geo:
      continent: AS
```

#### 7.2.2 一致性哈希分片

```java
@Component
public class GlobalShardingStrategy {

    private final ConsistentHash<DataCenter> dataCenterHash;
    private final Map<String, DataCenter> dataCenters;

    public GlobalShardingStrategy() {
        this.dataCenters = initDataCenters();
        this.dataCenterHash = new ConsistentHash<>(dataCenters.values(), 100);
    }

    private Map<String, DataCenter> initDataCenters() {
        Map<String, DataCenter> centers = new HashMap<>();
        centers.put("us-east", new DataCenter("us-east", "美国东部", 40.7128, -74.0060));
        centers.put("eu-west", new DataCenter("eu-west", "欧洲西部", 51.5074, -0.1278));
        centers.put("ap-southeast", new DataCenter("ap-southeast", "亚太东南", 1.3521, 103.8198));
        return centers;
    }

    /**
     * 根据短码选择数据中心
     */
    public DataCenter selectDataCenter(String shortCode) {
        return dataCenterHash.get(shortCode);
    }

    /**
     * 根据用户位置选择最近的数据中心
     */
    public DataCenter selectNearestDataCenter(double latitude, double longitude) {
        return dataCenters.values().stream()
                .min(Comparator.comparingDouble(dc ->
                    calculateDistance(latitude, longitude, dc.getLatitude(), dc.getLongitude())))
                .orElse(dataCenters.get("us-east"));
    }

    private double calculateDistance(double lat1, double lon1, double lat2, double lon2) {
        // Haversine公式计算地球表面两点间距离
        double R = 6371; // 地球半径
        double dLat = Math.toRadians(lat2 - lat1);
        double dLon = Math.toRadians(lon2 - lon1);
        double a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                   Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) *
                   Math.sin(dLon/2) * Math.sin(dLon/2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c;
    }
}
```

### 7.3 数据一致性方案

#### 7.3.1 最终一致性同步

```java
@Service
public class DataSyncService {

    @Autowired
    private MessageProducerService messageProducer;

    @Autowired
    private GlobalDataCenterRegistry dataCenterRegistry;

    /**
     * 同步新创建的短链接到其他数据中心
     */
    @Async
    public void syncShortUrlToGlobal(ShortUrl shortUrl) {
        SyncMessage syncMessage = SyncMessage.builder()
                .type(SyncType.SHORT_URL_CREATE)
                .data(shortUrl)
                .sourceDataCenter(getCurrentDataCenter())
                .timestamp(System.currentTimeMillis())
                .build();

        // 发送到全局同步队列
        messageProducer.sendToGlobalSync(syncMessage);
    }

    /**
     * 处理来自其他数据中心的同步消息
     */
    @RocketMQMessageListener(topic = "global-sync", consumerGroup = "global_sync_consumer")
    public void handleGlobalSync(SyncMessage message) {
        if (message.getSourceDataCenter().equals(getCurrentDataCenter())) {
            return; // 忽略自己的消息
        }

        try {
            switch (message.getType()) {
                case SHORT_URL_CREATE:
                    handleShortUrlSync(message);
                    break;
                case CLICK_COUNT_UPDATE:
                    handleClickCountSync(message);
                    break;
                default:
                    log.warn("Unknown sync message type: {}", message.getType());
            }
        } catch (Exception e) {
            log.error("Failed to handle global sync message", e);
            // 可以考虑重试或者记录到死信队列
        }
    }

    private void handleShortUrlSync(SyncMessage message) {
        ShortUrl shortUrl = message.getData(ShortUrl.class);

        // 检查本地是否已存在（避免重复同步）
        if (!shortUrlRepository.existsByShortCode(shortUrl.getShortCode())) {
            shortUrl.setId(null); // 清除原ID，让数据库重新生成
            shortUrlRepository.save(shortUrl);

            // 更新本地缓存
            cacheService.put(shortUrl.getShortCode(), shortUrl.getOriginalUrl());

            log.info("Synced short URL from {}: {}",
                    message.getSourceDataCenter(), shortUrl.getShortCode());
        }
    }
}
```

## 8. 监控和运维

### 8.1 全链路监控

#### 8.1.1 Metrics收集

```java
@Component
public class ShortUrlMetrics {

    private final MeterRegistry meterRegistry;
    private final Timer createTimer;
    private final Timer resolveTimer;
    private final Counter createCounter;
    private final Counter resolveCounter;

    public ShortUrlMetrics(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
        this.createTimer = Timer.builder("shorturl.create.duration")
                .description("Short URL creation duration")
                .register(meterRegistry);
        this.resolveTimer = Timer.builder("shorturl.resolve.duration")
                .description("Short URL resolution duration")
                .register(meterRegistry);
        this.createCounter = Counter.builder("shorturl.create.total")
                .description("Total short URL creations")
                .register(meterRegistry);
        this.resolveCounter = Counter.builder("shorturl.resolve.total")
                .description("Total short URL resolutions")
                .register(meterRegistry);
    }

    public void recordCreate(Duration duration, boolean success) {
        createTimer.record(duration);
        createCounter.increment(
            Tags.of(
                Tag.of("result", success ? "success" : "failure")
            )
        );
    }

    public void recordResolve(Duration duration, boolean success, boolean cached) {
        resolveTimer.record(duration);
        resolveCounter.increment(
            Tags.of(
                Tag.of("result", success ? "success" : "failure"),
                Tag.of("source", cached ? "cache" : "database")
            )
        );
    }

    @EventListener
    public void handleShortUrlCreated(ShortUrlCreatedEvent event) {
        Gauge.builder("shorturl.total.count")
                .description("Total number of short URLs")
                .register(meterRegistry, this, ShortUrlMetrics::getTotalCount);
    }

    private Number getTotalCount(ShortUrlMetrics metrics) {
        // 从数据库或缓存获取总数
        return shortUrlRepository.count();
    }
}
```

#### 8.1.2 分布式追踪

```java
@Component
public class TracingConfiguration {

    @Bean
    public Sender sender() {
        return OkHttpSender.create("http://jaeger:14268/api/traces");
    }

    @Bean
    public AsyncReporter<Span> spanReporter() {
        return AsyncReporter.create(sender());
    }

    @Bean
    public Tracing tracing() {
        return Tracing.newBuilder()
                .localServiceName("shorturl-service")
                .spanReporter(spanReporter())
                .sampler(Sampler.create(1.0f)) // 100%采样
                .build();
    }
}

@Aspect
@Component
public class TracingAspect {

    private final Tracing tracing;

    public TracingAspect(Tracing tracing) {
        this.tracing = tracing;
    }

    @Around("@annotation(Traced)")
    public Object traceMethod(ProceedingJoinPoint joinPoint) throws Throwable {
        Tracer tracer = tracing.tracer();
        Span span = tracer.nextSpan()
                .name(joinPoint.getSignature().getName())
                .tag("class", joinPoint.getTarget().getClass().getSimpleName())
                .start();

        try (Tracer.SpanInScope ws = tracer.withSpanInScope(span)) {
            return joinPoint.proceed();
        } catch (Exception e) {
            span.tag("error", e.getMessage());
            throw e;
        } finally {
            span.end();
        }
    }
}
```

### 8.2 告警配置

#### 8.2.1 Prometheus告警规则

```yaml
groups:
  - name: shorturl-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(shorturl_resolve_total{result="failure"}[5m]) / rate(shorturl_resolve_total[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in short URL resolution"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: DatabaseConnectionHigh
        expr: hikaricp_connections_active / hikaricp_connections_max > 0.8
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Database connection pool usage high"

      - alert: RedisConnectionFailed
        expr: increase(redis_connection_pool_created_connections_total[5m]) > 100
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis connection failures detected"

      - alert: SlowResponse
        expr: histogram_quantile(0.95, rate(shorturl_resolve_duration_seconds_bucket[5m])) > 0.5
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "95th percentile response time is high"
```

## 9. 性能优化总结

### 9.1 各阶段性能对比

| 阶段 | 数据量 | QPS | 响应时间 | 可用性 | 架构特点 |
|------|--------|-----|----------|--------|----------|
| 单机版 | 万级 | 100 | 50ms | 95% | 单应用+MySQL |
| 缓存优化 | 十万级 | 1K | 20ms | 99% | +Redis缓存 |
| 读写分离 | 百万级 | 10K | 30ms | 99.5% | +主从分离 |
| 分库分表 | 千万级 | 50K | 40ms | 99.9% | +分片+MQ |
| 微服务 | 亿级 | 100K | 35ms | 99.95% | +服务拆分 |
| 全球化 | 十亿级 | 500K | 25ms | 99.99% | +多地域 |

### 9.2 核心优化技术

1. **缓存优化**
   - 多级缓存：本地缓存 + Redis + 数据库
   - 缓存预热和更新策略
   - 布隆过滤器防穿透

2. **数据库优化**
   - 读写分离
   - 分库分表
   - 索引优化
   - 连接池调优

3. **应用优化**
   - 异步处理
   - 批量操作
   - 连接复用
   - JVM调优

4. **架构优化**
   - 微服务拆分
   - 消息队列解耦
   - CDN加速
   - 负载均衡

## 10. 总结

本文详细介绍了短连接URL系统从0到1的完整演变过程，涵盖了从万级到十亿级数据的架构设计。每个阶段都有明确的技术选型和优化方向：

1. **渐进式架构演进**：从单机到分布式到微服务到全球化
2. **技术栈选型合理**：Java + MySQL + Redis + 消息队列
3. **性能持续优化**：缓存、数据库、应用、架构四个维度
4. **可运维性强**：完善的监控、告警、追踪体系

这种设计方案具有很好的扩展性和实用性，可以根据实际业务需求选择合适的阶段进行实施。