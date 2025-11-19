# JWT 常见问题与故障排除

## 常见错误及解决方案

### 1. Token验证失败

#### 错误信息
```
JsonWebTokenError: invalid signature
JsonWebTokenError: jwt malformed
TokenExpiredError: jwt expired
```

#### 可能原因和解决方案

**原因1：签名密钥不匹配**
```javascript
// 错误示例
const token = jwt.sign(payload, 'secret1');
const decoded = jwt.verify(token, 'secret2'); // 不同的密钥

// 正确做法
const SECRET_KEY = process.env.JWT_SECRET || 'your-secret-key';
const token = jwt.sign(payload, SECRET_KEY);
const decoded = jwt.verify(token, SECRET_KEY);
```

**原因2：Token格式错误**
```javascript
// 错误：直接传递完整的Authorization头
const authHeader = 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...';
jwt.verify(authHeader, secret); // 错误

// 正确：提取token部分
const token = authHeader.split(' ')[1];
jwt.verify(token, secret); // 正确
```

**原因3：Token已过期**
```javascript
// 检查token是否过期
try {
    const decoded = jwt.verify(token, secret);
    console.log('Token valid:', decoded);
} catch (error) {
    if (error.name === 'TokenExpiredError') {
        console.log('Token expired at:', error.expiredAt);
        // 尝试刷新token或重新登录
    } else {
        console.log('Token invalid:', error.message);
    }
}
```

### 2. CORS问题

#### 错误信息
```
Access to XMLHttpRequest at 'http://api.example.com' from origin 'http://localhost:3000' 
has been blocked by CORS policy
```

#### 解决方案

**Express.js配置**
```javascript
const cors = require('cors');

// 基本CORS配置
app.use(cors({
    origin: ['http://localhost:3000', 'https://yourdomain.com'],
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));

// 或者手动设置
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', 'http://localhost:3000');
    res.header('Access-Control-Allow-Credentials', 'true');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
    res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    
    if (req.method === 'OPTIONS') {
        res.sendStatus(200);
    } else {
        next();
    }
});
```

**Spring Boot配置**
```java
@Configuration
@EnableWebSecurity
public class WebSecurityConfig {
    
    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration configuration = new CorsConfiguration();
        configuration.setAllowedOriginPatterns(Arrays.asList("*"));
        configuration.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "DELETE", "OPTIONS"));
        configuration.setAllowedHeaders(Arrays.asList("*"));
        configuration.setAllowCredentials(true);
        
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);
        return source;
    }
}
```

### 3. 前端Token管理问题

#### 问题：页面刷新后Token丢失

**解决方案1：使用localStorage**
```javascript
class AuthService {
    setToken(token) {
        localStorage.setItem('accessToken', token);
    }
    
    getToken() {
        return localStorage.getItem('accessToken');
    }
    
    removeToken() {
        localStorage.removeItem('accessToken');
    }
    
    isAuthenticated() {
        const token = this.getToken();
        if (!token) return false;
        
        try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            return payload.exp > Date.now() / 1000;
        } catch {
            return false;
        }
    }
}
```

**解决方案2：使用Vuex持久化**
```javascript
// 安装 vuex-persistedstate
npm install vuex-persistedstate

// store/index.js
import { createStore } from 'vuex';
import createPersistedState from 'vuex-persistedstate';

const store = createStore({
    // ... your store config
    plugins: [
        createPersistedState({
            paths: ['auth.accessToken', 'auth.user']
        })
    ]
});
```

#### 问题：Token自动刷新

```javascript
// Axios拦截器实现自动刷新
let isRefreshing = false;
let failedQueue = [];

const processQueue = (error, token = null) => {
    failedQueue.forEach(prom => {
        if (error) {
            prom.reject(error);
        } else {
            prom.resolve(token);
        }
    });
    
    failedQueue = [];
};

api.interceptors.response.use(
    (response) => response,
    async (error) => {
        const originalRequest = error.config;
        
        if (error.response?.status === 401 && !originalRequest._retry) {
            if (isRefreshing) {
                // 如果正在刷新，将请求加入队列
                return new Promise((resolve, reject) => {
                    failedQueue.push({ resolve, reject });
                }).then(token => {
                    originalRequest.headers.Authorization = `Bearer ${token}`;
                    return api(originalRequest);
                }).catch(err => {
                    return Promise.reject(err);
                });
            }
            
            originalRequest._retry = true;
            isRefreshing = true;
            
            try {
                const refreshToken = localStorage.getItem('refreshToken');
                const response = await axios.post('/api/auth/refresh', {
                    refreshToken
                });
                
                const { accessToken } = response.data;
                localStorage.setItem('accessToken', accessToken);
                
                processQueue(null, accessToken);
                
                originalRequest.headers.Authorization = `Bearer ${accessToken}`;
                return api(originalRequest);
                
            } catch (refreshError) {
                processQueue(refreshError, null);
                localStorage.clear();
                window.location.href = '/login';
                return Promise.reject(refreshError);
            } finally {
                isRefreshing = false;
            }
        }
        
        return Promise.reject(error);
    }
);
```

### 4. 性能问题

#### 问题：Token过大导致请求头过大

**解决方案：减少Payload大小**
```javascript
// 错误：在token中存储过多信息
const payload = {
    userId: user.id,
    username: user.username,
    email: user.email,
    profile: user.profile, // 大量数据
    permissions: user.permissions, // 大量权限数据
    settings: user.settings // 用户设置
};

// 正确：只存储必要信息
const payload = {
    userId: user.id,
    username: user.username,
    role: user.role
};

// 其他信息通过API获取
app.get('/api/user/profile', authenticateToken, async (req, res) => {
    const user = await User.findById(req.user.userId)
        .select('profile settings permissions');
    res.json(user);
});
```

#### 问题：频繁的Token验证影响性能

**解决方案：使用缓存**
```javascript
const NodeCache = require('node-cache');
const tokenCache = new NodeCache({ stdTTL: 300 }); // 5分钟缓存

const authenticateToken = async (req, res, next) => {
    const token = req.headers.authorization?.split(' ')[1];
    
    if (!token) {
        return res.status(401).json({ message: 'Token required' });
    }
    
    // 检查缓存
    const cachedUser = tokenCache.get(token);
    if (cachedUser) {
        req.user = cachedUser;
        return next();
    }
    
    try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        
        // 缓存用户信息
        tokenCache.set(token, decoded);
        
        req.user = decoded;
        next();
    } catch (error) {
        res.status(403).json({ message: 'Invalid token' });
    }
};
```

### 5. 安全问题

#### 问题：XSS攻击窃取Token

**解决方案：**
```javascript
// 1. 使用httpOnly cookie存储敏感token
res.cookie('refreshToken', refreshToken, {
    httpOnly: true,
    secure: true,
    sameSite: 'strict'
});

// 2. 内容安全策略(CSP)
app.use((req, res, next) => {
    res.setHeader(
        'Content-Security-Policy',
        "default-src 'self'; script-src 'self' 'unsafe-inline'"
    );
    next();
});

// 3. 输入验证和输出编码
const validator = require('validator');

app.post('/api/comment', (req, res) => {
    const { content } = req.body;
    
    // 验证和清理输入
    if (!validator.isLength(content, { min: 1, max: 1000 })) {
        return res.status(400).json({ message: 'Invalid content length' });
    }
    
    const sanitizedContent = validator.escape(content);
    // 保存sanitizedContent...
});
```

#### 问题：CSRF攻击

**解决方案：**
```javascript
const csrf = require('csurf');

// 使用CSRF保护
const csrfProtection = csrf({ 
    cookie: {
        httpOnly: true,
        secure: true,
        sameSite: 'strict'
    }
});

app.use(csrfProtection);

// 提供CSRF token给前端
app.get('/api/csrf-token', (req, res) => {
    res.json({ csrfToken: req.csrfToken() });
});
```

## 调试技巧

### 1. JWT解码工具

```javascript
// 在浏览器控制台中解码JWT
function decodeJWT(token) {
    try {
        const parts = token.split('.');
        const header = JSON.parse(atob(parts[0]));
        const payload = JSON.parse(atob(parts[1]));
        
        console.log('Header:', header);
        console.log('Payload:', payload);
        console.log('Expires at:', new Date(payload.exp * 1000));
        console.log('Issued at:', new Date(payload.iat * 1000));
        
        return { header, payload };
    } catch (error) {
        console.error('Invalid JWT:', error);
    }
}

// 使用方法
const token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...';
decodeJWT(token);
```

### 2. 日志记录

```javascript
const winston = require('winston');

const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
    ),
    transports: [
        new winston.transports.File({ filename: 'auth.log' }),
        new winston.transports.Console()
    ]
});

const authenticateToken = (req, res, next) => {
    const token = req.headers.authorization?.split(' ')[1];
    
    logger.info('Token verification attempt', {
        ip: req.ip,
        userAgent: req.get('User-Agent'),
        hasToken: !!token
    });
    
    if (!token) {
        logger.warn('Missing token', { ip: req.ip });
        return res.status(401).json({ message: 'Token required' });
    }
    
    try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        logger.info('Token verified successfully', {
            userId: decoded.userId,
            ip: req.ip
        });
        req.user = decoded;
        next();
    } catch (error) {
        logger.error('Token verification failed', {
            error: error.message,
            ip: req.ip
        });
        res.status(403).json({ message: 'Invalid token' });
    }
};
```

### 3. 测试工具

```javascript
// Jest测试示例
const request = require('supertest');
const app = require('../app');
const jwt = require('jsonwebtoken');

describe('JWT Authentication', () => {
    let token;
    
    beforeEach(() => {
        token = jwt.sign(
            { userId: 1, username: 'testuser' },
            process.env.JWT_SECRET,
            { expiresIn: '1h' }
        );
    });
    
    test('should access protected route with valid token', async () => {
        const response = await request(app)
            .get('/api/protected')
            .set('Authorization', `Bearer ${token}`);
            
        expect(response.status).toBe(200);
    });
    
    test('should reject invalid token', async () => {
        const response = await request(app)
            .get('/api/protected')
            .set('Authorization', 'Bearer invalid-token');
            
        expect(response.status).toBe(403);
    });
    
    test('should reject expired token', async () => {
        const expiredToken = jwt.sign(
            { userId: 1, username: 'testuser' },
            process.env.JWT_SECRET,
            { expiresIn: '-1h' } // 已过期
        );
        
        const response = await request(app)
            .get('/api/protected')
            .set('Authorization', `Bearer ${expiredToken}`);
            
        expect(response.status).toBe(403);
    });
});
```

## 监控和告警

### 1. 异常监控

```javascript
const Sentry = require('@sentry/node');

Sentry.init({ dsn: 'YOUR_SENTRY_DSN' });

const authenticateToken = (req, res, next) => {
    try {
        // token验证逻辑...
    } catch (error) {
        // 记录到Sentry
        Sentry.captureException(error, {
            tags: {
                component: 'jwt-auth'
            },
            extra: {
                ip: req.ip,
                userAgent: req.get('User-Agent')
            }
        });
        
        res.status(403).json({ message: 'Invalid token' });
    }
};
```

### 2. 指标收集

```javascript
const prometheus = require('prom-client');

// 创建指标
const authAttempts = new prometheus.Counter({
    name: 'jwt_auth_attempts_total',
    help: 'Total number of JWT authentication attempts',
    labelNames: ['status']
});

const authDuration = new prometheus.Histogram({
    name: 'jwt_auth_duration_seconds',
    help: 'JWT authentication duration in seconds'
});

const authenticateToken = (req, res, next) => {
    const start = Date.now();
    
    try {
        // token验证逻辑...
        authAttempts.inc({ status: 'success' });
        next();
    } catch (error) {
        authAttempts.inc({ status: 'failure' });
        res.status(403).json({ message: 'Invalid token' });
    } finally {
        authDuration.observe((Date.now() - start) / 1000);
    }
};

// 暴露指标端点
app.get('/metrics', (req, res) => {
    res.set('Content-Type', prometheus.register.contentType);
    res.end(prometheus.register.metrics());
});
```

这些解决方案和调试技巧可以帮助您快速定位和解决JWT使用过程中遇到的各种问题。