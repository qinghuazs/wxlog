# JWT 实战应用

## Spring Boot + JWT 完整示例

### 1. 项目依赖

```xml
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-security</artifactId>
    </dependency>
    <dependency>
        <groupId>io.jsonwebtoken</groupId>
        <artifactId>jjwt-api</artifactId>
        <version>0.11.5</version>
    </dependency>
    <dependency>
        <groupId>io.jsonwebtoken</groupId>
        <artifactId>jjwt-impl</artifactId>
        <version>0.11.5</version>
        <scope>runtime</scope>
    </dependency>
    <dependency>
        <groupId>io.jsonwebtoken</groupId>
        <artifactId>jjwt-jackson</artifactId>
        <version>0.11.5</version>
        <scope>runtime</scope>
    </dependency>
</dependencies>
```

### 2. JWT工具类

```java
@Component
public class JwtUtil {
    
    private static final String SECRET = "mySecretKey";
    private static final int JWT_EXPIRATION = 86400000; // 24小时
    
    private Key getSigningKey() {
        byte[] keyBytes = SECRET.getBytes(StandardCharsets.UTF_8);
        return Keys.hmacShaKeyFor(keyBytes);
    }
    
    // 生成JWT令牌
    public String generateToken(String username, List<String> roles) {
        Map<String, Object> claims = new HashMap<>();
        claims.put("roles", roles);
        return createToken(claims, username);
    }
    
    private String createToken(Map<String, Object> claims, String subject) {
        return Jwts.builder()
                .setClaims(claims)
                .setSubject(subject)
                .setIssuedAt(new Date(System.currentTimeMillis()))
                .setExpiration(new Date(System.currentTimeMillis() + JWT_EXPIRATION))
                .signWith(getSigningKey(), SignatureAlgorithm.HS256)
                .compact();
    }
    
    // 验证JWT令牌
    public Boolean validateToken(String token, String username) {
        final String tokenUsername = getUsernameFromToken(token);
        return (tokenUsername.equals(username) && !isTokenExpired(token));
    }
    
    // 从令牌中获取用户名
    public String getUsernameFromToken(String token) {
        return getClaimFromToken(token, Claims::getSubject);
    }
    
    // 从令牌中获取过期时间
    public Date getExpirationDateFromToken(String token) {
        return getClaimFromToken(token, Claims::getExpiration);
    }
    
    // 从令牌中获取角色
    @SuppressWarnings("unchecked")
    public List<String> getRolesFromToken(String token) {
        Claims claims = getAllClaimsFromToken(token);
        return (List<String>) claims.get("roles");
    }
    
    public <T> T getClaimFromToken(String token, Function<Claims, T> claimsResolver) {
        final Claims claims = getAllClaimsFromToken(token);
        return claimsResolver.apply(claims);
    }
    
    private Claims getAllClaimsFromToken(String token) {
        return Jwts.parserBuilder()
                .setSigningKey(getSigningKey())
                .build()
                .parseClaimsJws(token)
                .getBody();
    }
    
    private Boolean isTokenExpired(String token) {
        final Date expiration = getExpirationDateFromToken(token);
        return expiration.before(new Date());
    }
}
```

### 3. JWT过滤器

```java
@Component
public class JwtRequestFilter extends OncePerRequestFilter {
    
    @Autowired
    private UserDetailsService userDetailsService;
    
    @Autowired
    private JwtUtil jwtUtil;
    
    @Override
    protected void doFilterInternal(HttpServletRequest request, 
                                  HttpServletResponse response, 
                                  FilterChain chain) throws ServletException, IOException {
        
        final String requestTokenHeader = request.getHeader("Authorization");
        
        String username = null;
        String jwtToken = null;
        
        // JWT Token格式为 "Bearer token"
        if (requestTokenHeader != null && requestTokenHeader.startsWith("Bearer ")) {
            jwtToken = requestTokenHeader.substring(7);
            try {
                username = jwtUtil.getUsernameFromToken(jwtToken);
            } catch (IllegalArgumentException e) {
                System.out.println("Unable to get JWT Token");
            } catch (ExpiredJwtException e) {
                System.out.println("JWT Token has expired");
            }
        } else {
            logger.warn("JWT Token does not begin with Bearer String");
        }
        
        // 验证令牌
        if (username != null && SecurityContextHolder.getContext().getAuthentication() == null) {
            UserDetails userDetails = this.userDetailsService.loadUserByUsername(username);
            
            if (jwtUtil.validateToken(jwtToken, userDetails.getUsername())) {
                UsernamePasswordAuthenticationToken authToken = 
                    new UsernamePasswordAuthenticationToken(
                        userDetails, null, userDetails.getAuthorities());
                authToken.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                SecurityContextHolder.getContext().setAuthentication(authToken);
            }
        }
        chain.doFilter(request, response);
    }
}
```

### 4. 认证控制器

```java
@RestController
@RequestMapping("/api/auth")
public class AuthController {
    
    @Autowired
    private AuthenticationManager authenticationManager;
    
    @Autowired
    private JwtUtil jwtUtil;
    
    @Autowired
    private UserDetailsService userDetailsService;
    
    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody LoginRequest loginRequest) {
        try {
            authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(
                    loginRequest.getUsername(), 
                    loginRequest.getPassword())
            );
        } catch (BadCredentialsException e) {
            throw new BadCredentialsException("Invalid credentials", e);
        }
        
        final UserDetails userDetails = userDetailsService
            .loadUserByUsername(loginRequest.getUsername());
        
        List<String> roles = userDetails.getAuthorities().stream()
            .map(GrantedAuthority::getAuthority)
            .collect(Collectors.toList());
        
        final String jwt = jwtUtil.generateToken(userDetails.getUsername(), roles);
        
        return ResponseEntity.ok(new JwtResponse(jwt));
    }
    
    @PostMapping("/refresh")
    public ResponseEntity<?> refreshToken(@RequestBody RefreshTokenRequest request) {
        String token = request.getToken();
        String username = jwtUtil.getUsernameFromToken(token);
        
        if (username != null && !jwtUtil.isTokenExpired(token)) {
            UserDetails userDetails = userDetailsService.loadUserByUsername(username);
            List<String> roles = userDetails.getAuthorities().stream()
                .map(GrantedAuthority::getAuthority)
                .collect(Collectors.toList());
            
            String newToken = jwtUtil.generateToken(username, roles);
            return ResponseEntity.ok(new JwtResponse(newToken));
        }
        
        return ResponseEntity.badRequest().body("Invalid token");
    }
}
```

### 5. 请求和响应模型

```java
// 登录请求
public class LoginRequest {
    private String username;
    private String password;
    
    // getters and setters
}

// JWT响应
public class JwtResponse {
    private String token;
    private String type = "Bearer";
    
    public JwtResponse(String token) {
        this.token = token;
    }
    
    // getters and setters
}

// 刷新令牌请求
public class RefreshTokenRequest {
    private String token;
    
    // getters and setters
}
```

## Express.js + JWT 示例

### 1. 安装依赖

```bash
npm install express jsonwebtoken bcryptjs cors helmet
npm install --save-dev nodemon
```

### 2. JWT中间件

```javascript
const jwt = require('jsonwebtoken');
const config = require('../config/config');

// JWT验证中间件
const authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN
    
    if (!token) {
        return res.status(401).json({ message: 'Access token required' });
    }
    
    jwt.verify(token, config.JWT_SECRET, (err, user) => {
        if (err) {
            return res.status(403).json({ message: 'Invalid or expired token' });
        }
        req.user = user;
        next();
    });
};

// 角色验证中间件
const authorizeRoles = (...roles) => {
    return (req, res, next) => {
        if (!req.user) {
            return res.status(401).json({ message: 'User not authenticated' });
        }
        
        if (!roles.includes(req.user.role)) {
            return res.status(403).json({ message: 'Insufficient permissions' });
        }
        
        next();
    };
};

module.exports = { authenticateToken, authorizeRoles };
```

### 3. 认证路由

```javascript
const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const User = require('../models/User');
const config = require('../config/config');
const { authenticateToken } = require('../middleware/auth');

const router = express.Router();

// 用户登录
router.post('/login', async (req, res) => {
    try {
        const { username, password } = req.body;
        
        // 查找用户
        const user = await User.findOne({ username });
        if (!user) {
            return res.status(401).json({ message: 'Invalid credentials' });
        }
        
        // 验证密码
        const isValidPassword = await bcrypt.compare(password, user.password);
        if (!isValidPassword) {
            return res.status(401).json({ message: 'Invalid credentials' });
        }
        
        // 生成JWT
        const payload = {
            userId: user._id,
            username: user.username,
            role: user.role
        };
        
        const accessToken = jwt.sign(payload, config.JWT_SECRET, { 
            expiresIn: config.JWT_EXPIRES_IN 
        });
        
        const refreshToken = jwt.sign(payload, config.JWT_REFRESH_SECRET, { 
            expiresIn: config.JWT_REFRESH_EXPIRES_IN 
        });
        
        // 保存refresh token到数据库
        user.refreshToken = refreshToken;
        await user.save();
        
        res.json({
            message: 'Login successful',
            accessToken,
            refreshToken,
            user: {
                id: user._id,
                username: user.username,
                role: user.role
            }
        });
        
    } catch (error) {
        res.status(500).json({ message: 'Server error', error: error.message });
    }
});

// 刷新令牌
router.post('/refresh', async (req, res) => {
    try {
        const { refreshToken } = req.body;
        
        if (!refreshToken) {
            return res.status(401).json({ message: 'Refresh token required' });
        }
        
        // 验证refresh token
        const decoded = jwt.verify(refreshToken, config.JWT_REFRESH_SECRET);
        
        // 查找用户
        const user = await User.findById(decoded.userId);
        if (!user || user.refreshToken !== refreshToken) {
            return res.status(403).json({ message: 'Invalid refresh token' });
        }
        
        // 生成新的access token
        const payload = {
            userId: user._id,
            username: user.username,
            role: user.role
        };
        
        const newAccessToken = jwt.sign(payload, config.JWT_SECRET, { 
            expiresIn: config.JWT_EXPIRES_IN 
        });
        
        res.json({ accessToken: newAccessToken });
        
    } catch (error) {
        res.status(403).json({ message: 'Invalid refresh token' });
    }
});

// 登出
router.post('/logout', authenticateToken, async (req, res) => {
    try {
        const user = await User.findById(req.user.userId);
        if (user) {
            user.refreshToken = null;
            await user.save();
        }
        
        res.json({ message: 'Logout successful' });
    } catch (error) {
        res.status(500).json({ message: 'Server error' });
    }
});

module.exports = router;
```

### 4. 受保护的路由示例

```javascript
const express = require('express');
const { authenticateToken, authorizeRoles } = require('../middleware/auth');

const router = express.Router();

// 所有用户都可以访问
router.get('/profile', authenticateToken, (req, res) => {
    res.json({
        message: 'Profile data',
        user: req.user
    });
});

// 只有管理员可以访问
router.get('/admin', authenticateToken, authorizeRoles('admin'), (req, res) => {
    res.json({
        message: 'Admin only data',
        user: req.user
    });
});

// 管理员和版主可以访问
router.get('/moderator', 
    authenticateToken, 
    authorizeRoles('admin', 'moderator'), 
    (req, res) => {
        res.json({
            message: 'Moderator data',
            user: req.user
        });
    }
);

module.exports = router;
```

## 前端集成示例

### React + Axios

```javascript
// api.js - Axios配置
import axios from 'axios';

const API_BASE_URL = 'http://localhost:3000/api';

// 创建axios实例
const api = axios.create({
    baseURL: API_BASE_URL,
});

// 请求拦截器 - 添加JWT token
api.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('accessToken');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// 响应拦截器 - 处理token过期
api.interceptors.response.use(
    (response) => {
        return response;
    },
    async (error) => {
        const originalRequest = error.config;
        
        if (error.response?.status === 401 && !originalRequest._retry) {
            originalRequest._retry = true;
            
            try {
                const refreshToken = localStorage.getItem('refreshToken');
                const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
                    refreshToken
                });
                
                const { accessToken } = response.data;
                localStorage.setItem('accessToken', accessToken);
                
                // 重试原始请求
                originalRequest.headers.Authorization = `Bearer ${accessToken}`;
                return api(originalRequest);
                
            } catch (refreshError) {
                // Refresh token也过期了，跳转到登录页
                localStorage.removeItem('accessToken');
                localStorage.removeItem('refreshToken');
                window.location.href = '/login';
                return Promise.reject(refreshError);
            }
        }
        
        return Promise.reject(error);
    }
);

export default api;
```

### Vue.js 示例

```javascript
// store/auth.js - Vuex store
import api from '@/services/api';

const state = {
    user: null,
    accessToken: localStorage.getItem('accessToken'),
    refreshToken: localStorage.getItem('refreshToken'),
    isAuthenticated: false
};

const mutations = {
    SET_USER(state, user) {
        state.user = user;
        state.isAuthenticated = !!user;
    },
    SET_TOKENS(state, { accessToken, refreshToken }) {
        state.accessToken = accessToken;
        state.refreshToken = refreshToken;
        localStorage.setItem('accessToken', accessToken);
        localStorage.setItem('refreshToken', refreshToken);
    },
    CLEAR_AUTH(state) {
        state.user = null;
        state.accessToken = null;
        state.refreshToken = null;
        state.isAuthenticated = false;
        localStorage.removeItem('accessToken');
        localStorage.removeItem('refreshToken');
    }
};

const actions = {
    async login({ commit }, credentials) {
        try {
            const response = await api.post('/auth/login', credentials);
            const { user, accessToken, refreshToken } = response.data;
            
            commit('SET_USER', user);
            commit('SET_TOKENS', { accessToken, refreshToken });
            
            return response.data;
        } catch (error) {
            throw error;
        }
    },
    
    async logout({ commit }) {
        try {
            await api.post('/auth/logout');
        } catch (error) {
            console.error('Logout error:', error);
        } finally {
            commit('CLEAR_AUTH');
        }
    },
    
    async refreshToken({ commit, state }) {
        try {
            const response = await api.post('/auth/refresh', {
                refreshToken: state.refreshToken
            });
            
            const { accessToken } = response.data;
            commit('SET_TOKENS', { 
                accessToken, 
                refreshToken: state.refreshToken 
            });
            
            return accessToken;
        } catch (error) {
            commit('CLEAR_AUTH');
            throw error;
        }
    }
};

export default {
    namespaced: true,
    state,
    mutations,
    actions
};
```

## 安全考虑和最佳实践

### 1. Token存储

```javascript
// 推荐：使用httpOnly cookie存储refresh token
res.cookie('refreshToken', refreshToken, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'strict',
    maxAge: 7 * 24 * 60 * 60 * 1000 // 7天
});

// Access token可以存储在内存中或localStorage
// 内存存储更安全但页面刷新会丢失
class TokenManager {
    constructor() {
        this.accessToken = null;
    }
    
    setAccessToken(token) {
        this.accessToken = token;
    }
    
    getAccessToken() {
        return this.accessToken;
    }
    
    clearTokens() {
        this.accessToken = null;
    }
}
```

### 2. Token黑名单

```javascript
// Redis实现token黑名单
const redis = require('redis');
const client = redis.createClient();

// 添加token到黑名单
const blacklistToken = async (token, expiresIn) => {
    await client.setex(`blacklist_${token}`, expiresIn, 'true');
};

// 检查token是否在黑名单中
const isTokenBlacklisted = async (token) => {
    const result = await client.get(`blacklist_${token}`);
    return result === 'true';
};

// 在JWT验证中间件中使用
const authenticateToken = async (req, res, next) => {
    const token = req.headers.authorization?.split(' ')[1];
    
    if (!token) {
        return res.status(401).json({ message: 'Token required' });
    }
    
    // 检查黑名单
    if (await isTokenBlacklisted(token)) {
        return res.status(401).json({ message: 'Token has been revoked' });
    }
    
    // 验证token...
};
```

### 3. 速率限制

```javascript
const rateLimit = require('express-rate-limit');

// 登录接口限制
const loginLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15分钟
    max: 5, // 最多5次尝试
    message: 'Too many login attempts, please try again later',
    standardHeaders: true,
    legacyHeaders: false,
});

app.use('/api/auth/login', loginLimiter);
```

这些示例展示了JWT在实际项目中的完整应用，包括后端API、前端集成和安全考虑。根据具体需求，可以进一步定制和优化这些实现。