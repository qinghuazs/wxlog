# JWT (JSON Web Token) 基础知识

## 什么是JWT？

JWT（JSON Web Token）是一种开放标准（RFC 7519），用于在各方之间安全地传输信息。它是一种紧凑且自包含的方式，用于在各方之间以JSON对象的形式安全地传输信息。

### JWT的特点

- **紧凑性**：由于其较小的尺寸，JWT可以通过URL、POST参数或HTTP头发送
- **自包含**：payload包含了关于用户的所有必需信息，避免了多次查询数据库
- **跨语言支持**：JSON是语言无关的，所以JWT可以在不同的编程语言中使用

## JWT的结构

JWT由三部分组成，用点（.）分隔：

```
header.payload.signature
```

### 1. Header（头部）

头部通常由两部分组成：
- 令牌的类型（即JWT）
- 所使用的签名算法（如HMAC SHA256或RSA）

```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

### 2. Payload（载荷）

载荷包含声明（claims）。声明是关于实体（通常是用户）和其他数据的声明。有三种类型的声明：

#### 注册声明（Registered claims）
- `iss`（issuer）：签发者
- `exp`（expiration time）：过期时间
- `sub`（subject）：主题
- `aud`（audience）：受众
- `nbf`（Not Before）：生效时间
- `iat`（Issued At）：签发时间
- `jti`（JWT ID）：编号

#### 公共声明（Public claims）
可以随意定义，但为了避免冲突，应该在IANA JSON Web Token Registry中定义。

#### 私有声明（Private claims）
用于在同意使用它们的各方之间共享信息。

```json
{
  "sub": "1234567890",
  "name": "John Doe",
  "iat": 1516239022,
  "exp": 1516242622,
  "role": "admin"
}
```

### 3. Signature（签名）

要创建签名部分，您必须获取编码的header、编码的payload、一个secret、header中指定的算法，并对其进行签名。

```javascript
HMACSHA256(
  base64UrlEncode(header) + "." +
  base64UrlEncode(payload),
  secret)
```

## JWT的工作流程

1. **用户登录**：用户使用用户名和密码登录
2. **服务器验证**：服务器验证用户凭据
3. **生成JWT**：如果凭据有效，服务器创建JWT并返回给客户端
4. **客户端存储**：客户端存储JWT（通常在localStorage或cookie中）
5. **后续请求**：客户端在每个请求的Authorization头中发送JWT
6. **服务器验证**：服务器验证JWT并处理请求

## 如何使用JWT

### 1. 在HTTP头中使用

```http
Authorization: Bearer <token>
```

### 2. 在URL参数中使用

```
http://example.com/api/users?token=<token>
```

### 3. 在POST请求体中使用

```json
{
  "token": "<token>",
  "data": "..."
}
```

## 实际应用示例

### Node.js 示例

#### 安装依赖

```bash
npm install jsonwebtoken
```

#### 生成JWT

```javascript
const jwt = require('jsonwebtoken');

// 生成token
const payload = {
  userId: 123,
  username: 'john_doe',
  role: 'user'
};

const secret = 'your-secret-key';
const options = {
  expiresIn: '1h' // 1小时后过期
};

const token = jwt.sign(payload, secret, options);
console.log('Generated token:', token);
```

#### 验证JWT

```javascript
// 验证token
try {
  const decoded = jwt.verify(token, secret);
  console.log('Decoded payload:', decoded);
} catch (error) {
  console.error('Token verification failed:', error.message);
}
```

### Java 示例

#### Maven依赖

```xml
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
```

#### 生成和验证JWT

```java
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.security.Keys;
import java.security.Key;
import java.util.Date;

public class JWTExample {
    private static final Key key = Keys.secretKeyFor(SignatureAlgorithm.HS256);
    
    // 生成JWT
    public static String generateToken(String username, String role) {
        return Jwts.builder()
                .setSubject(username)
                .claim("role", role)
                .setIssuedAt(new Date())
                .setExpiration(new Date(System.currentTimeMillis() + 3600000)) // 1小时
                .signWith(key)
                .compact();
    }
    
    // 验证JWT
    public static Claims validateToken(String token) {
        return Jwts.parserBuilder()
                .setSigningKey(key)
                .build()
                .parseClaimsJws(token)
                .getBody();
    }
}
```

### Python 示例

#### 安装依赖

```bash
pip install PyJWT
```

#### 使用示例

```python
import jwt
import datetime

# 生成JWT
payload = {
    'user_id': 123,
    'username': 'john_doe',
    'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
}

secret = 'your-secret-key'
token = jwt.encode(payload, secret, algorithm='HS256')
print(f'Generated token: {token}')

# 验证JWT
try:
    decoded = jwt.decode(token, secret, algorithms=['HS256'])
    print(f'Decoded payload: {decoded}')
except jwt.ExpiredSignatureError:
    print('Token has expired')
except jwt.InvalidTokenError:
    print('Invalid token')
```

## JWT的优势

1. **无状态**：服务器不需要存储会话信息
2. **可扩展性**：适合分布式系统
3. **跨域支持**：可以在不同域之间使用
4. **移动友好**：适合移动应用
5. **性能**：避免了数据库查询

## JWT的劣势

1. **令牌大小**：比传统的session ID大
2. **无法撤销**：在过期前无法撤销令牌
3. **安全性**：需要妥善保管secret key
4. **存储**：客户端需要安全存储令牌

## 安全最佳实践

1. **使用HTTPS**：始终通过HTTPS传输JWT
2. **设置过期时间**：为JWT设置合理的过期时间
3. **保护Secret**：妥善保管签名密钥
4. **验证声明**：验证iss、aud、exp等声明
5. **使用强算法**：使用安全的签名算法
6. **避免敏感信息**：不要在payload中存储敏感信息

## 常见使用场景

1. **身份认证**：用户登录后的身份验证
2. **信息交换**：在各方之间安全地传输信息
3. **单点登录（SSO）**：在多个应用间共享身份
4. **API授权**：保护REST API端点
5. **微服务通信**：服务间的安全通信

## 总结

JWT是一种强大且灵活的令牌格式，特别适合现代Web应用和API。正确使用JWT可以提供安全、高效的身份验证和授权机制。但是，也需要注意其安全性和最佳实践，以确保应用的安全性。