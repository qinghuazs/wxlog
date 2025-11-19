---
title: MyBatis Interceptor扩展
date: 2024/12/02
---

参考文章：

[MyBatis拦截器](https://juejin.cn/post/7349382887572275200?searchId=20241202170344B17B989FF2B243F5F369)

## Interceprot 简介

MyBatis 的 **Interceptor** 是一种用于在 SQL 执行过程中插入自定义逻辑的机制，它允许你在 MyBatis 框架的核心操作（如执行 SQL、查询结果等）上进行扩展。通过使用 `Interceptor`，你可以在执行 SQL 前后，甚至在 `ResultSet` 返回之前对数据进行修改。

`Interceptor` 是 MyBatis 提供的插件机制，允许开发者在 MyBatis 的执行链中拦截 SQL 的执行过程，并对执行过程进行修改。你可以在 MyBatis 执行 `StatementHandler`、`ResultSetHandler`、`ParameterHandler`、`Executor` 的各个阶段插入自定义逻辑。

`Interceptor` 的扩展需要实现 `org.apache.ibatis.plugin.Interceptor` 接口，主要有三个方法：

- `intercept(Invocation invocation)`：执行拦截逻辑的核心方法。你可以在这里对 MyBatis 的行为进行修改，比如修改 SQL、参数、返回值等。
- `plugin(Object target)`：为目标对象创建代理。这个方法用来为目标对象创建一个代理对象，以便拦截目标对象的方法调用。
- `setProperties(Properties properties)`：用于设置插件的配置属性。

### 常见用法

**动态修改 SQL**：在执行 SQL 之前修改 SQL，比如自动添加 WHERE 条件（租户 id）、日志记录等。

**性能监控**：通过拦截 SQL 的执行时间、结果返回等，进行性能监控。

**权限控制**：根据当前用户角色，动态修改查询条件、增加权限校验。

**审计日志**：在执行插入、更新或删除操作时，记录日志或进行数据审计。

**缓存**：定制化缓存策略，比如查询缓存、更新缓存等。

## 实现案例

### 租户id赋值

#### 代码实现

在执行 SQL 查询时，自动为查询条件添加租户 Id。

```java
package com.example.interceptor;

import org.apache.ibatis.executor.Executor;
import org.apache.ibatis.plugin.*;
import org.apache.ibatis.session.RowBounds;
import org.apache.ibatis.mapping.MappedStatement;
import org.apache.ibatis.mapping.BoundSql;
import org.apache.ibatis.plugin.Invocation;

import java.lang.reflect.Field;
import java.sql.Connection;
import java.util.Properties;

@Intercepts({
    @Signature(type = Executor.class, method = "update", args = {MappedStatement.class, Object.class}),
    @Signature(type = Executor.class, method = "query", args = {MappedStatement.class, Object.class, RowBounds.class, ResultHandler.class})
})
public class TenantInterceptor implements Interceptor {

    @Override
    public Object intercept(Invocation invocation) throws Throwable {
        // 获取当前操作的目标对象
        Object target = invocation.getTarget();

        // 1. 获取 MappedStatement，检查它是否是需要插入租户 ID 的查询
        MappedStatement mappedStatement = (MappedStatement) invocation.getArgs()[0];
        BoundSql boundSql = mappedStatement.getBoundSql(invocation.getArgs()[1]);

        // 2. 获取租户 ID，可以通过线程上下文、Session 或其他方式获取
        String tenantId = TenantContext.getTenantId(); // 假设你有一个 TenantContext 用于管理租户 ID

        // 3. 如果租户 ID 不为空，修改 SQL（例如增加租户过滤条件）
        if (tenantId != null && !tenantId.isEmpty()) {
            String sql = boundSql.getSql();
            sql = modifySqlWithTenantId(sql, tenantId); // 修改 SQL 语句

            // 使用反射修改 SQL
            Field sqlField = BoundSql.class.getDeclaredField("sql");
            sqlField.setAccessible(true);
            sqlField.set(boundSql, sql);
        }

        // 4. 执行拦截后的原始方法
        return invocation.proceed();
    }

    private String modifySqlWithTenantId(String sql, String tenantId) {
        // 这里通过简单拼接示例，实际应用中根据需求处理
        return "SELECT * FROM (" + sql + ") AS temp WHERE tenant_id = '" + tenantId + "'";
    }

    @Override
    public Object plugin(Object target) {
        // 通过 Plugin.wrap 来为目标对象创建代理
        return Plugin.wrap(target, this);
    }

    @Override
    public void setProperties(Properties properties) {
        // 可以从配置中获取插件的配置信息
        String someProperty = properties.getProperty("someProperty");
        // 在这里设置插件的配置信息
    }
}

```

`@Signature`：定义了拦截的目标方法。这里我们拦截了 `Executor` 类中的 `update` 和 `query` 方法，分别对应 SQL 更新和查询操作。

`intercept`：在这里我们获取 `MappedStatement` 和 `BoundSql`，并检查 SQL 是否需要加入租户 ID。如果租户 ID 不为空，就修改 SQL 语句来加上租户过滤条件。

`plugin`：通过 `Plugin.wrap()` 创建代理对象，使得拦截逻辑生效。

`setProperties`：可以用来获取配置文件中传入的属性（比如租户信息的获取方式等）。

#### 配置插件

在 `mybatis-config.xml` 中注册该插件：

```xml
<plugins>
    <plugin interceptor="com.example.interceptor.TenantInterceptor">
        <!-- 如果有需要传递的属性，可以在这里设置 -->
        <property name="someProperty" value="someValue"/>
    </plugin>
</plugins>

```

