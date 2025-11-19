---
title: Arthas常用命令
date: 2024/04/28
---

## 工具下载

如果服务器上没有 arthas，可以通过命令进行下载。

```bash
curl -O https://arthas.aliyun.com/arthas-boot.jar
```

如果下载速度比较慢，可以使用阿里的镜像

```bash
java -jar arthas-boot.jar --repo-mirror aliyun --use-http
```

## sc

`-d` 获取类的详细信息  

`-f` 获取类的属性信息

```bash
[arthas@23271]$ sc -d com.qinghuazs.lemon2024.controller.markdown.MarkdownController
 class-info        com.qinghuazs.lemon2024.controller.markdown.MarkdownController                                                          
 code-source       /Users/user/Documents/company/code/lemon2024/lemon2024/target/classes/                                            
 name              com.qinghuazs.lemon2024.controller.markdown.MarkdownController                                                          
 isInterface       false                                                                                                                   
 isAnnotation      false                                                                                                                   
 isEnum            false                                                                                                                   
 isAnonymousClass  false                                                                                                                   
 isArray           false                                                                                                                   
 isLocalClass      false                                                                                                                   
 isMemberClass     false                                                                                                                   
 isPrimitive       false                                                                                                                   
 isSynthetic       false                                                                                                                   
 simple-name       MarkdownController                                                                                                      
 modifier          public                                                                                                                  
 annotation        org.springframework.web.bind.annotation.RestController                                                                  
 interfaces                                                                                                                                
 super-class       +-java.lang.Object                                                                                                      
 class-loader      +-jdk.internal.loader.ClassLoaders$AppClassLoader@4e0e2f2a                                                              
                     +-jdk.internal.loader.ClassLoaders$PlatformClassLoader@5a69404c                                                       
 classLoaderHash   4e0e2f2a                                                                                                                

Affect(row-cnt:1) cost in 29 ms.
```

classLoaderHash 是类加载器的 hash 值，这个 hash 值在 ognl 中还会用到。

模糊匹配

```bash
[arthas@23593]$ sc *Controller
com.mysql.cj.protocol.ServerSessionStateController
com.mysql.cj.protocol.a.NativeServerSessionStateController
com.qinghuazs.lemon2024.controller.markdown.MarkdownController
com.taobao.arthas.core.shell.system.JobController
com.taobao.arthas.core.shell.system.impl.GlobalJobControllerImpl
com.taobao.arthas.core.shell.system.impl.JobControllerImpl
java.lang.ModuleLayer$Controller
java.security.AccessController
jdk.proxy2.$Proxy21
jdk.proxy2.$Proxy52
org.apache.naming.ContextAccessController
org.springframework.boot.autoconfigure.web.servlet.error.AbstractErrorController
org.springframework.boot.autoconfigure.web.servlet.error.BasicErrorController
org.springframework.boot.web.servlet.error.ErrorController
org.springframework.stereotype.Controller
org.springframework.web.bind.annotation.RestController
sun.net.www.protocol.jar.JarFileFactory
sun.net.www.protocol.jar.URLJarFile$URLJarFileCloseController
Affect(row-cnt:18) cost in 82 ms.
```

## watch

最常用的指令

```bash
watch com.qinghuazs.lemon2024.controller.markdown.MarkdownController insert '{params,returnObj,throwExp}'  -n 5 -e  -x 3 
```

`-n` 次数，watch n 次后停止 watch 行为

`-e` 只有抛出异常时才触发 watch 行为

`-x` 表示遍历深度，主要用于查看具体的参数值，默认为1，最大为4。不过一般还是推荐在 ognl 表示式中展示具体的信息，如

```bash
watch com.qinghuazs.lemon2024.controller.markdown.MarkdownController insert '{params[0]}'  -n 5  -x 3 
```

`params[0]` 表示方法入参的第一个参数。

条件表达式，主要用于请求过滤，满足特定的条件才会进行 watch .

```
watch com.qinghuazs.lemon2024.controller.markdown.MarkdownController insert '{params,returnObj,throwExp}' 'params[0].age < 18'  -n 5  -x 3 
```

字符串类型的过滤

```
watch com.company.iuap.yms.http.YmsHttpClient execute '{returnObj.getBodyString()}' 'params[0].getUrl().contains("api-forward") '  -n 5  -x 3 
```

也可以根据耗时来过滤

```
watch com.qinghuazs.lemon2024.controller.markdown.MarkdownController insert '{params,returnObj,throwExp}' 'params[0].age < 18' '#cost>200'  -n 5  -x 3 
```

cost 的对应单位是毫秒（ms）

## 查看类属性

```shell
vmtool --action getInstances --className com.company.yonbip.ctm.bankconnection.base.NJCAContext  --express 'instances[0]'
```





