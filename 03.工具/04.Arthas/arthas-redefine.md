---
title: Arthas在线热部署
date: 2024/10/22
---

## retransform

```bash
retransform <classFile>
```
目前用retransform 多一点，redefine 没用过。

## redefine

使用 Arthas 在服务器上反编译 class 文件并修改 Java 源码后，重新部署新的 class 文件。

获取到类的 classLoaderHash

```shell
$ sc -d com.company.yonbip.ctm.bam.bankAccountSync.service.open.BankEnterpriseAccountSyncOpenHandleService
```

classLoaderHash 可能有多个，一般会有 2 个，一个是类对象本身，一个是 Spring 的代理对象。

反编译成 Java 代码

```shell
$ jad --source-only com.company.yonbip.ctm.bam.bankAccountSync.service.open.BankEnterpriseAccountSyncOpenHandleService > /tmp/BankEnterpriseAccountSyncOpenHandleService.java
```

在服务器上编辑 Java 文件，或者将文件下载到本地编辑，修改完成后再上传到服务器

编译修改后的源码

```shell
$ mc -c 6f02402a /tmp/BankEnterpriseAccountSyncOpenHandleService.java -d /tmp
```

其中 `<classLoaderHash>` 是步骤 sc 命令获取的类加载器的 hashcode。

使用 `redefine` 命令加载新的 class 文件

```shell
$ redefine /tmp/com/company/yonbip/ctm/bam/bankAccountSync/service/open/BankEnterpriseAccountSyncOpenHandleService.class
```

注意点

- 确保在修改代码时不添加或删除方法、字段，因为这会导致热更新失败。
- 使用 `redefine` 命令后，如果再执行 `jad`、`watch`、`trace` 等命令，可能会重置字节码，因此需要谨慎操作。
- 热更新后的类在 JVM 重启后会恢复原样，因此这只是一种临时的解决方案。









