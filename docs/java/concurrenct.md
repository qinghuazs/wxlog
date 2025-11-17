---
title: Java Lock
date: 2025-11-17
permalink: /docs/java/concurrenct/
categories:
  - Java
  - Technology
---

# Java 锁机制

## 什么是锁
锁是Java并发编程中用于控制多个线程对共享资源访问的重要机制。它能确保在同一时刻只有一个线程可以访问被保护的资源，从而保证数据的一致性和线程安全。

## 锁的分类

### 1. 可重入锁（ReentrantLock）
ReentrantLock是Java中最常用的锁之一，它支持重入特性，即同一个线程可以多次获取同一把锁。

```java
public class Counter {
    private final ReentrantLock lock = new ReentrantLock();
    private int count = 0;

    public void increment() {
        lock.lock();
        try {
            count++;
        } finally {
            lock.unlock();
        }
    }
}
```

### 2. 读写锁（ReadWriteLock）
读写锁允许多个线程同时读取共享资源，但在写入时需要独占访问。

```java
public class Cache {
    private final ReadWriteLock rwLock = new ReentrantReadWriteLock();
    private final Lock readLock = rwLock.readLock();
    private final Lock writeLock = rwLock.writeLock();
    private Map<String, String> cache = new HashMap<>();

    public String get(String key) {
        readLock.lock();
        try {
            return cache.get(key);
        } finally {
            readLock.unlock();
        }
    }

    public void put(String key, String value) {
        writeLock.lock();
        try {
            cache.put(key, value);
        } finally {
            writeLock.unlock();
        }
    }
}
```

### 3. 公平锁与非公平锁
- 公平锁：按照线程请求锁的顺序获取锁
- 非公平锁：允许线程插队获取锁，可能导致某些线程饥饿，但整体性能更好

```java
// 公平锁
ReentrantLock fairLock = new ReentrantLock(true);
// 非公平锁（默认）
ReentrantLock unfairLock = new ReentrantLock(false);
```

## 锁的最佳实践

### 1. 选择合适的锁
- 简单同步场景：优先使用synchronized
- 需要高级特性：使用ReentrantLock
- 读多写少场景：使用ReadWriteLock

### 2. 正确使用锁
```java
Lock lock = new ReentrantLock();
lock.lock();
try {
    // 临界区代码
} finally {
    lock.unlock(); // 确保锁一定被释放
}
```

### 3. 避免死锁
- 固定加锁顺序
- 使用限时锁
- 避免嵌套锁

```java
// 使用限时锁避免死锁
if (lock.tryLock(1, TimeUnit.SECONDS)) {
    try {
        // 临界区代码
    } finally {
        lock.unlock();
    }
} else {
    // 获取锁失败的处理
}
```

## 性能考虑

### 1. 锁的粒度
- 降低锁的粒度可以提高并发性
- 过细的锁粒度会增加维护成本

### 2. 锁的竞争
- 减少锁的持有时间
- 避免在临界区进行耗时操作

## 总结
Java锁机制是并发编程中的重要工具，合理使用锁可以确保线程安全，但需要注意避免死锁、性能问题等潜在问题。选择合适的锁类型，遵循最佳实践，对于开发高质量的并发程序至关重要。
