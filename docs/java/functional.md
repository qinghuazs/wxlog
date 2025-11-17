---
title: FunctionalInterface注解
date: 2024/04/16
categories:
  - Java
  - Technology
---

`FunctionalInterface` 是一个信息性注解类型，用于指示接口类型符合 Java 语言规范定义的函数式接口要求。

从概念上讲，函数式接口只有一个抽象方法，其他方法都有默认的实现。

如果接口声明了一个覆盖 `java.lang.Object` 的公共方法之一的抽象方法，这也不会进入抽象方法计数，因为接口的任何实现都具有来自 `java.lang.Object` 或其他地方的实现。

请注意，函数式接口的实例可以使用 lambda 表达式、方法引用或构造函数引用来创建。

如果使用此注解类型对类型进行注解，则编译器需要生成错误消息，除非：

- 该类型是接口类型，而不是注释类型、枚举或类。
- 带注释的类型满足函数式接口的要求。

但是，无论接口声明中是否存在 `FunctionalInterface` 注解，编译器都会将满足函数式接口定义的任何接口视为函数式接口。

### 注解定义

```java
@Documented
@Retention(RetentionPolicy.RUNTIME)
//Target表明FunctionalInterface只能用来修饰接口、类和枚举
@Target(ElementType.TYPE)
public @interface FunctionalInterface {}
```

### 函数式接口只有一个抽象方法

```java
@FunctionalInterface
public interface HelloInterface {

    void test1(String str);

    void test2();
}
```

在 HelloInterface 中定义两个抽象方法，编译器会给出提示：在 HelloInterface 中找到多个非重写的 abtract 方法。

### 接口声明覆盖 java.lang.Object的抽象方法，不会进行抽象方法计数

```java
@FunctionalInterface
public interface HelloInterface {

    void test1(String str);

    String toString();
}
```

HelloInterface 定义了 toString() 方法，编译器不会给出错误提示，编译可以通过。

### 函数式接口的实例可以使用 lambda 表达式、方法引用或构造函数引用来创建

```java
public class Test {

    public static void main(String[] args) {
        //lambda表达式方法 定义抽象方法test1的实现，生成接口实例
        HelloInterface hello = x -> System.out.println("hello " + x);
        //进行接口调用
        hello.test1("qinghuazs");
       
        //构造器方式来实现抽象方法，生成接口实例
        HelloInterface hello2 = new HelloInterface() {
            @Override
            public void test1(String str) {
                System.out.println("hello construction " + str);
            }
        };
        hello2.test1("qinghuazs");
    }
}
```

### FunctionalInterface 用来修饰类或枚举会编译失败

因为 FunctionalInterface 要求必须有一个抽象方法，所以普通的类肯定是不满足条件的，我们来看一下抽象类。

```java
@FunctionalInterface
public abstract class AbstractTest {
    void test();
}

@FunctionalInterface
public enum EnumTest {
    
}
```

编译器提示 AbstractTest 不是函数式接口，编译失败。