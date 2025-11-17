---
title: ImmuatalePair和ImmutableTriple
date: 2024/04/15
permalink: /docs/java/first/
categories:
  - Java
  - Technology
---

`ImmuatalePair` 和 `ImmutableTriple` 是 Apache Commons Lang 库中提供的工具类，`ImmuatalePair` 用于表示一个不可变的二元组， `ImmutableTriple` 表示不可变的三元组。

`Immutable*` 中的元素可以是任意类型，一旦创建，就不可修改。

ImmuatalePair 和  ImmuataleTriple 可以用来进行字符串的组合，尤其是在 Map 中 key 为 2 个或者 3 个元素组成时，非常方便。

```java
public class ImmutableTripleTest {

    public static void main(String[] args) {
        User user = new User();
        user.setUsername("qinghuazs");
        user.setPassword("qinghuazs");
        user.setId(1);
        user.setAge(18);

        ImmutableTriple<String, String, Integer> triple = new ImmutableTriple<>(user.getUsername(), user.getPassword(), user.getId());
        System.out.println(triple);

        ImmutablePair<String, String> pair = new ImmutablePair<>(user.getUsername(), user.getPassword());
        System.out.println(pair);

        Map<ImmutableTriple<String, String, Integer>, User> map = new HashMap<>();
        map.put(triple, user);

        Map<ImmutablePair<String, String>, User> map2 = new HashMap<>();
        map2.put(pair, user);
    }
}
```

ImmutablePair 和 ImmutableTriple 的 `toString()` 方法复用的父类 `Pair` 和 `Triple` 中的 toString() 方法。

```java
@Override
public String toString() {
    return "(" + getLeft() + ',' + getRight() + ')';
}

public String toString() {
    return "(" + getLeft() + "," + getMiddle() + "," + getRight() + ")";
}
```

打印出来即为

```
(qinghuazs,qinghuazs,1)
(qinghuazs,qinghuazs)
```

当 Map 的 key 需要支持特定的格式，可以自定义 Pair 和 Triple，重写 toString() 方法即可。

### 自定义Pair 

```java
public class ImmutablePairKey<L, R> {

    private final ImmutablePair<L, R> pair;

    public ImmutablePairKey(ImmutablePair<L, R> pair) {
        this.pair = pair;
    }
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ImmutablePairKey<?, ?> that = (ImmutablePairKey<?, ?>) o;
        return pair.equals(that.pair);
    }

    @Override
    public int hashCode() {
        return pair.hashCode();
    }

    @Override
    public String toString() {
        return pair.getLeft().toString() + "-" + pair.getRight().toString();
    }
}
```

### 自定义 Triple

```java
public class ImmutableTripleKey<L, M, R> {

    private final ImmutableTriple<L, M, R> triple;

    public ImmutableTripleKey(ImmutableTriple<L, M, R> triple) {
        this.triple = triple;
    }
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ImmutableTripleKey that = (ImmutableTripleKey<?, ?, ?>) o;
        return triple.equals(that.triple);
    }

    @Override
    public int hashCode() {
        return triple.hashCode();
    }

    @Override
    public String toString() {
        return triple.getLeft() + "-" + triple.getMiddle() + "-" + triple.getRight();
    }
}
```

