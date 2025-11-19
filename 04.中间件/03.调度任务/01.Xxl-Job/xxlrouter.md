---
title: 路由策略
date: 2024/04/17
---

XXL-JOB 在 `com.xxl.job.admin.core.route.ExecutorRouteStrategyEnum` 中内置了 10 种路由策略。

```java
public enum ExecutorRouteStrategyEnum {
     
    /**
     * 第一个地址
     */
    FIRST(I18nUtil.getString("jobconf_route_first"), new ExecutorRouteFirst()),
    /**
     * 最后一个地址
     */
    LAST(I18nUtil.getString("jobconf_route_last"), new ExecutorRouteLast()),
    /**
     * 轮询
     */
    ROUND(I18nUtil.getString("jobconf_route_round"), new ExecutorRouteRound()),
    /**
     * 随机
     */
    RANDOM(I18nUtil.getString("jobconf_route_random"), new ExecutorRouteRandom()),
    /**
     * 一致性哈希
     */
    CONSISTENT_HASH(I18nUtil.getString("jobconf_route_consistenthash"), new ExecutorRouteConsistentHash()),
    /**
     * LFU 最不经常使用
     */
    LEAST_FREQUENTLY_USED(I18nUtil.getString("jobconf_route_lfu"), new ExecutorRouteLFU()),
    /**
     * LRU 最近最少使用
     */
    LEAST_RECENTLY_USED(I18nUtil.getString("jobconf_route_lru"), new ExecutorRouteLRU()),
    /**
     * 故障转移
     */
    FAILOVER(I18nUtil.getString("jobconf_route_failover"), new ExecutorRouteFailover()),
    /**
     * 忙碌转移
     */
    BUSYOVER(I18nUtil.getString("jobconf_route_busyover"), new ExecutorRouteBusyover()),
    /**
     * 分片广播
     */
    SHARDING_BROADCAST(I18nUtil.getString("jobconf_route_shard"), null);
}
```

### 第一个

取执行器地址列表中第一条数据作为路由，简单粗暴。

```java
public class ExecutorRouteFirst extends ExecutorRouter {

    @Override
    public ReturnT<String> route(TriggerParam triggerParam, List<String> addressList){
        return new ReturnT<String>(addressList.get(0));
    }
}
```

### 最后一个

取执行器地址列表中最后数据作为路由。

```java
public class ExecutorRouteLast extends ExecutorRouter {

    @Override
    public ReturnT<String> route(TriggerParam triggerParam, List<String> addressList) {
        return new ReturnT<String>(addressList.get(addressList.size()-1));
    }
}
```

第一个和最后一个的策略会导致所有的请求都打到某个特定的执行器中，导致单个执行器的压力过大，不太推荐。

### 轮询策略

轮询策略是在 `routeCountEachJob` 缓存中（Map结构，key 是JobId, value 是上一次执行器的下标）记录每一个 Job 上一次执行时的地址索引，然后索引位置加1，再按照执行器数量进行取余操作，算出本次路由的地址。

为了避免首次轮询时都取第一个地址，造成瞬时的请求压力，在初始化时会对count进行一次随机取数，随机选择一个执行器进行执行，相当于走了一次随机策略。

```java
public class ExecutorRouteRound extends ExecutorRouter {

    private static ConcurrentMap<Integer, AtomicInteger> routeCountEachJob = new ConcurrentHashMap<>();
    private static long CACHE_VALID_TIME = 0;

    private static int count(int jobId) {
        //缓存24小时后失效
        if (System.currentTimeMillis() > CACHE_VALID_TIME) {
            routeCountEachJob.clear();
            CACHE_VALID_TIME = System.currentTimeMillis() + 1000*60*60*24;
        }

        AtomicInteger count = routeCountEachJob.get(jobId);
        if (count == null || count.get() > 1000000) {
            // 初始化时主动Random一次，缓解首次压力
            count = new AtomicInteger(new Random().nextInt(100));
        } else {
            // count++
            count.addAndGet(1);
        }
        routeCountEachJob.put(jobId, count);
        return count.get();
    }

    @Override
    public ReturnT<String> route(TriggerParam triggerParam, List<String> addressList) {
        //通过取余操作获取地址下标，从而获取到具体的地址
        String address = addressList.get(count(triggerParam.getJobId())%addressList.size());
        return new ReturnT<String>(address);
    }
}
```

### 随机策略

随机策略的实现也比较简单，就是实例化一个 Random 对象，生成一个随机下标，取随机下标对应的地址。

```java
public class ExecutorRouteRandom extends ExecutorRouter {

    private static Random localRandom = new Random();

    @Override
    public ReturnT<String> route(TriggerParam triggerParam, List<String> addressList) {
        String address = addressList.get(localRandom.nextInt(addressList.size()));
        return new ReturnT<String>(address);
    }
}
```

随机策略可能会导致分配不均衡，一些执行器可能会收到更多的任务，而另一些执行器可能会收到较少的任务，这可能导致系统资源的浪费或者某些执行器的负载过重。

### 一致性哈希策略

一致性哈希策略解决了随机策略的不均衡问题，同时保证了每个 Job 固定命中到同一个执行器上。

```java
public class ExecutorRouteConsistentHash extends ExecutorRouter {

    /**
     * 每个执行器都定义100个虚拟节点
     */
    private static int VIRTUAL_NODE_NUM = 100;

    /**
     * get hash code on 2^32 ring (md5散列的方式计算hash值)
     * @param key
     * @return
     */
    private static long hash(String key) {
        //这里省略了具体的hash实现，hash方法的作用主要是为了减少hash冲突
        return truncateHashCode;
    }

    public String hashJob(int jobId, List<String> addressList) {

        // ------A1------A2-------A3------
        // -----------J1------------------
        TreeMap<Long, String> addressRing = new TreeMap<Long, String>();
        //将每个执行器的地址虚拟化100个节点，放入到addressRing中
        for (String address: addressList) {
            for (int i = 0; i < VIRTUAL_NODE_NUM; i++) {
                long addressHash = hash("SHARD-" + address + "-NODE-" + i);
                addressRing.put(addressHash, address);
            }
        }
        //根据 JobId 计算 hash 值
        long jobHash = hash(String.valueOf(jobId));
        // 获取 KEY 大于等于指定key的部分数据
        SortedMap<Long, String> lastRing = addressRing.tailMap(jobHash);
        // 如果取不到大于等于 hash(jobId) 的数据，则表示在环中节点的位置位于0和1之间，所以取第一个节点即可
        // 如果有大于等于 hash(jobId) 的数据，则取子集合SortedMap中第一个节点，lastRing.firstKey()是子集合中第一个节点的key
        if (!lastRing.isEmpty()) {
            return lastRing.get(lastRing.firstKey());
        }
        return addressRing.firstEntry().getValue();
    }

    @Override
    public ReturnT<String> route(TriggerParam triggerParam, List<String> addressList) {
        String address = hashJob(triggerParam.getJobId(), addressList);
        return new ReturnT<String>(address);
    }
}
```

### LFU策略

LFU 最少使用（Least Frequently Used）策略，在该策略中，使用最少得那个执行器将被选中。

```java
public class ExecutorRouteLFU extends ExecutorRouter {

    /**
     * key jobId
     * value job使用过的执行器地址和使用次数的map
     */
    private static ConcurrentMap<Integer, HashMap<String, Integer>> jobLfuMap = new ConcurrentHashMap<Integer, HashMap<String, Integer>>();
    /**
     * 缓存的时间，目前是24小时清理缓存
     */
    private static long CACHE_VALID_TIME = 0;

    public String route(int jobId, List<String> addressList) {
        //缓存24小时失效
        if (System.currentTimeMillis() > CACHE_VALID_TIME) {
            jobLfuMap.clear();
            CACHE_VALID_TIME = System.currentTimeMillis() + 1000*60*60*24;
        }

        // lfu item init
        // Key排序可以用TreeMap+构造入参Compare；Value排序暂时只能通过ArrayList；
        // key：执行器地址, value : 执行器地址列表的下标
        HashMap<String, Integer> lfuItemMap = jobLfuMap.get(jobId);
        if (lfuItemMap == null) {
            lfuItemMap = new HashMap<String, Integer>();
            // 避免重复覆盖
            jobLfuMap.putIfAbsent(jobId, lfuItemMap);
        }

        // put new
        for (String address: addressList) {
            if (!lfuItemMap.containsKey(address) || lfuItemMap.get(address) >1000000 ) {
                // 初始化时主动Random一次，缓解首次压力
                lfuItemMap.put(address, new Random().nextInt(addressList.size()));
            }
        }

        // remove old 缓存中有的地址，在最新的地址列表中已经失效了，所以需要在缓存中删除掉这些地址
        List<String> delKeys = new ArrayList<>();
        for (String existKey: lfuItemMap.keySet()) {
            if (!addressList.contains(existKey)) {
                delKeys.add(existKey);
            }
        }
        if (delKeys.size() > 0) {
            for (String delKey: delKeys) {
                lfuItemMap.remove(delKey);
            }
        }

        // load least userd count address   按使用次数进行排序，拿到最少使用的那个地址
        List<Map.Entry<String, Integer>> lfuItemList = new ArrayList<Map.Entry<String, Integer>>(lfuItemMap.entrySet());
        Collections.sort(lfuItemList, new Comparator<Map.Entry<String, Integer>>() {
            @Override
            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                return o1.getValue().compareTo(o2.getValue());
            }
        });

        //使用一次之后，使用次数+1
        Map.Entry<String, Integer> addressItem = lfuItemList.get(0);
        String minAddress = addressItem.getKey();
        addressItem.setValue(addressItem.getValue() + 1);

        return addressItem.getKey();
    }

    @Override
    public ReturnT<String> route(TriggerParam triggerParam, List<String> addressList) {
        String address = route(triggerParam.getJobId(), addressList);
        return new ReturnT<String>(address);
    }
}
```

### LRU策略

LRU 最近最少被使用（Least Recently Used）策略的实现是基于 `LinkedHashMap` 的 `accessOrder` 属性实现的。`accessOrder` 指定顺序模式，当 `accessOrder`  为 false 时，则`LinkedHashMap`会按照元素插入的顺序来维护元素的顺序，即新添加的元素会被放到链表的末尾。在这种模式下，当遍历`LinkedHashMap`时，元素的顺序与插入顺序一致。当 `accessOrder`  为 true 时， 则`LinkedHashMap`会根据元素的访问顺序来调整元素的位置，即每次访问（通过`get()`、`put()`等方法）一个元素时，该元素会被移到链表的末尾。在这种模式下，元素的顺序与访问顺序一致，最近被访问过的元素会被放到链表的末尾。通过这个特性，能确保最近被访问过的元素总是位于链表的末尾，而最久未被访问的元素会被放到链表的前面，从而方便实现 RLU 策略。

```java
public class ExecutorRouteLRU extends ExecutorRouter {

    /**
     * key jobId
     * value ： map   key address    value address
     */
    private static ConcurrentMap<Integer, LinkedHashMap<String, String>> jobLRUMap = new ConcurrentHashMap<Integer, LinkedHashMap<String, String>>();
    /**
     * 缓存失效时间
     */
    private static long CACHE_VALID_TIME = 0;

    public String route(int jobId, List<String> addressList) {
        // 清理缓存
        if (System.currentTimeMillis() > CACHE_VALID_TIME) {
            jobLRUMap.clear();
            CACHE_VALID_TIME = System.currentTimeMillis() + 1000*60*60*24;
        }

        // init lru
        //key address  value address
        LinkedHashMap<String, String> lruItem = jobLRUMap.get(jobId);
        if (lruItem == null) {
            /**
             * LinkedHashMap
             *      a、accessOrder：true=访问顺序排序（get/put时排序）；false=插入顺序排期；
             *      b、removeEldestEntry：新增元素时将会调用，返回true时会删除最老元素；可封装LinkedHashMap并重写该方法，比如定义最大容量，超出是返回true即可实现固定长度的LRU算法；
             */
            lruItem = new LinkedHashMap<String, String>(16, 0.75f, true);
            jobLRUMap.putIfAbsent(jobId, lruItem);
        }
        // put new   初始化所有地址到lruItem中
        for (String address: addressList) {
            if (!lruItem.containsKey(address)) {
                lruItem.put(address, address);
            }
        }
        // remove old   移除缓存中已失效的地址
        List<String> delKeys = new ArrayList<>();
        for (String existKey: lruItem.keySet()) {
            if (!addressList.contains(existKey)) {
                delKeys.add(existKey);
            }
        }
        if (delKeys.size() > 0) {
            for (String delKey: delKeys) {
                lruItem.remove(delKey);
            }
        }

        // load   取lruItem中第一个数据，因为accessOrder设置为true，每次被访问的数据都会放到链表的末尾，所以链表头节点就是最近最少使用的那条数据
        String eldestKey = lruItem.entrySet().iterator().next().getKey();
        String eldestValue = lruItem.get(eldestKey);
        return eldestValue;
    }

    @Override
    public ReturnT<String> route(TriggerParam triggerParam, List<String> addressList) {
        String address = route(triggerParam.getJobId(), addressList);
        return new ReturnT<String>(address);
    }
}
```

### 故障转移

故障转移策略的实现机制是遍历执行器地址列表，找到第一个心跳结果是成功的地址，并返回。

故障转移策略下也会出现大量请求打到同一个节点上的问题。

```java
public class ExecutorRouteFailover extends ExecutorRouter {

    @Override
    public ReturnT<String> route(TriggerParam triggerParam, List<String> addressList) {
        StringBuffer beatResultSB = new StringBuffer();
        for (String address : addressList) {
            // beat
            ReturnT<String> beatResult = null;
            try {
                //获取心跳信息
                ExecutorBiz executorBiz = XxlJobScheduler.getExecutorBiz(address);
                beatResult = executorBiz.beat();
            } catch (Exception e) {
                logger.error(e.getMessage(), e);
                beatResult = new ReturnT<String>(ReturnT.FAIL_CODE, ""+e );
            }
            beatResultSB.append( (beatResultSB.length()>0)?"<br><br>":"")
                    .append(I18nUtil.getString("jobconf_beat") + "：")
                    .append("<br>address：").append(address)
                    .append("<br>code：").append(beatResult.getCode())
                    .append("<br>msg：").append(beatResult.getMsg());

            //如果心跳结果是成功，则直接返回；如果心跳结果不是成功，则继续查询下一个地址的心跳信息
            if (beatResult.getCode() == ReturnT.SUCCESS_CODE) {
                beatResult.setMsg(beatResultSB.toString());
                beatResult.setContent(address);
                return beatResult;
            }
        }
        return new ReturnT<String>(ReturnT.FAIL_CODE, beatResultSB.toString());
    }
}
```

### 忙碌转移

忙碌转移策略的实现机制和故障转移策略类似，也是借助心跳信息去获取空闲心跳结果为成功的那个节点，并返回。

```java
public class ExecutorRouteBusyover extends ExecutorRouter {

    @Override
    public ReturnT<String> route(TriggerParam triggerParam, List<String> addressList) {
        StringBuffer idleBeatResultSB = new StringBuffer();
        for (String address : addressList) {
            // beat
            ReturnT<String> idleBeatResult = null;
            try {
                ExecutorBiz executorBiz = XxlJobScheduler.getExecutorBiz(address);
                idleBeatResult = executorBiz.idleBeat(new IdleBeatParam(triggerParam.getJobId()));
            } catch (Exception e) {
                logger.error(e.getMessage(), e);
                idleBeatResult = new ReturnT<String>(ReturnT.FAIL_CODE, ""+e );
            }
            idleBeatResultSB.append( (idleBeatResultSB.length()>0)?"<br><br>":"")
                    .append(I18nUtil.getString("jobconf_idleBeat") + "：")
                    .append("<br>address：").append(address)
                    .append("<br>code：").append(idleBeatResult.getCode())
                    .append("<br>msg：").append(idleBeatResult.getMsg());

            // beat success
            if (idleBeatResult.getCode() == ReturnT.SUCCESS_CODE) {
                idleBeatResult.setMsg(idleBeatResultSB.toString());
                idleBeatResult.setContent(address);
                return idleBeatResult;
            }
        }
        return new ReturnT<String>(ReturnT.FAIL_CODE, idleBeatResultSB.toString());
    }
}
```

### 分片广播

分片广播策略并没有单独的策略类，而是在 `XxlJobTriggerd#trigger()` 方法中进行了判断，如果配置了分片广播策略，则遍历执行每个注册地址，进行集群广播。

```java
if (ExecutorRouteStrategyEnum.SHARDING_BROADCAST==ExecutorRouteStrategyEnum.match(jobInfo.getExecutorRouteStrategy(), null)
    && group.getRegistryList()!=null && !group.getRegistryList().isEmpty()
    && shardingParam==null) {
    for (int i = 0; i < group.getRegistryList().size(); i++) {
        processTrigger(group, jobInfo, finalFailRetryCount, triggerType, i, group.getRegistryList().size());
    }
} else {
    if (shardingParam == null) {
        shardingParam = new int[]{0, 1};
    }
    processTrigger(group, jobInfo, finalFailRetryCount, triggerType, shardingParam[0], shardingParam[1]);
}
```

