---
title: 比特币交易构建与广播 - 视频脚本
date: 2025-01-22
categories:
  - Technology
  - Learning
---

# 比特币交易构建与广播 - 视频脚本

**视频时长:10分钟**
**目标受众:想要实现比特币交易功能的开发者**

---

## 【开场Hook】(30秒,约120字)

大家好!你有没有想过,点击"发送"按钮后,比特币交易是如何构建的?为什么有时候手续费高,有时候低?什么是找零?如何签名?如何广播?

构建一笔比特币交易,涉及UTXO选择、输入输出构建、签名生成、手续费计算等多个步骤,每一步都有讲究!

今天这期视频,我要带你从零实现一笔完整的比特币交易,从构建到签名再到广播,让你掌握比特币交易的底层原理!

---

## 【主题介绍】(30秒,约130字)

这期视频我会为你讲解交易构建的四个核心步骤:

- **第一**,交易结构,理解输入、输出、见证数据的组成
- **第二**,UTXO选择,学习如何选择最优的未花费输出
- **第三**,交易签名,掌握ECDSA签名和scriptSig构建
- **第四**,交易广播,看看如何将交易发送到网络并监控状态

掌握这些,你就能完全控制比特币交易的每个细节,实现专业的钱包功能!

---

## 【第一部分:交易结构详解】(2.5分钟,约600字)

### 1. 交易的组成

一笔比特币交易包含4个部分:

**版本号 (4字节):**
- 当前版本:1或2
- 版本2支持BIP68相对时间锁

**输入列表 (TxIn):**
- 引用前序交易的输出
- 提供解锁脚本(scriptSig)
- 证明你有权花费

**输出列表 (TxOut):**
- 指定接收方和金额
- 提供锁定脚本(scriptPubKey)
- 定义花费条件

**锁定时间 (4字节):**
- 0表示立即有效
- 非0表示在指定时间/区块后有效

### 2. 交易输入(TxIn)

每个输入包含:

**前序交易哈希 (32字节):**
- 你要花费的UTXO来自哪笔交易

**输出索引 (4字节):**
- 前序交易的第几个输出

**解锁脚本 (变长):**
- scriptSig
- 包含签名和公钥
- 证明你是UTXO的所有者

**序列号 (4字节):**
- 0xFFFFFFFE:启用RBF(Replace-By-Fee)
- 0xFFFFFFFF:禁用RBF
- 其他值:相对时间锁

**示例:**

Alice要花费一个UTXO:
```
前序交易:a1b2c3d4...
输出索引:0
解锁脚本:<签名> <公钥>
序列号:0xFFFFFFFE
```

### 3. 交易输出(TxOut)

每个输出包含:

**金额 (8字节):**
- 单位:聪(satoshi)
- 1 BTC = 100,000,000聪

**锁定脚本 (变长):**
- scriptPubKey
- 定义谁能花费这个输出

**P2PKH输出示例:**

```
金额:100,000,000聪(1 BTC)
锁定脚本:
  OP_DUP
  OP_HASH160
  <公钥哈希>
  OP_EQUALVERIFY
  OP_CHECKSIG
```

意思是:"只有提供公钥和对应的签名,才能花费"

### 4. 完整交易示例

Alice向Bob转账1 BTC:

**输入:**
- 前序交易:txid_123...
- 输出索引:0
- 金额:1.5 BTC
- 解锁脚本:<Alice签名> <Alice公钥>

**输出1(支付):**
- 金额:1.0 BTC
- 锁定脚本:OP_DUP OP_HASH160 <Bob公钥哈希> ...

**输出2(找零):**
- 金额:0.499 BTC
- 锁定脚本:OP_DUP OP_HASH160 <Alice公钥哈希> ...

**手续费:**
- 1.5 - 1.0 - 0.499 = 0.001 BTC
- 给矿工的奖励

### 5. 交易大小

典型P2PKH交易:
```
版本:4字节
输入数量:1字节
输入:148字节/个
输出数量:1字节
输出:34字节/个
锁定时间:4字节

1输入2输出:
4 + 1 + 148 + 1 + 68 + 4 = 226字节
```

**手续费计算:**
```
手续费 = 交易大小 × 费率
226字节 × 10 sat/byte = 2,260聪
```

---

## 【第二部分:UTXO选择策略】(2.5分钟,约600字)

### 1. 为什么需要UTXO选择?

比特币没有"账户余额",只有UTXO(未花费输出)!

**示例:**

Alice的钱包:
- UTXO1:0.5 BTC
- UTXO2:0.3 BTC
- UTXO3:1.0 BTC
- UTXO4:0.1 BTC

总余额:1.9 BTC

如果Alice要转账0.8 BTC,选哪些UTXO?

### 2. 选择策略

**策略1:贪心算法(最简单)**

按金额从大到小排序,贪心选择:
```
目标:0.8 BTC
选择UTXO3(1.0 BTC) → 足够!
```

**优点:**简单快速
**缺点:**可能产生大量找零

**策略2:精确匹配**

尝试找到总和正好等于目标的组合:
```
目标:0.8 BTC
尝试:0.5 + 0.3 = 0.8 BTC ✓
选择UTXO1 + UTXO2
```

**优点:**无找零,节省费用
**缺点:**计算复杂

**策略3:Branch and Bound(Bitcoin Core使用)**

综合考虑:
- 尽量精确匹配
- 避免找零
- 减少交易大小

### 3. 手续费估算

在选择UTXO时,需要估算手续费:

```java
long estimateFee(int numInputs, int numOutputs, long feeRate) {
    // P2PKH交易大小估算
    int txSize = 10                    // 版本+锁定时间
               + numInputs * 148       // 每个输入约148字节
               + numOutputs * 34;      // 每个输出约34字节

    return txSize * feeRate;
}
```

**示例:**

1输入2输出,费率10 sat/byte:
```
大小 = 10 + 1×148 + 2×34 = 226字节
手续费 = 226 × 10 = 2,260聪 ≈ 0.000023 BTC
```

### 4. 找零处理

**计算找零:**

```
总输入 = 1.0 BTC
支付金额 = 0.7 BTC
手续费 = 0.0023 BTC
找零 = 1.0 - 0.7 - 0.0023 = 0.2977 BTC
```

**粉尘限制:**

如果找零 < 546聪,不创建找零输出!

**原因:**
- 546聪是粉尘限制
- 低于这个金额,花费成本 > 自身价值
- 作为额外手续费给矿工

### 5. 实战示例

Alice要转账0.5 BTC给Bob:

**步骤1:查询UTXO**
```
UTXO1: 0.3 BTC
UTXO2: 0.4 BTC
UTXO3: 0.2 BTC
```

**步骤2:选择UTXO**
```
目标:0.5 BTC
选择UTXO1(0.3) + UTXO2(0.4) = 0.7 BTC
```

**步骤3:计算找零**
```
输入:0.7 BTC
输出:0.5 BTC(给Bob)
手续费:0.0023 BTC(估算)
找零:0.7 - 0.5 - 0.0023 = 0.1977 BTC
```

**步骤4:构建交易**
```
输入:
  - UTXO1 (0.3 BTC)
  - UTXO2 (0.4 BTC)

输出:
  - Bob: 0.5 BTC
  - Alice(找零): 0.1977 BTC
```

---

## 【第三部分:交易签名】(2.5分钟,约600字)

### 1. 为什么需要签名?

签名证明:
- 你是UTXO的所有者
- 你授权这笔交易
- 交易未被篡改

### 2. 签名流程

**步骤1:构建签名哈希**

不是对整个交易签名,而是对"签名哈希"签名!

**流程:**
1. 复制交易
2. 清空所有输入的scriptSig
3. 将当前输入的scriptSig设置为前序输出的scriptPubKey
4. 序列化交易
5. 添加SIGHASH类型(4字节)
6. 双重SHA256

**示例:**

签名输入0:
```
原交易:
  输入0: scriptSig=<签名> <公钥>
  输入1: scriptSig=<签名> <公钥>

签名哈希版本:
  输入0: scriptSig=<前序输出的scriptPubKey>
  输入1: scriptSig=<空>

序列化 + 添加SIGHASH_ALL(0x01)
双重SHA256 → 签名哈希
```

**步骤2:ECDSA签名**

使用私钥对签名哈希签名:
```
签名 = ECDSA.sign(签名哈希, 私钥)
```

**步骤3:构建scriptSig**

P2PKH的scriptSig:
```
<签名> <公钥>
```

**步骤4:设置到输入**

将scriptSig设置到对应输入。

### 3. SIGHASH类型

控制签名覆盖的范围:

**SIGHASH_ALL (0x01):**
- 签名所有输入和输出
- 最常用
- 交易不可修改

**SIGHASH_NONE (0x02):**
- 签名所有输入,不签名输出
- 任何人可以修改输出
- 用于"空白支票"

**SIGHASH_SINGLE (0x03):**
- 签名所有输入和对应的一个输出
- 其他输出可修改

**SIGHASH_ANYONECANPAY (0x80):**
- 只签名当前输入
- 可与上述组合
- 用于众筹

### 4. 完整签名代码逻辑

```java
void signInput(Transaction tx, int inputIndex, PrivateKey key, UTXO utxo) {
    // 1. 构建签名哈希
    byte[] sigHash = createSignatureHash(tx, inputIndex, utxo);

    // 2. ECDSA签名
    byte[] signature = ECDSA.sign(sigHash, key);

    // 3. 添加SIGHASH类型
    byte[] sigWithHashType = append(signature, 0x01);

    // 4. 获取公钥
    byte[] publicKey = key.getPublicKey().toBytes();

    // 5. 构建scriptSig
    Script scriptSig = new Script();
    scriptSig.addData(sigWithHashType);
    scriptSig.addData(publicKey);

    // 6. 设置到输入
    tx.getInputs().get(inputIndex).setScriptSig(scriptSig);
}
```

### 5. 验证签名

矿工和节点验证:

1. 提取scriptSig中的签名和公钥
2. 重建签名哈希
3. 验证:`ECDSA.verify(签名哈希, 签名, 公钥)`
4. 检查公钥哈希是否匹配scriptPubKey

**如果验证失败:**
- 交易无效
- 被节点拒绝
- 不会被打包

---

## 【第四部分:交易广播与监控】(2分钟,约500字)

### 1. 广播方式

**方式1:通过Bitcoin Core**

最可靠的方式:
```java
BitcoinRPC rpc = new BitcoinRPC("user", "pass", "localhost", 18332);

// 测试交易(可选)
Map<String, Object> testResult = rpc.testMempoolAccept(
    Arrays.asList(rawTxHex)
);

if (!testResult.get("allowed")) {
    String reason = testResult.get("reject-reason");
    throw new Exception("交易被拒绝: " + reason);
}

// 广播交易
String txId = rpc.sendRawTransaction(rawTxHex);
System.out.println("交易已广播: " + txId);
```

**方式2:通过公共API**

Blockchain.com API:
```java
String apiUrl = "https://blockchain.info/pushtx";
HttpResponse response = httpPost(apiUrl, "tx=" + rawTxHex);

if (response.statusCode() == 200) {
    System.out.println("交易已广播");
}
```

BlockCypher API:
```java
String apiUrl = "https://api.blockcypher.com/v1/btc/test3/txs/push";
String json = "{\"tx\": \"" + rawTxHex + "\"}";
HttpResponse response = httpPost(apiUrl, json);
```

**方式3:P2P网络直连**

连接比特币节点,发送`tx`消息。

### 2. 监控交易状态

**查询确认数:**

```java
void monitorTransaction(String txId) throws Exception {
    while (true) {
        Map<String, Object> txInfo = rpc.getTransaction(txId);
        int confirmations = (int) txInfo.get("confirmations");

        System.out.println("确认数: " + confirmations);

        if (confirmations >= 6) {
            System.out.println("交易已充分确认!");
            break;
        }

        Thread.sleep(30_000);  // 等待30秒
    }
}
```

**确认时间:**
- 0次:在内存池,未确认
- 1次:被打包进区块(约10分钟)
- 3次:较安全(约30分钟)
- 6次:充分确认(约1小时)

### 3. 高级功能

**RBF(Replace-By-Fee):**

提高手续费加速交易:
```java
// 原交易sequence < 0xfffffffe
Transaction newTx = originalTx.clone();
newTx.increaseFee();  // 提高手续费
newTx.resign();       // 重新签名
broadcast(newTx);     // 广播替换
```

**CPFP(Child Pays For Parent):**

创建子交易加速父交易:
```java
// 花费父交易的输出
Transaction childTx = new Transaction();
childTx.addInput(parentTxId, outputIndex);
childTx.addOutput(address, amount - highFee);
childTx.sign();
broadcast(childTx);

// 矿工会同时打包父子交易(组合费率高)
```

**批量支付:**

一笔交易支付多个地址,节省手续费:
```java
Transaction batchTx = new Transaction();
batchTx.addInput(utxo);

// 添加多个输出
for (Recipient r : recipients) {
    batchTx.addOutput(r.address, r.amount);
}

// 手续费节省:约40-60%
```

---

## 【总结回顾】(30秒,约150字)

让我们回顾比特币交易构建的核心步骤:

**1. 交易结构**
- 版本、输入、输出、锁定时间
- P2PKH输出最常用
- 交易大小影响手续费

**2. UTXO选择**
- 贪心算法、精确匹配
- Branch and Bound最优
- 找零需考虑粉尘限制

**3. 交易签名**
- 签名哈希构建
- ECDSA签名
- scriptSig包含签名和公钥

**4. 交易广播**
- Bitcoin Core RPC最可靠
- 公共API备选
- 监控确认数

---

## 【结尾互动】(30秒,约120字)

恭喜你!现在你已经掌握了比特币交易的底层原理:从UTXO选择到签名生成,从交易构建到网络广播,每个细节都了如指掌!

**建议:**
- 先在Regtest测试
- 再在Testnet验证
- 充分测试后上主网
- 注意手续费和粉尘限制

如果这期视频让你学会了交易构建,请点赞支持。下期我会讲比特币SPV轻节点实现。关注我,我们下期见!

---

**视频脚本总字数:约2500字**
**预计语速:220-250字/分钟**
**实际时长:10-11分钟**
