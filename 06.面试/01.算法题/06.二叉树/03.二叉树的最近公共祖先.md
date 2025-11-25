---
title: 二叉树的最近公共祖先
date: 2025-09-19
categories:
  - Algorithm
  - LeetCode
---

## 1. 问题描述和示例

### 问题描述
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为："对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。"

### 示例

**示例 1：**
```
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。

        3
       / \
      5   1
     / \ / \
    6  2 0  8
      / \
     7   4
```

**示例 2：**
```
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出：5
解释：节点 5 和节点 4 的最近公共祖先是节点 5 。因为根据定义最近公共祖先节点可以为节点本身。

        3
       / \
      5   1
     / \ / \
    6  2 0  8
      / \
     7   4
```

**示例 3：**
```
输入：root = [1,2], p = 1, q = 2
输出：1
解释：只有两个节点，根节点就是最近公共祖先。

    1
   /
  2
```

## 2. 核心难点分析

### 主要难点
1. **祖先定义理解**：一个节点可以是它自己的祖先
2. **递归思维**：如何通过递归找到最近公共祖先
3. **返回值设计**：递归函数应该返回什么信息
4. **边界条件**：处理空节点和特殊情况

### 关键要点
- 最近公共祖先必须是p和q的共同祖先中深度最大的
- 如果当前节点是p或q之一，它可能就是最近公共祖先
- 需要在左右子树中分别查找p和q
- 根据左右子树的查找结果来判断最近公共祖先的位置

## 3. 多种解法对比

### 解法一：递归法（推荐）
- **思路**：后序遍历，从底向上返回信息
- **优点**：代码简洁，时间复杂度最优
- **缺点**：需要理解递归的精髓

### 解法二：存储父节点法
- **思路**：先遍历存储所有节点的父节点，再向上查找
- **优点**：思路直观，容易理解
- **缺点**：需要额外空间存储父节点信息

### 解法三：路径记录法
- **思路**：分别找到从根到p和q的路径，然后找最后一个公共节点
- **优点**：逻辑清晰
- **缺点**：需要两次遍历，空间复杂度较高

## 4. 详细Java代码实现

### 二叉树节点定义
```java
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x) { val = x; }
}
```

### 解法一：递归法（最优解）
```java
public class Solution {
    /**
     * 寻找二叉树中两个节点的最近公共祖先
     * @param root 二叉树根节点
     * @param p 目标节点p
     * @param q 目标节点q
     * @return 最近公共祖先节点
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // 基础情况：如果当前节点为空，返回null
        if (root == null) {
            return null;
        }
        
        // 如果当前节点是p或q之一，返回当前节点
        if (root == p || root == q) {
            return root;
        }
        
        // 在左子树中查找p和q的最近公共祖先
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        
        // 在右子树中查找p和q的最近公共祖先
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        
        // 根据左右子树的返回结果判断最近公共祖先
        if (left != null && right != null) {
            // 如果左右子树都找到了目标节点，当前节点就是最近公共祖先
            return root;
        }
        
        // 如果只有一边找到了目标节点，返回找到的那一边
        return left != null ? left : right;
    }
}
```

### 解法二：存储父节点法
```java
import java.util.*;

public class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // 存储每个节点的父节点
        Map<TreeNode, TreeNode> parent = new HashMap<>();
        
        // DFS遍历，记录每个节点的父节点
        dfs(root, parent);
        
        // 存储从p到根节点的所有祖先
        Set<TreeNode> ancestors = new HashSet<>();
        
        // 从p开始向上遍历到根节点
        while (p != null) {
            ancestors.add(p);
            p = parent.get(p);
        }
        
        // 从q开始向上遍历，找到第一个在ancestors中的节点
        while (q != null) {
            if (ancestors.contains(q)) {
                return q;
            }
            q = parent.get(q);
        }
        
        return null;
    }
    
    /**
     * DFS遍历二叉树，记录每个节点的父节点
     */
    private void dfs(TreeNode root, Map<TreeNode, TreeNode> parent) {
        if (root == null) {
            return;
        }
        
        if (root.left != null) {
            parent.put(root.left, root);
            dfs(root.left, parent);
        }
        
        if (root.right != null) {
            parent.put(root.right, root);
            dfs(root.right, parent);
        }
    }
}
```

### 解法三：路径记录法
```java
import java.util.*;

public class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // 找到从根节点到p的路径
        List<TreeNode> pathToP = new ArrayList<>();
        findPath(root, p, pathToP);
        
        // 找到从根节点到q的路径
        List<TreeNode> pathToQ = new ArrayList<>();
        findPath(root, q, pathToQ);
        
        // 找到两条路径的最后一个公共节点
        TreeNode lca = null;
        int minLength = Math.min(pathToP.size(), pathToQ.size());
        
        for (int i = 0; i < minLength; i++) {
            if (pathToP.get(i) == pathToQ.get(i)) {
                lca = pathToP.get(i);
            } else {
                break;
            }
        }
        
        return lca;
    }
    
    /**
     * 找到从根节点到目标节点的路径
     */
    private boolean findPath(TreeNode root, TreeNode target, List<TreeNode> path) {
        if (root == null) {
            return false;
        }
        
        // 将当前节点加入路径
        path.add(root);
        
        // 如果找到目标节点，返回true
        if (root == target) {
            return true;
        }
        
        // 在左子树或右子树中查找
        if (findPath(root.left, target, path) || findPath(root.right, target, path)) {
            return true;
        }
        
        // 如果在当前路径上没找到，移除当前节点
        path.remove(path.size() - 1);
        return false;
    }
}
```

## 5. 测试用例和预期结果

### 测试用例
```java
public class Test {
    public static void main(String[] args) {
        Solution solution = new Solution();
        
        // 构建测试用例1的二叉树
        //        3
        //       / \
        //      5   1
        //     / \ / \
        //    6  2 0  8
        //      / \
        //     7   4
        TreeNode root1 = new TreeNode(3);
        TreeNode node5 = new TreeNode(5);
        TreeNode node1 = new TreeNode(1);
        TreeNode node6 = new TreeNode(6);
        TreeNode node2 = new TreeNode(2);
        TreeNode node0 = new TreeNode(0);
        TreeNode node8 = new TreeNode(8);
        TreeNode node7 = new TreeNode(7);
        TreeNode node4 = new TreeNode(4);
        
        root1.left = node5;
        root1.right = node1;
        node5.left = node6;
        node5.right = node2;
        node1.left = node0;
        node1.right = node8;
        node2.left = node7;
        node2.right = node4;
        
        // 测试用例1：p=5, q=1
        TreeNode result1 = solution.lowestCommonAncestor(root1, node5, node1);
        System.out.println("测试用例1结果: " + result1.val); // 预期输出：3
        
        // 测试用例2：p=5, q=4
        TreeNode result2 = solution.lowestCommonAncestor(root1, node5, node4);
        System.out.println("测试用例2结果: " + result2.val); // 预期输出：5
        
        // 测试用例3：p=6, q=7
        TreeNode result3 = solution.lowestCommonAncestor(root1, node6, node7);
        System.out.println("测试用例3结果: " + result3.val); // 预期输出：5
        
        // 构建测试用例4的简单二叉树
        //    1
        //   /
        //  2
        TreeNode root2 = new TreeNode(1);
        TreeNode node2_2 = new TreeNode(2);
        root2.left = node2_2;
        
        // 测试用例4：p=1, q=2
        TreeNode result4 = solution.lowestCommonAncestor(root2, root2, node2_2);
        System.out.println("测试用例4结果: " + result4.val); // 预期输出：1
        
        // 测试用例5：单节点树
        TreeNode root3 = new TreeNode(1);
        TreeNode result5 = solution.lowestCommonAncestor(root3, root3, root3);
        System.out.println("测试用例5结果: " + result5.val); // 预期输出：1
    }
}
```

## 6. 边界情况处理

### 关键边界情况
1. **空树**：root为null
2. **单节点树**：只有根节点
3. **p或q为根节点**：其中一个目标节点就是根节点
4. **p是q的祖先**：一个节点是另一个节点的祖先
5. **p和q相同**：两个目标节点是同一个节点

### 边界处理技巧
```java
// 输入验证
if (root == null || p == null || q == null) {
    return null;
}

// 特殊情况：p和q相同
if (p == q) {
    return p;
}

// 递归基础情况
if (root == null) {
    return null;
}
if (root == p || root == q) {
    return root;
}
```

## 7. 相关题目

### 类似题目
1. **235. 二叉搜索树的最近公共祖先**：BST的特殊性质
2. **1644. 二叉树的最近公共祖先 II**：节点可能不存在
3. **1650. 二叉树的最近公共祖先 III**：有父指针的情况
4. **1676. 二叉树的最近公共祖先 IV**：多个节点的最近公共祖先
5. **865. 具有所有最深节点的最小子树**：最深节点的最近公共祖先

### 题目关联
- 都涉及树的遍历和祖先关系
- 递归思想的应用
- 二叉搜索树版本可以利用BST性质优化
- 有父指针的版本可以简化为链表相交问题

## 8. 复杂度分析

### 时间复杂度
- **递归法**：O(N)，其中N是二叉树的节点数，最坏情况下需要访问所有节点
- **存储父节点法**：O(N)，需要遍历整棵树
- **路径记录法**：O(N)，需要两次DFS遍历

### 空间复杂度
- **递归法**：O(H)，其中H是树的高度，递归栈的深度
- **存储父节点法**：O(N)，需要存储所有节点的父节点信息
- **路径记录法**：O(H)，需要存储从根到目标节点的路径

### 最优解选择
推荐使用**递归法**，因为：
- 时间复杂度最优：O(N)
- 空间复杂度较好：O(H)
- 代码最简洁
- 最容易理解和记忆

## 9. 面试要点总结

### 核心考点
1. **递归思维**：如何设计递归函数
2. **返回值含义**：递归函数返回值的设计
3. **边界条件**：递归的终止条件
4. **后序遍历**：从子树向父节点传递信息

### 面试回答要点
1. **问题理解**：明确最近公共祖先的定义
2. **方法选择**：推荐递归法，解释原因
3. **算法思路**：后序遍历，根据子树返回结果判断
4. **代码实现**：写出清晰的递归代码
5. **复杂度分析**：分析时间和空间复杂度

### 常见面试问题
- 为什么使用后序遍历？
- 递归函数的返回值代表什么？
- 如何处理p或q不存在的情况？
- 能否优化空间复杂度？

## 10. 性能优化技巧

### 优化策略
1. **提前终止**：找到答案后立即返回
2. **路径压缩**：在某些变种中可以应用
3. **迭代实现**：避免递归栈溢出
4. **缓存结果**：对于重复查询可以缓存结果

### 迭代实现（避免递归）
```java
import java.util.*;

public class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        Stack<TreeNode> stack = new Stack<>();
        Map<TreeNode, TreeNode> parent = new HashMap<>();
        
        parent.put(root, null);
        stack.push(root);
        
        // 遍历直到找到p和q的父节点
        while (!parent.containsKey(p) || !parent.containsKey(q)) {
            TreeNode node = stack.pop();
            
            if (node.left != null) {
                parent.put(node.left, node);
                stack.push(node.left);
            }
            
            if (node.right != null) {
                parent.put(node.right, node);
                stack.push(node.right);
            }
        }
        
        // 收集p的所有祖先
        Set<TreeNode> ancestors = new HashSet<>();
        while (p != null) {
            ancestors.add(p);
            p = parent.get(p);
        }
        
        // 找到q的第一个在ancestors中的祖先
        while (!ancestors.contains(q)) {
            q = parent.get(q);
        }
        
        return q;
    }
}
```

### 针对BST的优化
```java
// 如果是二叉搜索树，可以利用BST性质
public TreeNode lowestCommonAncestorBST(TreeNode root, TreeNode p, TreeNode q) {
    while (root != null) {
        if (p.val < root.val && q.val < root.val) {
            root = root.left;
        } else if (p.val > root.val && q.val > root.val) {
            root = root.right;
        } else {
            return root;
        }
    }
    return null;
}
```

### 性能对比
| 方法 | 时间复杂度 | 空间复杂度 | 代码复杂度 | 推荐度 |
|------|------------|------------|------------|--------|
| 递归法 | O(N) | O(H) | 简单 | ⭐⭐⭐⭐⭐ |
| 存储父节点 | O(N) | O(N) | 中等 | ⭐⭐⭐ |
| 路径记录 | O(N) | O(H) | 中等 | ⭐⭐⭐ |
| 迭代实现 | O(N) | O(N) | 复杂 | ⭐⭐ |

## 总结

二叉树的最近公共祖先是树形结构中的经典问题，核心在于：
1. 理解祖先关系和递归思维
2. 掌握后序遍历的应用
3. 正确设计递归函数的返回值
4. 处理各种边界情况

这道题考查的是对树结构和递归的深入理解，是很多树形问题的基础。掌握了这道题的递归解法，就能更好地理解树的遍历和信息传递机制。建议重点掌握递归解法，并理解其背后的思想。