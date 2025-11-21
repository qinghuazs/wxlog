---
title: 有效的括号
date: 2025-09-19
categories:
  - Algorithm
  - LeetCode
---

## 1. 问题描述和示例

### 问题描述
给定一个只包含 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s`，判断字符串是否有效。

有效字符串需满足：
1. 左括号必须用相同类型的右括号闭合
2. 左括号必须以正确的顺序闭合
3. 每个右括号都有一个对应的相同类型的左括号

### 示例
```
示例 1:
输入: s = "()"
输出: true

示例 2:
输入: s = "()[]{}"
输出: true

示例 3:
输入: s = "(]"
输出: false

示例 4:
输入: s = "([)]"
输出: false

示例 5:
输入: s = "{[]}"
输出: true
```

## 2. 核心难点分析

### 主要挑战
1. **匹配顺序**: 括号必须按照正确的顺序匹配，不能交叉
2. **类型匹配**: 左右括号必须是同一类型
3. **数量平衡**: 左右括号数量必须相等
4. **嵌套处理**: 需要正确处理括号的嵌套关系

### 关键观察
- 遇到左括号时，需要记住它，等待匹配的右括号
- 遇到右括号时，需要与最近的未匹配左括号进行匹配
- 这种"后进先出"的特性天然适合用栈来解决

## 3. 解题思路（栈的应用）

### 核心思想
使用栈来存储待匹配的左括号：
1. 遍历字符串中的每个字符
2. 遇到左括号时，将其压入栈中
3. 遇到右括号时，检查栈是否为空，以及栈顶元素是否与当前右括号匹配
4. 最终检查栈是否为空

### 算法步骤
1. 初始化一个空栈
2. 遍历字符串：
   - 如果是左括号 `'('`, `'{'`, `'['`，压入栈
   - 如果是右括号 `')'`, `'}'`, `']'`：
     - 检查栈是否为空（如果为空则不匹配）
     - 弹出栈顶元素，检查是否与当前右括号匹配
3. 遍历结束后，检查栈是否为空

## 4. Java代码实现

### 方法一：使用栈（推荐）
```java
import java.util.*;

public class ValidParentheses {
    
    /**
     * 判断括号字符串是否有效
     * 时间复杂度: O(n)
     * 空间复杂度: O(n)
     * 
     * @param s 输入的括号字符串
     * @return 如果括号有效返回true，否则返回false
     */
    public boolean isValid(String s) {
        // 边界情况：空字符串或长度为奇数
        if (s == null || s.length() % 2 == 1) {
            return false;
        }
        
        // 使用栈存储左括号
        Stack<Character> stack = new Stack<>();
        
        // 遍历字符串中的每个字符
        for (char c : s.toCharArray()) {
            // 遇到左括号，压入栈中
            if (c == '(' || c == '{' || c == '[') {
                stack.push(c);
            } 
            // 遇到右括号，进行匹配检查
            else if (c == ')' || c == '}' || c == ']') {
                // 栈为空说明没有对应的左括号
                if (stack.isEmpty()) {
                    return false;
                }
                
                // 弹出栈顶元素进行匹配
                char top = stack.pop();
                if (!isMatching(top, c)) {
                    return false;
                }
            }
        }
        
        // 最终栈应该为空，表示所有括号都已匹配
        return stack.isEmpty();
    }
    
    /**
     * 检查左右括号是否匹配
     * 
     * @param left 左括号
     * @param right 右括号
     * @return 如果匹配返回true，否则返回false
     */
    private boolean isMatching(char left, char right) {
        return (left == '(' && right == ')') ||
               (left == '{' && right == '}') ||
               (left == '[' && right == ']');
    }
}
```

### 方法二：使用HashMap优化
```java
import java.util.*;

public class ValidParenthesesOptimized {
    
    /**
     * 使用HashMap存储括号映射关系，代码更简洁
     * 时间复杂度: O(n)
     * 空间复杂度: O(n)
     */
    public boolean isValid(String s) {
        if (s == null || s.length() % 2 == 1) {
            return false;
        }
        
        // 创建括号映射表
        Map<Character, Character> map = new HashMap<>();
        map.put(')', '(');
        map.put('}', '{');
        map.put(']', '[');
        
        Stack<Character> stack = new Stack<>();
        
        for (char c : s.toCharArray()) {
            // 如果是右括号
            if (map.containsKey(c)) {
                // 检查栈是否为空或栈顶元素是否匹配
                if (stack.isEmpty() || stack.pop() != map.get(c)) {
                    return false;
                }
            } else {
                // 左括号直接入栈
                stack.push(c);
            }
        }
        
        return stack.isEmpty();
    }
}
```

### 方法三：字符串替换法（不推荐，仅作了解）
```java
public class ValidParenthesesReplace {
    
    /**
     * 通过不断替换成对的括号来判断
     * 时间复杂度: O(n²)
     * 空间复杂度: O(n)
     * 注意：此方法效率较低，不推荐使用
     */
    public boolean isValid(String s) {
        if (s == null || s.length() % 2 == 1) {
            return false;
        }
        
        // 不断替换成对的括号
        while (s.contains("()") || s.contains("{}") || s.contains("[]")) {
            s = s.replace("()", "")
                 .replace("{}", "")
                 .replace("[]", "");
        }
        
        return s.isEmpty();
    }
}
```

## 5. 复杂度分析

### 栈方法（推荐）
- **时间复杂度**: O(n)，其中n是字符串长度，需要遍历字符串一次
- **空间复杂度**: O(n)，最坏情况下栈中存储所有的左括号

### HashMap优化方法
- **时间复杂度**: O(n)，遍历字符串一次，HashMap操作为O(1)
- **空间复杂度**: O(n)，栈空间 + HashMap空间

### 字符串替换方法
- **时间复杂度**: O(n²)，每次替换可能需要O(n)时间，最多替换n/2次
- **空间复杂度**: O(n)，字符串操作需要额外空间

## 6. 测试用例和边界情况

```java
public class TestValidParentheses {
    
    public static void main(String[] args) {
        ValidParentheses solution = new ValidParentheses();
        
        // 测试用例
        testCase(solution, "", true, "空字符串");
        testCase(solution, "()", true, "简单括号");
        testCase(solution, "()[]{}", true, "多种括号");
        testCase(solution, "(]", false, "类型不匹配");
        testCase(solution, "([)]", false, "顺序错误");
        testCase(solution, "{[]}", true, "嵌套括号");
        testCase(solution, "(((", false, "只有左括号");
        testCase(solution, ")))", false, "只有右括号");
        testCase(solution, "()()()"，true, "连续括号");
        testCase(solution, "(()())", true, "复杂嵌套");
        testCase(solution, "(", false, "单个左括号");
        testCase(solution, ")", false, "单个右括号");
        testCase(solution, "([{}])", true, "多层嵌套");
    }
    
    private static void testCase(ValidParentheses solution, String input, 
                                boolean expected, String description) {
        boolean result = solution.isValid(input);
        System.out.printf("%s: 输入='%s', 期望=%b, 实际=%b, %s%n", 
                         description, input, expected, result, 
                         result == expected ? "✓" : "✗");
    }
}
```

### 边界情况总结
1. **空字符串**: 应返回true
2. **单个字符**: 任何单个括号都应返回false
3. **奇数长度**: 必然无法完全匹配，直接返回false
4. **只有左括号**: 栈不为空，返回false
5. **只有右括号**: 栈为空时遇到右括号，返回false
6. **类型不匹配**: 如"(]"，返回false

## 7. 相关变种问题

### 7.1 最长有效括号 (LeetCode 32)
```java
/**
 * 找出最长有效括号子串的长度
 * 输入: "(()"
 * 输出: 2
 * 解释: 最长有效括号子串是 "()"
 */
public int longestValidParentheses(String s) {
    // 使用动态规划或栈来解决
}
```

### 7.2 删除无效的括号 (LeetCode 301)
```java
/**
 * 删除最少的括号使字符串有效
 * 输入: "()())"
 * 输出: ["()()", "(())"]
 */
public List<String> removeInvalidParentheses(String s) {
    // 使用BFS或DFS来解决
}
```

### 7.3 括号生成 (LeetCode 22)
```java
/**
 * 生成所有可能的有效括号组合
 * 输入: n = 3
 * 输出: ["((()))","(()())","(())()","()(())","()()()"]
 */
public List<String> generateParenthesis(int n) {
    // 使用回溯算法
}
```

## 8. 面试要点总结

### 8.1 关键考点
1. **数据结构选择**: 为什么选择栈？栈的LIFO特性与括号匹配的天然契合
2. **边界处理**: 空字符串、奇数长度、单个字符等边界情况
3. **算法优化**: 从基础实现到HashMap优化的思路
4. **代码质量**: 清晰的变量命名、适当的注释、模块化设计

### 8.2 常见面试问题
**Q: 为什么使用栈而不是其他数据结构？**
A: 括号匹配需要"最近匹配"原则，栈的LIFO特性完美符合这个需求。

**Q: 如何处理多种类型的括号？**
A: 可以使用HashMap建立映射关系，或者使用if-else判断，HashMap方式更简洁。

**Q: 空间复杂度能否优化到O(1)？**
A: 对于一般情况不行，因为需要存储未匹配的左括号。但对于特定情况（如只有一种括号），可以用计数器。

### 8.3 扩展思考
- 如果括号种类很多怎么办？
- 如何处理嵌套层数限制？
- 在流式数据中如何实时判断？

## 9. 性能优化技巧

### 9.1 早期终止优化
```java
public boolean isValidOptimized(String s) {
    if (s == null || s.length() % 2 == 1) {
        return false; // 奇数长度直接返回false
    }
    
    Stack<Character> stack = new Stack<>();
    int maxSize = s.length() / 2; // 栈的最大可能大小
    
    for (char c : s.toCharArray()) {
        if (c == '(' || c == '{' || c == '[') {
            stack.push(c);
            // 如果栈大小超过一半，必然无法匹配
            if (stack.size() > maxSize) {
                return false;
            }
        } else {
            if (stack.isEmpty() || !isMatching(stack.pop(), c)) {
                return false;
            }
        }
    }
    
    return stack.isEmpty();
}
```

### 9.2 使用数组模拟栈
```java
public boolean isValidArray(String s) {
    if (s == null || s.length() % 2 == 1) {
        return false;
    }
    
    char[] stack = new char[s.length()];
    int top = -1;
    
    for (char c : s.toCharArray()) {
        if (c == '(' || c == '{' || c == '[') {
            stack[++top] = c;
        } else {
            if (top == -1 || !isMatching(stack[top--], c)) {
                return false;
            }
        }
    }
    
    return top == -1;
}
```

### 9.3 位运算优化（仅适用于单一括号类型）
```java
/**
 * 仅适用于只有 '(' 和 ')' 的情况
 * 使用计数器代替栈
 */
public boolean isValidSingleType(String s) {
    int count = 0;
    for (char c : s.toCharArray()) {
        if (c == '(') {
            count++;
        } else if (c == ')') {
            count--;
            if (count < 0) {
                return false; // 右括号多于左括号
            }
        }
    }
    return count == 0;
}
```

## 10. 实际应用场景

### 10.1 编译器和解释器
- **语法分析**: 检查代码中的括号、大括号、方括号是否匹配
- **表达式求值**: 数学表达式中的括号优先级处理
- **JSON解析**: 验证JSON格式的大括号和方括号匹配

### 10.2 文本编辑器
- **括号高亮**: 实时检查和高亮匹配的括号对
- **代码折叠**: 根据括号匹配确定代码块边界
- **自动补全**: 输入左括号时自动补全右括号

### 10.3 数据验证
- **配置文件验证**: 检查XML、JSON等配置文件格式
- **SQL语句验证**: 检查SQL中的括号匹配
- **正则表达式验证**: 验证正则表达式中的分组括号

### 10.4 实际代码示例
```java
/**
 * 简单的表达式验证器
 */
public class ExpressionValidator {
    
    public boolean isValidExpression(String expression) {
        // 移除所有非括号字符，只保留括号进行验证
        String brackets = expression.replaceAll("[^(){}\\[\\]]", "");
        return isValid(brackets);
    }
    
    // 使用之前实现的isValid方法
    private boolean isValid(String s) {
        // ... 实现代码
    }
    
    public static void main(String[] args) {
        ExpressionValidator validator = new ExpressionValidator();
        
        System.out.println(validator.isValidExpression("(a + b) * [c - d]"));  // true
        System.out.println(validator.isValidExpression("func(arr[i], {x: y})"));  // true
        System.out.println(validator.isValidExpression("if (x > 0) { return arr[x]; }"));  // true
        System.out.println(validator.isValidExpression("(a + b] * c"));  // false
    }
}
```

---

## 总结

有效的括号问题是栈数据结构的经典应用，体现了"后进先出"原则在实际问题中的巧妙运用。通过这个问题，我们学习到：

1. **算法思维**: 识别问题特征，选择合适的数据结构
2. **代码实现**: 从基础版本到优化版本的演进过程
3. **边界处理**: 全面考虑各种边界情况和异常输入
4. **性能优化**: 多种优化策略的应用
5. **实际应用**: 算法在真实场景中的广泛应用

掌握这个问题不仅有助于面试，更重要的是培养了解决类似问题的思维模式和实现能力。