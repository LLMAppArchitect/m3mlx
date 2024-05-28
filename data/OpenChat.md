下面是一个简单的DAG任务调度器的Go源码，包括创建、添加子节点和执行任务。请注意，这个实现可能不完全适合生产环境，但可以作为参考。

```go
package main

import (
        "fmt"
)

// 定义结点类型为结构体
type Node struct {
        ID       int         // 结点ID
        Name     string      // 结点名称
        TaskFunc func() bool // 任务函数
        Children []*Node     // 子结点列表
}

// 创建新的结点
func NewNode(id, name string) *Node {
        return &Node{
                ID:    id,
                Name:  name,
                TaskFunc: nil,
        }
}

// 为结点分配任务函数
func (n *Node) SetTask(taskFunc func() bool) {
        n.TaskFunc = taskFunc
}

// 添加子结点
func (n *Node) AddChild(child *Node) {
        if n.Children == nil {
                n.Children = make([]*Node, 0)
        }
        n.Children = append(n.Children, child)
}

// 执行所有叶子结点的任务
func Schedule(node *Node) {
        if node!= nil && len(node.Children) == 0 { // 判断是否是叶子结点
                fmt.Printf("Executing task for %s\n", node.Name)
                node.TaskFunc()
        } else if node!= nil {
                for _, child := range node.Children { // 遍历子结点并递归调用schedule方法
                        Schedule(child)
                }
        }
}

func main() {
        root := NewNode(1, "root")
        child1 := NewNode(2, "child1")
        child2 := NewNode(3, "child2")

        root.SetTask(func() bool { return true })             // 设置根结点任务函数
        child1.AddChild(NewNode(4, "grandchild1"))          // 给子结点添加孙子结点
        child2.SetTask(func() bool { fmt.Println("child2 done"); return false })

        root.AddChild(child1)
        root.AddChild(child2)

        Schedule(root) // 调用schedule方法开始执行任务链
}
```

在上述代码中，我们首先定义了`Node`结构体，它表示一个有向无环图（DAG）中的结点，每个结点具有一个唯一的 ID、一 个名字、一 个任务函数以及一组子结点。在我们的例子中，任务函数返回一个布尔值，用于表示任务完成情况。

我们还提供了一些辅助函数来操作这些结点，例如：

- `NewNode`: 用于创建新结点。
- `SetTask`: 将任务函数设置到特定结点。
- `AddChild`: 向某个结点添加子结点。

最后，在主程序中，我们构建了一个简单的任务链，其中 root 是父节点， child1 和 child2 是其子节点， grandchild1 是子节点的孙子节点。然后我们分别为它们设置任务函数，并通过调用`Schedule`函数启动整个过程。这将遍历任务链，从叶子节点开始，然后依次处理每个父母节点。

总之，本示例展示了如何使用 Go 编写一个简单的 DAG 任务调度器。需要注意的是，该实现可能无法满足严格的生产环境要求，但它确实可以帮助你了解如何解决相关问题。你也许会发现，为了使其更健壮，你可能需要考虑引入一些错误检查和恢复策略。
