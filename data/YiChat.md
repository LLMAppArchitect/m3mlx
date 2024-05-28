下面是一个简单的DAG任务调度器的Golang实现示例。在这个例子中，我们将创建一个拓扑排序的算法来确定任务的执行顺序，并在有向无环图（DAG）中避免循环。

```go
package main

import (
        "container/list"
        "fmt"
)

// Task表示单个任务的结构体
type Task struct {
    name string // 任务名称
    dep []string // 该任务的所有依赖项
}

// Node用于在图中存储Task结构体的简单封装
type Node struct {
    task *Task
}

// Graph是整个DAG结构的容器
type Graph map[*Node]bool

func (t *Task) hasDependencies() bool {
    return len(t.dep) > 0
}

// TopoSort进行拓扑排序，返回结果数组
func TopoSort(graph Graph, result chan<- [][]string) {
    inDegree := make(map[*Node]int)  // 记录每个节点的入度
    zeroInDegreeNodesList := list.New() // 将所有入度为0的节点加入链表

    for _, node := range graph {
        if node.task.hasDependencies() {
            inDegree[node] = len(node.task.dep)
        } else {
            inDegree[node] = 1
            zeroInDegreeNodesList.PushBack(node)
        }
    }

    var sortedOrder []string
    removedNodes := make(map[*Node]bool)

    for zeroInDegreeNodesList.Len() > 0 {
        currentNode := zeroInDegreeNodesList.Front().Value.(*Node)
        sortedOrder = append(sortedOrder, currentNode.task.name)
        removedNodes[currentNode] = true
        zeroInDegreeNodesList.Remove(zeroInDegreeNodesList.Front())

        for i := range inDegree {
            if contains(i.task.dep, currentNode.task.name) {
                inDegree[i]--
                if removedNodes[i] == false && inDegree[i] <= 0 {
                    zeroInDegreeNodesList.PushBack(i)
                }
            }
        }
    }

    result <- [](sortedOrder)
}

// check是否包含某个元素
func contains(slice []string, itemToFind string) bool {
    for _, a := range slice {
        if a == itemToFind {
            return true
        }
    }
    return false
}

func buildGraphWithDep(tasks []*Task) Graph {
    graph := make(map[[2]*Task]bool)
    dependenciesMap := make(map[string][]string)

    for index, t := range tasks {
        dependenciesMap[t.name] = t.dep
    }

    for key, deps := range dependenciesMap {
        graph[key+deps] = true
    }

    return graph
}

func topoSortMain(tasks []*Task) [][]string {
    graph := buildGraphWithDep(tasks)
    resultCh := make([][]string, 0)

    TopoSort(graph, resultCh)

    return resultCh
}

func createTasks() []*Task {
    a := &Task{"A", nil}
    b := &Task{"B", []string{"A"}}
    c := &Task{"C", []string{"A", "B"}}
    d := &Task{"D", []string{"E"}}
    e := &Task{"E", []string{"A"}}
    f := &Task{"F", []string{}}
    g := &Task{"G", []string{"D", "F"}}

    allTasks := []*Task{}
    allTasks = append(allTasks, a, b, c, d, e, f, g)

    return allTasks
}

func main() {
    allTasks := createTasks()

    resChannels := topoSortMain(allTasks)

    for resChannelIdx := range resChannels {
        fmt.Println(resChannels[resChannelIdx])
    }
}
```

这段代码首先定义了`Task`和`Node`结构体以及`Graph`类型，然后实现了以下函数：

- `hasDependencies()`检查当前任务是否有未完成的依赖。
- `TopoSort()`使用Kahn算法实现拓扑排序。
- `contains()`是一个辅助函数，用来判断一个给定的值是否存在于列表中。
- `buildGraphWithDep()`构建任务的图并考虑其依赖关系。
- `topoSortMain()`是主调度器入口点，它调用其他函数来执行实际的任务调度逻辑。

最后，我们在`main()`函数中创建了一系列依赖于彼此的任务（以字母表示），并将它们传递到`topoSortMain()`函数中以获取它们的正确执行顺序。

请注意，这个例子中的DAG非常简单，实际的DAG可能具有更复杂的结构和更多的边缘。此外，这里的实现是为了演示目的而简化的；在生产环境中，你可能需要处理错误恢复、取消机制和其他高级功能。