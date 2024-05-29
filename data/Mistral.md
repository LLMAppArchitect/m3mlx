==========
Prompt: 46.411 tokens-per-sec
Generation: 16.307 tokens-per-sec

下面是一个简单的 Directed Acyclic Graph (DAG) Task Scheduler 的 Go 语言实现。该程序允许你定义一系列依赖关系和任务，然后按照优先级（高优先级任务首先执行）和依赖关系进行调度。

```go
package main

import (
        "fmt"
        "sort"
)

type Node struct {
        name      string // node name
        priority  int    // priority of the node, higher value means earlier execution
        dependencies []string // dependencies for this node
}

// map from a dependency to its dependent nodes. This allows us to easily find all nodes that depend on a given dependency when adding new edges.
var depsMap = make(map[string][]*Node)

func addDependency(from *Node, to *Node) {
        to.dependencies = append(to.dependencies, from.name)
        depsMap[from.name] = append(depsMap[from.name], to)
}

// check if there is a cycle in the graph. If there's a cycle, return an error message and exit.
func hasCycle() bool {
        visited := make(map[string]bool)
        stack := []string{}

        for _, key := range depsMap {
                if !hasCycleHelper(key[0].name, visited, stack) {
                        return false
                }
        }
        return true
}

func hasCycleHelper(nodeName string, visited map[string]bool, stack []string) bool {
        if visited[nodeName] {
                // found a cycle
                return true
        }

        visited[nodeName] = true
        stack = append(stack, nodeName)

        for _, dep := range depsMap[nodeName] {
                if hasCycleHelper(dep.name, visited, stack) {
                        return true
                }
        }

        // pop current node from the stack
        last := len(stack) - 1
        stack = stack[:last]
        visited[nodeName] = false

        return false
}

func scheduleTasks() {
        // sort nodes based on their priorities
        nodes := make([]*Node, 0, len(depsMap))
        for _, v := range depsMap {
                for _, n := range v {
                        nodes = append(nodes, n)
                }
        }
        sort.Slice(nodes, func(i, j int) bool {
                return nodes[i].priority > nodes[j].priority
        })

        // execute tasks with no dependencies first
        executingNodes := make([]*Node, 0)
        for _, node := range nodes {
                if len(node.dependencies) == 0 {
                        executingNodes = append(executingNodes, node)
                }
        }

        for len(executingNodes) != 0 {
                current := executingNodes[len(executingNodes)-1]
                delete(depsMap, current.name)

                for i, other := range depsMap {
                        if contains(other, current) {
                                copy(depsMap[i][:], append(depsMap[i][:len(depsMap[i])-1], depsMap[i][len(depsMap[i])]))
                                delete(depsMap, i)
                                break
                        }
                }

                fmt.Printf("Executing %s\n", current.name)

                newDependencies := make([]string, 0)
                for _, dep := range current.dependencies {
                        found := false
                        for _, node := range executingNodes {
                                if node.name == dep && len(node.dependencies) == 0 {
                                        // add it back since we just executed a node that was depending on it
                                        executingNodes = append(executingNodes, node)
                                        found = true
                                        break
                                }
                        }
                        if !found {
                                newDependencies = append(newDependencies, dep)
                        }
                }
                current.dependencies = newDependencies

                executedNodes := append(executedNodes, current)
                executingNodes = executingNodes[:len(executingNodes)-1]
        }

        // check if there are any orphan nodes left (nodes without any dependencies but not yet executed)
        orphans := make([]*Node, 0)
        for _, v := range depsMap {
                for _, n := range v {
                        if !contains(executedNodes, n) {
                                orphans = append(orphans, n)
                        }
                }
        }

        if len(orphans) > 0 {
                fmt.Println("\nWarning! The following tasks have no dependencies but were not scheduled for execution due to cycles in the dependency graph.")
                for _, o := range orphans {
                        fmt.Printf("- %s\n", o.name)
                }
        }
}

func contains(slice []*Node, item *Node) bool {
    for _, s := range slice {
        if s.name == item.name {
            return true
        }
    }
    return false
}

func main() {
        tasks := []struct{ name string; priority int; dependencies []string }{
                {"taskA", 5, nil},
                {"taskB", 3, []string{"taskA"}},
                {"taskC", 2, []string{"taskB", "taskD"}},
                {"taskD", 4, []string{"taskE"}},
                {"taskE", 1, nil},
        }

        var nodes []*Node
        for _, task := range tasks {
                node := &Node{
                        name:      task.name,
                        priority:  task.priority,
                        dependencies: task.dependencies,
                }
                nodes = append(nodes, node)
        }

        for _, from := range nodes {
                for _, to := range from.dependencies {
                        addDependency(from, getNodeByName(to))
                }
        }

        if hasCycle() {
                panic("There is a cycle in the dependency graph. Please adjust your tasks and try again.")
        }

        scheduleTasks()
}

func getNodeByName(name string) *Node {
        for _, n := range nodes {
                if n.name == name {
                        return n
                }
        }
        return nil
}
```

该程序定义了 `Node` 结构体，用于表示 DAG 中的每个节点。它包含任务名称、优先级和依赖关系。然后有一个 map（`depsMap`）来存储所有边，方便我们找到某个节点的所有依赖项。在执行时会检查图是否存在环，如果存在则抛出错误。最终使用排序将任务按照高优先级和依赖顺序进行调度。

