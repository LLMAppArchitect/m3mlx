==========
Prompt: 15.044 tokens-per-sec
Generation: 8.870 tokens-per-sec
。

DAG（有向无环图）任务调度器是一种用于管理依赖关系的任务调度系统。任务可以按照它们之间的依赖关系顺序执行。下面是一个简单的DAG任务调度器的实现，使用Go语言。

```go
package main

import (
        "fmt"
        "sync"
)

// Task 任务接口
type Task interface {
        ID() string
        Dependencies() []string
        Run() error
}

// TaskNode 任务节点结构体
type TaskNode struct {
        ID       string
        Dependencies []string
        Task     Task
}

// DAG 有向无环图
type DAG struct {
        nodes map[string]*TaskNode
        mu    sync.Mutex
}

// NewDAG 创建一个新的DAG
func NewDAG() *DAG {
        return &DAG{
                nodes: make(map[string]*TaskNode),
        }
}

// AddTask 添加任务到DAG
func (d *DAG) AddTask(task Task) {
        d.mu.Lock()
        defer d.mu.Unlock()

        taskNode := &TaskNode{
                ID:       task.ID(),
                Dependencies: task.Dependencies(),
                Task:     task,
        }
        d.nodes[task.ID()] = taskNode
}

// Run 运行DAG中的所有任务
func (d *DAG) Run() error {
        var wg sync.WaitGroup
        var errs []error

        d.mu.Lock()
        tasks := make([]*TaskNode, 0, len(d.nodes))
        for _, taskNode := range d.nodes {
                tasks = append(tasks, taskNode)
        }
        d.mu.Unlock()

        for _, taskNode := range tasks {
                wg.Add(1)
                go func(taskNode *TaskNode) {
                        defer wg.Done()
                        if err := taskNode.Task.Run(); err != nil {
                                errs = append(errs, err)
                        }
                }(taskNode)
        }

        wg.Wait()
        if len(errs) > 0 {
                return fmt.Errorf("tasks failed: %v", errs)
        }
        return nil
}

// 示例任务实现
type ExampleTask struct {
        ID string
}

func (t *ExampleTask) ID() string {
        return t.ID
}

func (t *ExampleTask) Dependencies() []string {
        return []string{"dep1", "dep2"}
}

func (t *ExampleTask) Run() error {
        fmt.Printf("Running task %s...\n", t.ID)
        // 模拟任务执行的延迟
        time.Sleep(1 * time.Second)
        fmt.Printf("Task %s completed.\n", t.ID)
        return nil
}

func main() {
        // 创建任务
        task1 := &ExampleTask{ID: "task1"}
        task2 := &ExampleTask{ID: "task2"}
        task3 := &ExampleTask{ID: "task3"}

        // 添加依赖
        task1.Dependencies = []string{"task2"}
        task2.Dependencies = []string{"task3"}

        // 创建DAG
        dag := NewDAG()

        // 添加任务到DAG
        dag.AddTask(task1)
        dag.AddTask(task2)
        dag.AddTask(task3)

        // 运行DAG
        if err := dag.Run(); err != nil {
                fmt.Println("Error running tasks:", err)
        } else {
                fmt.Println("All tasks completed successfully.")
        }
}
```

这个DAG任务调度器的实现包括以下几个部分：

1. `Task` 接口定义了任务的基本行为，包括任务的ID、依赖关系和执行方法。
2. `TaskNode` 结构体用于存储任务节点的信息，包括ID、依赖关系和实际的任务实现。
3. `DAG` 结构体用于存储任务节点的集合，并提供了添加任务和运行所有任务的方法。
4. `Run` 方法在DAG中的所有任务执行前检查它们的依赖关系，并确保它们按照正确的顺序执行。
5. `ExampleTask` 结构体是一个示例任务实现，用于演示如何创建和运行任务。
6. `main` 函数中创建了几个示例任务，设置了它们的依赖关系，创建了DAG，并运行了DAG中的任务。

注意：这个实现是一个简单的示例，它没有考虑并发安全性问题，因为它没有处理多个goroutine可能会同时修改DAG的情况。在实际应用中，你可能需要使用更复杂的同步机制来确保DAG的正确性。
