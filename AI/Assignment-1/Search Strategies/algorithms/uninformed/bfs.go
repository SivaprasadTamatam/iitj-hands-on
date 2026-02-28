package uninformed

import (
	"search-strategies/core"
	"sync/atomic"
	"time"
)

/*
Breadth-First Search (BFS)
What it does:
BFS explores nodes level by level, starting from the initial state. It visits all neighbors at the current depth before moving deeper.

How your implementation works:

Queue-based exploration: Uses a simple array as a FIFO (First-In-First-Out) queue. The queue := []*core.State{initial} initializes it with the starting state.

Visited tracking: visited := make(map[string]bool) uses a hash map to track explored states and prevent cycles. Each state is identified by its hash.

Main loop:

Dequeues the first state: current := queue[0] then removes it: queue = queue[1:]
Checks if it's the goal state
If not, generates all possible actions and creates new states
Only adds unvisited states to the queue
Properties:

✅ Complete: Always finds a solution if one exists
✅ Optimal: Finds the shortest path (minimum steps) because it explores level by level
❌ Space-heavy: Stores all frontier nodes (can use lots of memory)
Example: In a maze, BFS explores all paths 1 step away, then all paths 2 steps away, etc., guaranteeing the shortest route.
*/
func BFS(initial *core.State) *core.SearchResult {
	start := time.Now()
	queue := []*core.State{initial}
	visited := make(map[string]bool)
	visited[initial.Hash()] = true
	statesExplored := int32(0)

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		atomic.AddInt32(&statesExplored, 1)

		if current.IsGoal() {
			return &core.SearchResult{
				Algorithm:      "Breadth-First Search (BFS)",
				Success:        true,
				Path:           current.GetPath(),
				PathLength:     len(current.GetPath()),
				StatesExplored: int(statesExplored),
				TimeTaken:      time.Since(start),
			}
		}

		for _, action := range current.GetActions() {
			newState := current.ApplyAction(action)
			hash := newState.Hash()
			if !visited[hash] {
				visited[hash] = true
				queue = append(queue, newState)
			}
		}
	}

	return &core.SearchResult{
		Algorithm:      "Breadth-First Search (BFS)",
		Success:        false,
		StatesExplored: int(statesExplored),
		TimeTaken:      time.Since(start),
	}

}
