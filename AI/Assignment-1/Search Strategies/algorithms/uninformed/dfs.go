package uninformed

import (
	"fmt"
	"search-strategies/core"
	"time"
)

/*
Depth-First Search (DFS)
What it does:
DFS goes deep into one branch before backtracking. It explores as far as possible along each branch before going back.

How your implementation works:

Recursive approach: Uses a nested recursive function dfsRecursive that explores one path as deeply as possible.

Depth limiting: if depth > maxDepth { return nil } prevents infinite loops by setting a maximum search depth.

Visited map with backtracking:

Marks nodes as visited: visited[current.Hash()] = true
Crucially, removes them after exploring: delete(visited, current.Hash())
This allows the same state to be revisited through different paths, which is important for depth-limited search
Recursive exploration:

For each action, applies it to create a new state
Recursively searches that branch
If a solution is found, returns immediately (no further searching)
If that branch fails, it continues with other branches (backtracking)
Properties:

⚠️ Incomplete without depth limit: Could get stuck in infinite loops
✅ With depth limit: Becomes complete (explores all states up to max depth)
❌ Not optimal: May find longer solutions before shorter ones
✅ Space-efficient: Only stores the current path (not all frontier nodes)
*/
func DFS(initial *core.State, maxDepth int) *core.SearchResult {
	start := time.Now()
	visited := make(map[string]bool)
	statesExplored := 0

	var dfsRecursive func(*core.State, int) *core.State
	dfsRecursive = func(current *core.State, depth int) *core.State {
		if depth > maxDepth {
			return nil
		}

		statesExplored++
		if current.IsGoal() {
			return current
		}

		visited[current.Hash()] = true

		for _, action := range current.GetActions() {
			newState := current.ApplyAction(action)
			hash := newState.Hash()
			if !visited[hash] {
				result := dfsRecursive(newState, depth+1)
				if result != nil {
					return result
				}
			}
		}

		delete(visited, current.Hash()) // Allow revisiting in different paths
		return nil
	}

	result := dfsRecursive(initial, 0)

	if result != nil {
		return &core.SearchResult{
			Algorithm:      "Depth-First Search (DFS)",
			Success:        true,
			Parameters:     fmt.Sprintf("Max Depth: %d", maxDepth),
			Path:           result.GetPath(),
			PathLength:     len(result.GetPath()),
			StatesExplored: statesExplored,
			TimeTaken:      time.Since(start),
		}
	}

	return &core.SearchResult{
		Algorithm:      "Depth-First Search (DFS)",
		Success:        false,
		Parameters:     fmt.Sprintf("Max Depth: %d", maxDepth),
		StatesExplored: statesExplored,
		TimeTaken:      time.Since(start),
	}
}
