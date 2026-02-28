package informed

import (
	"container/heap"
	"search-strategies/core"
	"time"
)

// GreedyBFS implements Greedy Best-First Search algorithm.
//
// Strategy:
// - Expands the node with the lowest h(n) value (estimated cost to goal)
// - Ignores the actual cost g(n) from start to current node
// - Uses a priority queue ordered by heuristic value
//
// Characteristics:
// - Completeness: No (can get stuck in loops)
// - Optimality: No (ignores path cost, only considers heuristic estimate)
// - Time Complexity: O(b^m) - potentially explores many nodes
// - Space Complexity: O(b^m) - stores nodes in priority queue
// - Advantage: Generally faster than uninformed search
// - Disadvantage: May find suboptimal paths
//
// Parameters:
//   - initial: Starting state of the puzzle
//   - heuristicType: "h1" for misplaced tiles, "h2" for Manhattan distance
//
// Returns: SearchResult containing path, states explored, and timing information
func GreedyBFS(initial *core.State, heuristicType string) *core.SearchResult {
	start := time.Now()

	// Initialize the starting state with heuristic value
	initial.H = initial.H2() // Default to Manhattan distance
	heuristicName := "Manhattan Distance (h2)"
	if heuristicType == "h1" {
		initial.H = initial.H1()
		heuristicName = "Misplaced Tiles (h1)"
	}
	// For Greedy search, f(n) = h(n) only (no cost component)
	initial.F = initial.H

	// Create priority queue and add initial state
	pq := &core.PriorityQueue{initial}
	heap.Init(pq)

	// Track visited states to avoid cycles
	visited := make(map[string]bool)
	statesExplored := 0

	// Main search loop
	for pq.Len() > 0 {
		// Extract state with lowest h(n) from priority queue
		current := heap.Pop(pq).(*core.State)

		// Skip if already visited (handles duplicate detection)
		if visited[current.Hash()] {
			continue
		}
		visited[current.Hash()] = true
		statesExplored++

		// Check if goal state reached
		if current.IsGoal() {
			return &core.SearchResult{
				Algorithm:      "Greedy Best-First Search",
				Success:        true,
				Heuristic:      heuristicName,
				Path:           current.GetPath(),
				PathLength:     len(current.GetPath()),
				StatesExplored: statesExplored,
				TimeTaken:      time.Since(start),
			}
		}

		// Expand current state by generating all successor states
		for _, action := range current.GetActions() {
			newState := current.ApplyAction(action)

			// Only add unvisited states to the queue
			if !visited[newState.Hash()] {
				// Calculate heuristic value based on selected heuristic
				if heuristicType == "h1" {
					newState.H = newState.H1() // Count misplaced tiles
				} else {
					newState.H = newState.H2() // Calculate Manhattan distance
				}
				// Set f(n) = h(n) for greedy best-first
				newState.F = newState.H
				heap.Push(pq, newState)
			}
		}
	}

	// No solution found
	return &core.SearchResult{
		Algorithm:      "Greedy Best-First Search",
		Success:        false,
		Heuristic:      heuristicName,
		StatesExplored: statesExplored,
		TimeTaken:      time.Since(start),
	}
}
