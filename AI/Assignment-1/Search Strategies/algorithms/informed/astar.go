package informed

import (
	"container/heap"
	"search-strategies/core"
	"time"
)

// AStar implements the A* (A-Star) search algorithm.
//
// Strategy:
// - Expands nodes based on f(n) = g(n) + h(n), where:
//   - g(n) = actual cost from start node to current node
//   - h(n) = estimated cost from current node to goal
//
// - Balances between actual cost and heuristic estimate
// - Uses a priority queue ordered by f(n) value
//
// Characteristics:
// - Completeness: Yes (guaranteed to find a solution if one exists)
// - Optimality: Yes (guaranteed to find optimal path with admissible heuristic)
// - Admissible Heuristics:
//   - h1 (Misplaced Tiles): Never overestimates actual moves needed
//   - h2 (Manhattan Distance): Also admissible; generally more informed than h1
//
// - Time Complexity: O(b^d) - explores fewer nodes than BFS
// - Space Complexity: O(b^d) - must store nodes in priority queue
// - Advantages: Optimal, efficient, better than both BFS and Greedy
// - Disadvantage: Memory usage can be high for large search spaces
//
// Heuristic Functions:
// - h1: Counts the number of tiles not in their goal position (excluding blank)
// - h2: Manhattan distance = sum of |current_x - goal_x| + |current_y - goal_y| for each tile
//
// Parameters:
//   - initial: Starting state of the puzzle
//   - heuristicType: "h1" for misplaced tiles, "h2" for Manhattan distance
//
// Returns: SearchResult containing optimal path, states explored, and timing information
func AStar(initial *core.State, heuristicType string) *core.SearchResult {
	start := time.Now()

	// Initialize the starting state with cost and heuristic values
	initial.H = initial.H2() // Default to Manhattan distance
	heuristicName := "Manhattan Distance (h2)"
	if heuristicType == "h1" {
		initial.H = initial.H1()
		heuristicName = "Misplaced Tiles (h1)"
	}
	// g(n) = 0 for the start state (no moves yet)
	initial.G = 0
	// f(n) = g(n) + h(n) = 0 + h(start)
	initial.F = initial.G + initial.H

	// Create priority queue ordered by f(n) value
	pq := &core.PriorityQueue{initial}
	heap.Init(pq)

	// Track visited states to avoid revisiting and cycles
	visited := make(map[string]bool)
	statesExplored := 0

	// Main A* search loop
	for pq.Len() > 0 {
		// Extract node with lowest f(n) value from priority queue
		current := heap.Pop(pq).(*core.State)

		// Skip if already visited (duplicate elimination)
		if visited[current.Hash()] {
			continue
		}
		visited[current.Hash()] = true
		statesExplored++

		// Check if goal state reached
		if current.IsGoal() {
			return &core.SearchResult{
				Algorithm:      "A* Search",
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

			// Only process unvisited states
			if !visited[newState.Hash()] {
				// Calculate heuristic value for successor state
				if heuristicType == "h1" {
					newState.H = newState.H1() // Misplaced tiles heuristic
				} else {
					newState.H = newState.H2() // Manhattan distance heuristic
				}
				// Calculate f(n) = g(n) + h(n)
				// g(n) is already set by ApplyAction (increments parent's g value)
				newState.F = newState.G + newState.H
				heap.Push(pq, newState)
			}
		}
	}

	// No solution found (queue exhausted)
	return &core.SearchResult{
		Algorithm:      "A* Search",
		Success:        false,
		Heuristic:      heuristicName,
		StatesExplored: statesExplored,
		TimeTaken:      time.Since(start),
	}
}
