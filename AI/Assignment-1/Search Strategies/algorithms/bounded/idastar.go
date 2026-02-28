package bounded

import (
	"math"
	"search-strategies/core"
	"time"
)

// IDAStar implements Iterative Deepening A* (IDA*) algorithm.
//
// Strategy:
// - Combines depth-first search with A* evaluation function
// - Uses a threshold-based approach instead of a priority queue
// - Process:
//  1. Initialize threshold = h(start)
//  2. Perform depth-first search, pruning when f(n) > threshold
//  3. If no solution found, set threshold = minimum f-value that exceeded previous threshold
//  4. Repeat steps 2-3 until solution is found
//
// Characteristics:
// - Completeness: Yes (guaranteed to find a solution if one exists)
// - Optimality: Yes (guaranteed optimal path with admissible heuristic)
// - Time Complexity: O(b^d) where b = branching factor, d = depth
// - Space Complexity: O(d) - only stores current path (much better than A*)
// - Memory-Bounded: Excellent - uses only O(depth) memory
// - Advantage: Optimal solution with minimal memory requirements
// - Disadvantage: Regenerates nodes multiple times across iterations
//
// Heuristic Functions:
// - h1 (Misplaced Tiles): Counts tiles not in goal position
// - h2 (Manhattan Distance): Sum of |current_x - goal_x| + |current_y - goal_y|
//
// Parameters:
//   - initial: Starting state of the puzzle
//   - heuristicType: "h1" for misplaced tiles, "h2" for Manhattan distance
//
// Returns: SearchResult containing optimal path, states explored, and timing information
func IDAStar(initial *core.State, heuristicType string) *core.SearchResult {
	start := time.Now()
	statesExplored := 0
	heuristicName := "Manhattan Distance (h2)"

	// Helper function to get heuristic value based on type
	getHeuristic := func(s *core.State) int {
		if heuristicType == "h1" {
			heuristicName = "Misplaced Tiles (h1)"
			return s.H1()
		}
		return s.H2()
	}

	// Initialize threshold with heuristic of start state
	threshold := getHeuristic(initial)
	// Maintain current search path to detect cycles
	path := []*core.State{initial}

	// Recursive depth-first search with f(n) threshold pruning
	// Returns: (minimum f-value exceeding threshold, goal state if found)
	var search func([]*core.State, int, int) (int, *core.State)
	search = func(path []*core.State, g int, threshold int) (int, *core.State) {
		// Get current node (last in path)
		current := path[len(path)-1]
		statesExplored++

		// Calculate f(n) = g(n) + h(n)
		// g(n) = actual cost from start to current
		// h(n) = estimated cost from current to goal
		f := g + getHeuristic(current)

		// Prune: if f(n) exceeds threshold, return f value for next iteration
		if f > threshold {
			return f, nil
		}

		// Goal test: check if current state is goal
		if current.IsGoal() {
			return -1, current // Return -1 to signal solution found
		}

		// Track minimum f-value that exceeded threshold (for next iteration)
		min := math.MaxInt32

		// Explore all successors (expand current node)
		for _, action := range current.GetActions() {
			newState := current.ApplyAction(action)

			// Cycle detection: verify state is not already in path
			inPath := false
			for _, s := range path {
				if s.Hash() == newState.Hash() {
					inPath = true
					break
				}
			}
			if inPath {
				continue // Skip if already visited in this path
			}

			// Recursively search with updated path and cost
			path = append(path, newState)
			t, result := search(path, g+1, threshold) // g+1: cost increases by 1 per move

			if result != nil {
				return -1, result // Solution found
			}

			// Track minimum f-value for threshold in next iteration
			if t < min {
				min = t
			}

			path = path[:len(path)-1] // Backtrack: remove from path
		}

		return min, nil // No solution at this threshold
	}

	// Iterative deepening: repeat search with increasing thresholds
	// Each iteration explores deeper until solution is found
	for threshold != math.MaxInt32 {
		// Perform depth-first search with current threshold
		t, result := search(path, 0, threshold)

		// If solution found, return result
		if result != nil {
			return &core.SearchResult{
				Algorithm:      "Iterative Deepening A* (IDA*)",
				Success:        true,
				Heuristic:      heuristicName,
				Path:           result.GetPath(),
				PathLength:     len(result.GetPath()),
				StatesExplored: statesExplored,
				TimeTaken:      time.Since(start),
			}
		}

		// If no valid f-value found, no solution exists
		if t == math.MaxInt32 {
			break
		}

		// Set threshold to minimum f-value that exceeded previous threshold
		threshold = t
	}

	// No solution found after exhausting all thresholds
	return &core.SearchResult{
		Algorithm:      "Iterative Deepening A* (IDA*)",
		Success:        false,
		Heuristic:      heuristicName,
		StatesExplored: statesExplored,
		TimeTaken:      time.Since(start),
	}
}
