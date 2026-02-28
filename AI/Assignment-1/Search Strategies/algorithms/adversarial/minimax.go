package adversarial

import (
	"math"
	"search-strategies/core"
	"time"
)

// Minimax algorithm - Solves two-player adversarial game
//
// PROBLEM CONTEXT (Assignment D):
// - Robotic sorter (MAX player) vs. System glitch (MIN player)
// - Both players alternate moves on 3x3 manuscript sorting puzzle
// - MAX wants to reach goal state; MIN wants to prevent it
//
// GAME FORMULATION:
// Game States: 3x3 board configurations (same as puzzle states)
// Players: MAX (robotic sorter) and MIN (system glitch)
// Actions: UP, DOWN, LEFT, RIGHT (manuscript movements)
// Terminal Conditions:
//  1. Goal state [1,2,3,4,5,6,7,8,B] reached
//  2. Maximum depth reached
//  3. No legal moves available
//
// UTILITY FUNCTION:
//   - For MAX player: U(s) = -Manhattan_Distance(s, goal) when depth=0
//     = +∞ when goal reached
//   - For MIN player: U(s) = +Manhattan_Distance(s, goal) when depth=0
//     = -∞ when goal reached
//   - Manhattan distance: sum of absolute differences in row/col coordinates
//
// TIME COMPLEXITY: O(b^d) where b = branching factor (~3), d = depth
// SPACE COMPLEXITY: O(b*d) for recursion stack
func minimax(state *core.State, depth int, isMaxPlayer bool, nodesEvaluated *int) int {
	*nodesEvaluated++

	// TERMINAL CONDITIONS
	// 1. Maximum depth reached: return heuristic evaluation
	// 2. Goal state reached: return optimal value
	if depth == 0 || state.IsGoal() {
		return state.Utility(isMaxPlayer)
	}

	// 3. No legal moves available (rare case)
	actions := state.GetActions()
	if len(actions) == 0 {
		return state.Utility(isMaxPlayer)
	}

	// MINIMAX RECURSION
	if isMaxPlayer {
		// MAX PLAYER (Robotic Sorter) - chooses move with MAXIMUM utility
		// Goal: Minimize disorder (reach goal state)
		maxEval := math.MinInt32 // Start with worst possible
		for _, action := range actions {
			// Try each legal move
			child := state.ApplyAction(action)
			// Recursively evaluate: pass turn to MIN player
			eval := minimax(child, depth-1, false, nodesEvaluated)
			// Keep track of best outcome for MAX
			maxEval = max(maxEval, eval)
		}
		// Return the best outcome MAX can achieve assuming MIN plays optimally
		return maxEval
	} else {
		// MIN PLAYER (System Glitch) - chooses move with MINIMUM utility
		// Goal: Maximize disorder (prevent reaching goal)
		minEval := math.MaxInt32 // Start with worst possible for MIN (best for MAX)
		for _, action := range actions {
			// Try each legal move
			child := state.ApplyAction(action)
			// Recursively evaluate: pass turn to MAX player
			eval := minimax(child, depth-1, true, nodesEvaluated)
			// Keep track of best outcome for MIN
			minEval = min(minEval, eval)
		}
		// Return the best outcome MIN can achieve assuming MAX plays optimally
		return minEval
	}
}

// MinimaxDecision - Wrapper function to make best initial move decision
//
// ALGORITHM FLOW:
// 1. Evaluate all legal moves from current state
// 2. Each move passes control to MIN player at depth-1
// 3. Compare all outcomes and choose MAX player's best move
//
// RETURNS:
// - Algorithm name, best move, utility value, nodes evaluated, time taken
// - This best move is the optimal strategy assuming both players play perfectly
//
// EXAMPLE:
// If state has moves [UP, DOWN, LEFT, RIGHT]:
// - Evaluate: UP   → calls minimax(stateUP,   depth-1, false)
// - Evaluate: DOWN → calls minimax(stateDOWN, depth-1, false)
// - etc.
// - Choose move with highest utility value
func MinimaxDecision(state *core.State, depth int) *core.AdversarialResult {
	start := time.Now()
	nodesEvaluated := 0

	bestMove := ""
	bestValue := math.MinInt32 // Start worse than any possible outcome

	// Try all legal moves from current position
	actions := state.GetActions()
	for _, action := range actions {
		// Apply the potential move
		child := state.ApplyAction(action)
		// Evaluate the resulting state (MIN player responds)
		value := minimax(child, depth-1, false, &nodesEvaluated)
		// Update best move if this is better
		if value > bestValue {
			bestValue = value
			bestMove = action
		}
	}

	return &core.AdversarialResult{
		Algorithm:      "Minimax",
		BestMove:       bestMove,
		Utility:        bestValue,
		NodesEvaluated: nodesEvaluated,
		TimeTaken:      time.Since(start),
	}
}

// Helper functions for max/min operations
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
