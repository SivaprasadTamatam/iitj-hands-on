package local

import (
	"fmt"
	"math"
	"math/rand"
	"search-strategies/core"
	"time"
)

// SimulatedAnnealing implements Simulated Annealing local search algorithm.
//
// Strategy:
// - Probabilistic local search inspired by metallurgical annealing process
// - Escapes local optima by accepting worse solutions with decreasing probability
// - Process:
//  1. Start with initial state
//  2. Generate random neighbor state
//  3. If neighbor has lower energy (cost), always accept it
//  4. If neighbor has higher energy, accept with probability P = e^(-ΔE/T)
//     where ΔE = energy increase, T = temperature
//  5. Cool down: T = T * coolingRate
//  6. Repeat until solution found or temperature too low
//
// Characteristics:
// - Completeness: No (not guaranteed to find a solution)
// - Optimality: No (may settle on local optimum)
// - Time Complexity: O(maxIterations * |actions|)
// - Space Complexity: O(1) - only stores current state
// - Memory-Efficient: Excellent - constant memory regardless of depth
// - Advantage: Can escape local optima, good for large search spaces
// - Disadvantage: Not complete, not optimal, requires parameter tuning
//
// Energy Function:
// - Uses Manhattan distance (h2) as energy/cost function
// - Lower energy = closer to goal
// - Goal state has energy = 0
//
// Temperature Schedule:
//   - Cooling schedule: T(t) = T₀ * α^t
//     where T₀ = initial temperature, α = coolingRate (e.g., 0.95)
//   - Higher initial temperature: more random exploration
//   - Cooling rate controls convergence speed
//   - When T → 0: algorithm becomes greedy (only accept improvements)
//
// Parameters:
//   - initial: Starting state of the puzzle
//   - initialTemp: Starting temperature (e.g., 100.0)
//   - coolingRate: Multiplicative cooling rate (e.g., 0.95)
//   - maxIterations: Maximum number of iterations (e.g., 10000)
//
// Returns: SearchResult indicating if goal reached, with timing and iteration information
func SimulatedAnnealing(initial *core.State, initialTemp float64, coolingRate float64, maxIterations int) *core.SearchResult {
	start := time.Now()
	// Current state during search
	current := initial
	// Current temperature (decreases over time)
	temperature := initialTemp
	statesExplored := 0

	// Main iteration loop
	for i := 0; i < maxIterations; i++ {
		statesExplored++

		// Check if goal state reached
		if current.IsGoal() {
			return &core.SearchResult{
				Algorithm:      "Simulated Annealing",
				Success:        true,
				Parameters:     fmt.Sprintf("T0=%.1f, α=%.3f, Iterations=%d", initialTemp, coolingRate, i),
				Path:           current.GetPath(),
				PathLength:     len(current.GetPath()),
				StatesExplored: statesExplored,
				TimeTaken:      time.Since(start),
			}
		}

		// Generate random neighbor state
		actions := current.GetActions()
		if len(actions) == 0 {
			break // No valid moves available
		}
		randomAction := actions[rand.Intn(len(actions))]
		neighbor := current.ApplyAction(randomAction)

		// Calculate energy values using Manhattan distance heuristic
		// Energy = distance to goal (Manhattan distance)
		currentEnergy := current.H2()
		neighborEnergy := neighbor.H2()
		// ΔE = energy increase (positive if neighbor is worse)
		deltaE := neighborEnergy - currentEnergy

		// Acceptance criterion: Metropolis algorithm
		if deltaE < 0 {
			// Neighbor is better (lower energy): always accept
			current = neighbor
		} else {
			// Neighbor is worse (higher energy): accept with probability
			// P(accept) = e^(-ΔE/T)
			// Higher temperature → higher probability of accepting worse states
			// Lower temperature → more likely to reject worse states
			acceptanceProbability := math.Exp(-float64(deltaE) / temperature)
			if rand.Float64() < acceptanceProbability {
				current = neighbor // Accept worse state with probability
			}
			// Otherwise: reject and stay at current state
		}

		// Cool down: decrease temperature for next iteration
		// T(t+1) = T(t) * α, where α = coolingRate (e.g., 0.95)
		temperature *= coolingRate

		// Stop if temperature becomes negligible
		if temperature < 0.01 {
			break // Temperature too low, algorithm has converged
		}
	}

	// Goal not reached within iteration limit or temperature became negligible
	return &core.SearchResult{
		Algorithm:      "Simulated Annealing",
		Success:        false,
		Parameters:     fmt.Sprintf("T0=%.1f, α=%.3f, MaxIter=%d", initialTemp, coolingRate, maxIterations),
		StatesExplored: statesExplored,
		TimeTaken:      time.Since(start),
	}
}
