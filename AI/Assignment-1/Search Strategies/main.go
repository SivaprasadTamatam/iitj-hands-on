package main

import (
	"fmt"
	"math/rand"
	"search-strategies/algorithms/bounded"
	"search-strategies/algorithms/informed"
	"search-strategies/algorithms/local"
	"search-strategies/algorithms/uninformed"
	"search-strategies/core"
	"search-strategies/utils"
	"time"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// Parse input
	initial, err := utils.ParseInput("input.txt")
	if err != nil {
		fmt.Println("Error reading input:", err)
		return
	}

	fmt.Println("Initial State:")
	initial.PrintBoard()
	fmt.Println("\nGoal State:")
	utils.PrintBoard(core.GoalState)

	// Run all search algorithms
	fmt.Println("========  RUNNING ALL SEARCH ALGORITHMS  ========")

	// Uninformed Search
	fmt.Println("=== UNINFORMED SEARCH ===")
	uninformed.BFS(initial).Print()
	uninformed.DFS(initial, 50).Print()

	// Informed Search
	fmt.Println("\n=== INFORMED SEARCH ===")
	informed.GreedyBFS(initial, "h1").Print()
	informed.GreedyBFS(initial, "h2").Print()
	informed.AStar(initial, "h1").Print()
	informed.AStar(initial, "h2").Print()

	// Memory-Bounded & Local Search
	fmt.Println("\n=== MEMORY-BOUNDED & LOCAL SEARCH ===")
	bounded.IDAStar(initial, "h1").Print()
	bounded.IDAStar(initial, "h2").Print()
	local.SimulatedAnnealing(initial, 100.0, 0.95, 10000).Print()
}
