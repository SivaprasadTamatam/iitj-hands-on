package main

import (
	"fmt"
	"search-strategies/algorithms/adversarial"
	"search-strategies/core"
	"search-strategies/utils"
)

func main() {
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

	// Run adversarial search with depth 4
	adversarial.CompareAdversarialSearch(initial, 4)
}
