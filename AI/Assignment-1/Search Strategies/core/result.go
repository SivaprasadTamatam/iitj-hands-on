package core

import (
	"fmt"
	"strings"
	"time"
)

// SearchResult holds the results of a search algorithm
type SearchResult struct {
	Algorithm      string
	Success        bool
	Heuristic      string
	Parameters     string
	Path           []string
	PathLength     int
	StatesExplored int
	TimeTaken      time.Duration
}

func (sr *SearchResult) Print() {
	fmt.Printf("\n========================================\n")
	fmt.Printf("Algorithm: %s\n", sr.Algorithm)
	if sr.Success {
		fmt.Printf("Status: SUCCESS\n")
	} else {
		fmt.Printf("Status: FAILURE\n")
	}
	if sr.Heuristic != "" {
		fmt.Printf("Heuristic: %s\n", sr.Heuristic)
	}
	if sr.Parameters != "" {
		fmt.Printf("Parameters: %s\n", sr.Parameters)
	}
	if sr.Success {
		fmt.Printf("Optimal Path: %s\n", strings.Join(sr.Path, " → "))
		fmt.Printf("Path Length: %d\n", sr.PathLength)
	}
	fmt.Printf("Total States Explored: %d\n", sr.StatesExplored)
	fmt.Printf("Total Time Taken: %v\n", sr.TimeTaken)
	fmt.Printf("========================================\n")
}

// AdversarialResult holds results for minimax/alpha-beta
type AdversarialResult struct {
	Algorithm          string
	BestMove           string
	Utility            int
	NodesEvaluated     int
	NodesWithAlphaBeta int
	PruningGain        string
	TimeTaken          time.Duration
}

func (ar *AdversarialResult) Print() {
	fmt.Printf("\n========================================\n")
	fmt.Printf("Algorithm: %s\n", ar.Algorithm)
	fmt.Printf("Best Move: %s\n", ar.BestMove)
	fmt.Printf("Expected Utility: %d\n", ar.Utility)
	fmt.Printf("Total Nodes Evaluated: %d\n", ar.NodesEvaluated)
	if ar.NodesWithAlphaBeta > 0 {
		fmt.Printf("Nodes with Alpha-Beta: %d\n", ar.NodesWithAlphaBeta)
		fmt.Printf("Pruning Gain: %s\n", ar.PruningGain)
	}
	fmt.Printf("Total Time Taken: %v\n", ar.TimeTaken)
	fmt.Printf("========================================\n")
}
