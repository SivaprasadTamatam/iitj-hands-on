package core

import (
	"fmt"
)

// State represents an 8-puzzle board configuration
type State struct {
	Board    [3][3]int
	BlankRow int
	BlankCol int
	G        int // cost from start
	H        int // heuristic cost to goal
	F        int // g + h
	Parent   *State
	Action   string
}

// GoalState is the target configuration
var GoalState = [3][3]int{{1, 2, 3}, {4, 5, 6}, {7, 8, 0}} // 0 represents blank

// Hash function to convert board to string for visited tracking
func (s *State) Hash() string {
	return fmt.Sprintf("%v", s.Board)
}

// IsGoal checks if current state is goal
func (s *State) IsGoal() bool {
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if s.Board[i][j] != GoalState[i][j] {
				return false
			}
		}
	}
	return true
}

// GetActions returns valid actions from current state
func (s *State) GetActions() []string {
	actions := []string{}

	if s.BlankRow > 0 {
		actions = append(actions, "UP")
	}
	if s.BlankRow < 2 {
		actions = append(actions, "DOWN")
	}
	if s.BlankCol > 0 {
		actions = append(actions, "LEFT")
	}
	if s.BlankCol < 2 {
		actions = append(actions, "RIGHT")
	}
	return actions
}

// ApplyAction applies action and returns new state
func (s *State) ApplyAction(action string) *State {
	newState := &State{
		Board:    s.Board,
		BlankRow: s.BlankRow,
		BlankCol: s.BlankCol,
		Parent:   s,
		Action:   action,
		G:        s.G + 1,
	}

	swapRow, swapCol := s.BlankRow, s.BlankCol
	switch action {
	case "UP":
		swapRow--
	case "DOWN":
		swapRow++
	case "LEFT":
		swapCol--
	case "RIGHT":
		swapCol++
	}

	// Swap blank with target position
	newState.Board[s.BlankRow][s.BlankCol], newState.Board[swapRow][swapCol] =
		newState.Board[swapRow][swapCol], newState.Board[s.BlankRow][s.BlankCol]
	newState.BlankRow = swapRow
	newState.BlankCol = swapCol

	return newState
}

// H1 - Heuristic 1: Number of misplaced tiles
func (s *State) H1() int {
	count := 0
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if s.Board[i][j] != 0 && s.Board[i][j] != GoalState[i][j] {
				count++
			}
		}
	}
	return count
}

// H2 - Heuristic 2: Manhattan distance
func (s *State) H2() int {
	distance := 0
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			val := s.Board[i][j]
			if val != 0 {
				// Find goal position of this value
				goalRow, goalCol := (val-1)/3, (val-1)%3
				distance += abs(i-goalRow) + abs(j-goalCol)
			}
		}
	}
	return distance
}

// GetPath returns path from initial to current state
func (s *State) GetPath() []string {
	path := []string{}
	current := s
	for current.Parent != nil {
		path = append([]string{current.Action}, path...)
		current = current.Parent
	}
	return path
}

// Utility function for adversarial search
func (s *State) Utility(isMax bool) int {
	if s.IsGoal() {
		if isMax {
			return 1000
		}
		return -1000
	}

	manhattanDist := s.H2()
	if isMax {
		return -manhattanDist
	}
	return manhattanDist
}

// PrintBoard prints the board state
func (s *State) PrintBoard() {
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if s.Board[i][j] == 0 {
				fmt.Print("B ")
			} else {
				fmt.Printf("%d ", s.Board[i][j])
			}
		}
		fmt.Println()
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
