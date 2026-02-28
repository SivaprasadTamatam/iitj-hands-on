package utils

import (
	"bufio"
	"fmt"
	"os"
	"search-strategies/core"
	"strings"
)

// ParseInput reads and parses the input file
func ParseInput(filename string) (*core.State, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var startBoard [3][3]int
	blankRow, blankCol := 0, 0
	index := 0

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "START:") {
			parts := strings.Split(strings.TrimPrefix(line, "START:"), ",")
			for _, part := range parts {
				row := index / 3
				col := index % 3
				if part == "B" {
					startBoard[row][col] = 0
					blankRow = row
					blankCol = col
				} else {
					fmt.Sscanf(part, "%d", &startBoard[row][col])
				}
				index++
			}
		}
	}

	return &core.State{
		Board:    startBoard,
		BlankRow: blankRow,
		BlankCol: blankCol,
		G:        0,
		H:        0,
		F:        0,
		Parent:   nil,
		Action:   "",
	}, nil
}

// PrintBoard prints a board configuration
func PrintBoard(board [3][3]int) {
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if board[i][j] == 0 {
				fmt.Print("B ")
			} else {
				fmt.Printf("%d ", board[i][j])
			}
		}
		fmt.Println()
	}
}
