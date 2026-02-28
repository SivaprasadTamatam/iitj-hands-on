# AI Assignment 1 - Implementation in Go

This repository contains the complete implementation of Assignment 1 in **Golang (Go)**, covering search strategies and constraint satisfaction problems.

## Directory Structure

```
AI/Assignment-1/
├── Search Strategies/
│   ├── algorithms/                  # Modular algorithm implementations
│   │   ├── uninformed/             # BFS, DFS
│   │   ├── informed/               # Greedy, A*
│   │   ├── bounded/                # IDA*
│   │   ├── local/                  # Simulated Annealing
│   │   └── adversarial/            # Minimax, Alpha-Beta
│   ├── core/                        # Core data structures
│   │   ├── state.go                # State representation
│   │   ├── queue.go                # Priority queue
│   │   └── result.go               # Result types
│   ├── utils/                       # Utility functions
│   │   └── io.go                   # Input/output
│   ├── main.go                      # Main program runner
│   ├── run_adversarial.go          # Adversarial search runner
│   ├── input.txt                    # Input file for 8-puzzle
│   ├── theory.md                    # Theoretical explanations
│   ├── README.md                    # Module documentation
│   └── go.mod                       # Go module file
│
├── CSP/
│   ├── main.go                      # CSP bot scheduling implementation
│   ├── input.txt                    # Input file for CSP problem
│   ├── theory.md                    # CSP theory and answers
│   └── go.mod                       # Go module file
│
└── README.md                        # This file
```

## Prerequisites

- **Go (Golang)** version 1.16 or higher
- Install from: https://golang.org/dl/

Verify installation:
```powershell
go version
```

## Question 1: Search Strategies

### Running the Algorithms

#### 1. All Search Algorithms (BFS, DFS, Greedy, A*, IDA*, Simulated Annealing)

```powershell
cd "AI\Assignment-1\Search Strategies"
go run main.go
```

**Output includes:**
- Breadth-First Search (BFS)
- Depth-First Search (DFS) with depth limit
- Greedy Best-First Search (both h1 and h2 heuristics)
- A* Search (both h1 and h2 heuristics)
- Iterative Deepening A* (IDA*) (both heuristics)
- Simulated Annealing

#### 2. Adversarial Search (Minimax & Alpha-Beta Pruning)

```powershell
cd "AI\Assignment-1\Search Strategies"
go run adversarial_standalone.go
```

**Output includes:**
- Minimax algorithm results
- Alpha-Beta Pruning results
- Comparison table showing pruning efficiency

### Input Format (input.txt)

```
START:1,2,3,B,4,6,7,5,8
GOAL:1,2,3,4,5,6,7,8,B
```

- Use comma-separated values for the 3x3 board (row-major order)
- `B` represents the blank space
- START: Initial configuration
- GOAL: Target configuration

### Example Output

```
Algorithm: A* Search
Status: SUCCESS
Heuristic: Manhattan Distance (h2)
Optimal Path: RIGHT → DOWN → RIGHT
Path Length: 3
Total States Explored: 4
Total Time Taken: 0s
```

## Question 2: Constraint Satisfaction Problem

### Running the CSP Solver

```powershell
cd "AI\Assignment-1\CSP"
go run main.go
```

**Output includes:**
- Constraint graph visualization
- First 3 steps of backtracking with MRV heuristic demonstration
- Complete CSP solution with performance metrics

### Input Format (input.txt)

```
BOTS:A,B,C
SLOTS:1,2,3,4
CONSTRAINTS:
NO_BACK_TO_BACK:true
BOT_C_NO_SLOT_4:true
MINIMUM_COVERAGE:true
```

### Features Implemented

1. **Variable Selection:** Minimum Remaining Values (MRV) heuristic
2. **Inference:** Forward Checking
3. **Constraints:**
   - No Back-to-Back: Bot cannot work consecutive slots
   - Maintenance Break: Bot C cannot work in Slot 4
   - Minimum Coverage: All bots must be used at least once

### Example Output

```
Status: SUCCESS
Heuristic: Minimum Remaining Values (MRV)
Inference: Forward Checking

Final Assignment:
  Slot 1: Bot C
  Slot 2: Bot A
  Slot 3: Bot B
  Slot 4: Bot A

Total Assignments Attempted: 6
Backtracks: 1
Total Time Taken: 0.327ms
```

## Theory Documentation

Complete theoretical explanations and discussion answers are provided in Markdown files:

### Search Strategies Theory ([theory.md](Search%20Strategies/theory.md))

1. **Problem Formulation**
   - State representation
   - Actions and transitions
   - Goal state definition
   - Path cost function

2. **Algorithm Descriptions**
   - Uninformed Search (BFS, DFS)
   - Informed Search (Greedy, A*)
   - Memory-Bounded (IDA*)
   - Local Search (Simulated Annealing)
   - Adversarial Search (Minimax, Alpha-Beta)

3. **Discussion Answers**
   - Heuristic admissibility and consistency
   - A* vs IDA* comparison
   - DFS infinite path prevention
   - BFS vs DFS memory analysis
   - Minimax vs Alpha-Beta comparison

### CSP Theory ([CSP/theory.md](CSP/theory.md))

1. **Problem Definition**
   - Variables, domains, constraints
   - Constraint graph with visualization

2. **Backtracking with MRV**
   - Step-by-step demonstration
   - Forward checking process

3. **Discussion Answers**
   - Forward checking failure detection
   - Arc consistency (AC-3) explanation

## Implementation Features

### Search Strategies (Question 1)

**Architecture:**
✅ Modular package-based organization  
✅ Separation of concerns (core/algorithms/utils)  
✅ Each algorithm in separate file  
✅ Reusable core components  

**Algorithms:**
✅ State representation with 3x3 array  
✅ Visited state tracking (prevents infinite loops)  
✅ Parent tracking (prevents immediate cycles)  
✅ Multiple heuristics (h1: misplaced tiles, h2: Manhattan distance)  
✅ Priority queue for informed search  
✅ Depth limiting for DFS  
✅ Iterative deepening for IDA*  
✅ Cooling schedule for Simulated Annealing  
✅ Minimax with game tree search  
✅ Alpha-Beta pruning with cutoffs  

### CSP (Question 2)

✅ MRV heuristic for variable selection  
✅ Forward checking for inference  
✅ Binary constraints (no back-to-back)  
✅ Unary constraints (Bot C restriction)  
✅ Global constraints (minimum coverage)  
✅ Backtracking with constraint propagation  
✅ Domain reduction and consistency checking  

## Performance Metrics

All implementations track and report:

- **Success/Failure** status
- **Heuristic/Parameters** used
- **Modular Architecture:** Package-based organization by algorithm type
- **Clean Separation:** Core types, algorithms, and utilities in separate packages
- **Well-documented:** Comments throughout + detailed README files
- **Type-safe:** Leverages Go's strong typing
- **Idiomatic Go:** Follows Go best practices and project layout
- **Readable:** Clear naming conventions and file organization
- **Maintainable:** Easy to find, update, and extend algorithms
- **Professional:** Production-ready code structure
## Code Quality

- **Clean Architecture:** Separated concerns (state, search, I/O)
- **Well-documented:** Comments throughout
- **Type-safe:** Leverages Go's strong typing
- **Idiomatic Go:** Follows Go best practices
- **Readable:** Clear naming conventions

## Testing

All implementations have been tested with the provided input files and produce correct results:

- **BFS:** Finds optimal solution ✓
- **A*:** Finds optimal solution efficiently ✓
- **Minimax:** Evaluates game tree correctly ✓
- **Alpha-Beta:** Prunes branches effectively ✓
- **CSP:** Finds valid assignment ✓

## Compilation (Optional)

To create standalone executables:

```powershell
# Search Strategies
cd "AI\Assignment-1\Search Strategies"
go build -o search.exe main.go

# Adversarial Search
go build -o adversarial.exe adversarial_standalone.go

# CSP
cd "..\CSP"
go build -o csp.exe main.go
```

Then run:
```powershell
.\search.exe
.\adversarial.exe
.\csp.exe
```

## Assignment Deliverable

### PDF Report Contents

1. **Question 1 - Search Strategies**
   - Problem formulation (from theory.md)
   - Algorithm implementations (link to main.go, adversarial_standalone.go)
   - Output results (screenshots or copied text)
   - Discussion answers (from theory.md)
**Search Strategies (Modular Structure):**
- **Main Runner:** [Search Strategies/main.go](Search%20Strategies/main.go)
- **Adversarial Runner:** [Search Strategies/run_adversarial.go](Search%20Strategies/run_adversarial.go)
- **Core Package:** [Search Strategies/core/](Search%20Strategies/core/)
- **Algorithms:** [Search Strategies/algorithms/](Search%20Strategies/algorithms/)
- **Module README:** [Search Strategies/README.md](Search%20Strategies/README.md)

**CSP Solver:**
- **CSP Mainefinition (from CSP/theory.md)
   - Constraint graph (from program output)
   - Backtracking steps (from program output)
   - Implementation (link to CSP/main.go)
   - Discussion answers (from CSP/theory.md)

### Working Code Links

- **Search Strategies Main:** [Search Strategies/main.go](Search%20Strategies/main.go)
- **Adversarial Search:** [Search Strategies/adversarial_standalone.go](Search%20Strategies/adversarial_standalone.go)
- **CSP Solver:** [CSP/main.go](CSP/main.go)

### Theory Files

- **Search Strategies Theory:** [Search Strategies/theory.md](Search%20Strategies/theory.md)
- **CSP Theory:** [CSP/theory.md](CSP/theory.md)

## Troubleshooting

### Common Issues

1. **"go: command not found"**
   - Install Go from https://golang.org/dl/
   - Add Go to PATH environment variable

2. **"cannot find package"**
   - Ensure you're in the correct directory
   - Run `go mod tidy` in the directory

3. **Input file not found**
   - Verify `input.txt` exists in the same directory as the Go file
   - Check file format matches specification

## Author

**Language:** Golang (Go)  
**Assignment:** AI Assignment 1  
**Topics:** Search Strategies, Constraint Satisfaction Problems  
**Date:** February 2026

## Notes

- Theory explanations are comprehensive and answer all discussion questions
- All algorithms are implemented from scratch (no external AI libraries)
- Code follows Go conventions and best practices
- Performance metrics match expected complexity analysis
- Input/output format matches assignment requirements

---

For any questions or issues, please refer to the theory.md files in each directory for detailed explanations.
