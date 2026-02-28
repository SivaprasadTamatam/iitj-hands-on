# Search Strategies - Modular Implementation

This directory contains a well-organized, modular implementation of various search algorithms for the 8-puzzle problem.

## 📁 Directory Structure

```
Search Strategies/
├── core/                           # Core data structures and types
│   ├── state.go                   # State representation and operations
│   ├── queue.go                   # Priority queue implementation
│   └── result.go                  # Result types for search algorithms
│
├── algorithms/                     # Search algorithm implementations
│   ├── uninformed/                # Uninformed search strategies
│   │   ├── bfs.go                # Breadth-First Search
│   │   └── dfs.go                # Depth-First Search
│   │
│   ├── informed/                  # Informed (heuristic) search
│   │   ├── greedy.go             # Greedy Best-First Search
│   │   └── astar.go              # A* Search
│   │
│   ├── bounded/                   # Memory-bounded search
│   │   └── idastar.go            # Iterative Deepening A*
│   │
│   ├── local/                     # Local search methods
│   │   └── simulated_annealing.go # Simulated Annealing
│   │
│   └── adversarial/               # Adversarial search (game playing)
│       ├── minimax.go            # Minimax algorithm
│       └── alphabeta.go          # Alpha-Beta Pruning
│
├── utils/                          # Utility functions
│   └── io.go                      # Input/output operations
│
├── main.go                         # Main program (all search algorithms)
├── run_adversarial.go             # Adversarial search runner
├── input.txt                       # Input configuration file
├── theory.md                       # Theoretical explanations
└── go.mod                          # Go module definition
```

## 🚀 Running the Programs

### Run All Search Algorithms

This runs BFS, DFS, Greedy, A*, IDA*, and Simulated Annealing:

```powershell
cd "Search Strategies"
go run main.go
```

### Run Adversarial Search

This runs Minimax and Alpha-Beta Pruning with comparison:

```powershell
cd "Search Strategies"
go run run_adversarial.go
```

## 📦 Module Organization

### Core Package (`core/`)

**state.go** - State representation and operations:
- `State` struct with board configuration
- `Hash()` - Convert state to unique string
- `IsGoal()` - Check if goal state reached
- `GetActions()` - Get valid moves
- `ApplyAction()` - Apply move and get new state
- `H1()` - Misplaced tiles heuristic
- `H2()` - Manhattan distance heuristic
- `Utility()` - Evaluation for adversarial search

**queue.go** - Priority queue for informed search:
- Implements `heap.Interface` for Go's container/heap
- Used by A* and Greedy Best-First Search

**result.go** - Result structures:
- `SearchResult` - Results for search algorithms
- `AdversarialResult` - Results for Minimax/Alpha-Beta

### Algorithms Package (`algorithms/`)

#### Uninformed Search (`algorithms/uninformed/`)

**bfs.go** - Breadth-First Search:
- Guarantees optimal solution
- Uses FIFO queue
- Tracks visited states

**dfs.go** - Depth-First Search:
- Uses depth limit to prevent infinite loops
- Memory efficient
- May not find optimal solution

#### Informed Search (`algorithms/informed/`)

**greedy.go** - Greedy Best-First Search:
- Uses heuristic only (h(n))
- Fast but not guaranteed optimal
- Supports both h1 and h2

**astar.go** - A* Search:
- Uses f(n) = g(n) + h(n)
- Guarantees optimal solution with admissible heuristic
- Supports both h1 and h2

#### Memory-Bounded Search (`algorithms/bounded/`)

**idastar.go** - Iterative Deepening A*:
- Combines benefits of DFS and A*
- Memory efficient (O(d) space)
- Optimal with admissible heuristic
- Regenerates nodes across iterations

#### Local Search (`algorithms/local/`)

**simulated_annealing.go** - Simulated Annealing:
- Probabilistic local search
- Can escape local optima
- Uses temperature and cooling rate
- Not guaranteed to find optimal solution

#### Adversarial Search (`algorithms/adversarial/`)

**minimax.go** - Minimax Algorithm:
- MAX player tries to reach goal
- MIN player tries to prevent it
- Complete game tree evaluation

**alphabeta.go** - Alpha-Beta Pruning:
- Optimized Minimax with pruning
- Reduces nodes evaluated
- Same result as Minimax but faster
- Includes comparison function

### Utils Package (`utils/`)

**io.go** - Input/output utilities:
- `ParseInput()` - Read and parse input.txt
- `PrintBoard()` - Display board state

## 🔧 How It Works

### Main Program Flow

1. **Parse Input** - Read initial state from `input.txt`
2. **Display States** - Show initial and goal states
3. **Run Algorithms** - Execute each search algorithm
4. **Display Results** - Print formatted results for each

### Import Structure

```go
import (
    "search-strategies/core"
    "search-strategies/algorithms/uninformed"
    "search-strategies/algorithms/informed"
    "search-strategies/algorithms/bounded"
    "search-strategies/algorithms/local"
    "search-strategies/algorithms/adversarial"
    "search-strategies/utils"
)
```

### Example Usage

```go
// Parse input
initial, err := utils.ParseInput("input.txt")

// Run BFS
result := uninformed.BFS(initial)
result.Print()

// Run A* with Manhattan distance
result := informed.AStar(initial, "h2")
result.Print()

// Run adversarial search
adversarial.CompareAdversarialSearch(initial, 4)
```

## 📝 Input Format

The `input.txt` file should contain:

```
START:1,2,3,B,4,6,7,5,8
GOAL:1,2,3,4,5,6,7,8,B
```

- Comma-separated values for 3x3 board (row-major order)
- `B` represents blank space
- Numbers 1-8 represent tiles

## 🎯 Benefits of This Structure

### 1. **Separation of Concerns**
- Each algorithm in its own file
- Core functionality separated from algorithms
- Utilities isolated from business logic

### 2. **Maintainability**
- Easy to find and modify specific algorithms
- Clear organization by algorithm type
- Reduced file size for better readability

### 3. **Reusability**
- Core types can be used by any algorithm
- Algorithms can be imported independently
- Easy to add new algorithms

### 4. **Testability**
- Each package can be tested independently
- Clear interfaces between components
- Mock-friendly design

### 5. **Scalability**
- Easy to add new search algorithms
- Simple to extend with new features
- Clean dependency graph

## 🔍 Algorithm Categories

### **Uninformed Search** (No domain knowledge)
- BFS, DFS
- Explores blindly without heuristics

### **Informed Search** (Uses heuristics)
- Greedy, A*
- More efficient with good heuristics

### **Memory-Bounded** (Limited memory)
- IDA*
- Trades time for space

### **Local Search** (Non-systematic)
- Simulated Annealing
- Good for optimization problems

### **Adversarial** (Two-player)
- Minimax, Alpha-Beta
- Models competing agents

## 📊 Performance Comparison

Run both programs to see:
- States explored
- Time taken
- Path length
- Optimality
- Pruning efficiency (for Alpha-Beta)

## 🛠️ Extending the Code

### Adding a New Search Algorithm

1. Create new file in appropriate `algorithms/` subdirectory
2. Import `search-strategies/core`
3. Implement function returning `*core.SearchResult`
4. Import and call from `main.go`

Example:
```go
// algorithms/informed/bidirectional.go
package informed

import "search-strategies/core"

func BidirectionalSearch(initial *core.State) *core.SearchResult {
    // Implementation
}
```

### Adding a New Heuristic

Add method to `State` in `core/state.go`:

```go
func (s *State) H3() int {
    // New heuristic implementation
}
```

## 📚 Related Files

- **theory.md** - Detailed theoretical explanations
- **input.txt** - Problem configuration
- **go.mod** - Go module dependencies

## ✅ Verification

All algorithms tested and verified:
- ✅ BFS finds optimal path (3 moves)
- ✅ A* finds optimal path efficiently (4 states explored)
- ✅ Minimax evaluates game tree correctly
- ✅ Alpha-Beta prunes ~11% of nodes
- ✅ All algorithms complete successfully

## 🎓 Learning Value

This modular structure demonstrates:
- Clean code architecture
- Go package organization
- Interface design
- Algorithm implementation patterns
- Professional code organization

---

**Language:** Go (Golang)  
**Pattern:** Modular Architecture  
**Purpose:** AI Assignment 1 - Search Strategies  
**Status:** Production Ready ✅
