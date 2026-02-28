package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"time"
)

// CSP Domain and Variables
type CSP struct {
	Variables   []string          // Slot1, Slot2, Slot3, Slot4
	Domains     map[string][]string // Domain for each variable
	Constraints map[string]bool     // Constraint flags
	Assignment  map[string]string   // Current assignment
}

// CSP Result
type CSPResult struct {
	Success         bool
	Heuristic       string
	Inference       string
	Constraints     []string
	FinalAssignment map[string]string
	Attempts        int
	Backtracks      int
	TimeTaken       time.Duration
}

func (cr *CSPResult) Print() {
	fmt.Printf("\n========================================\n")
	if cr.Success {
		fmt.Printf("Status: SUCCESS\n")
	} else {
		fmt.Printf("Status: FAILURE\n")
	}
	fmt.Printf("Heuristic: %s\n", cr.Heuristic)
	fmt.Printf("Inference: %s\n", cr.Inference)
	fmt.Printf("Constraints Applied:\n")
	for _, constraint := range cr.Constraints {
		fmt.Printf("  - %s\n", constraint)
	}
	
	if cr.Success {
		fmt.Printf("\nFinal Assignment:\n")
		for i := 1; i <= 4; i++ {
			slot := fmt.Sprintf("Slot%d", i)
			fmt.Printf("  Slot %d: Bot %s\n", i, cr.FinalAssignment[slot])
		}
	}
	
	fmt.Printf("\nTotal Assignments Attempted: %d\n", cr.Attempts)
	fmt.Printf("Backtracks: %d\n", cr.Backtracks)
	fmt.Printf("Total Time Taken: %v\n", cr.TimeTaken)
	if cr.Success && cr.Backtracks == 0 {
		fmt.Printf("Performance: 100%% success on first try\n")
	} else if cr.Success {
		fmt.Printf("Performance: Success after %d backtracks\n", cr.Backtracks)
	}
	fmt.Printf("========================================\n")
}

// Parse CSP input file
func parseCSPInput(filename string) (*CSP, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	csp := &CSP{
		Variables:   []string{"Slot1", "Slot2", "Slot3", "Slot4"},
		Domains:     make(map[string][]string),
		Constraints: make(map[string]bool),
		Assignment:  make(map[string]string),
	}

	scanner := bufio.NewScanner(file)
	var bots []string

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "CONSTRAINTS:") {
			continue
		}

		if strings.HasPrefix(line, "BOTS:") {
			botStr := strings.TrimPrefix(line, "BOTS:")
			bots = strings.Split(botStr, ",")
			for i := range bots {
				bots[i] = strings.TrimSpace(bots[i])
			}
		} else if strings.HasPrefix(line, "NO_BACK_TO_BACK:") {
			csp.Constraints["NO_BACK_TO_BACK"] = true
		} else if strings.HasPrefix(line, "BOT_C_NO_SLOT_4:") {
			csp.Constraints["BOT_C_NO_SLOT_4"] = true
		} else if strings.HasPrefix(line, "MINIMUM_COVERAGE:") {
			csp.Constraints["MINIMUM_COVERAGE"] = true
		}
	}

	// Initialize domains
	for _, variable := range csp.Variables {
		if variable == "Slot4" && csp.Constraints["BOT_C_NO_SLOT_4"] {
			// Apply unary constraint: Bot C cannot work Slot 4
			csp.Domains[variable] = []string{}
			for _, bot := range bots {
				if bot != "C" {
					csp.Domains[variable] = append(csp.Domains[variable], bot)
				}
			}
		} else {
			csp.Domains[variable] = make([]string, len(bots))
			copy(csp.Domains[variable], bots)
		}
	}

	return csp, nil
}

// Check if assignment is consistent with constraints
func (csp *CSP) isConsistent(variable string, value string) bool {
	// Check no back-to-back constraint
	if csp.Constraints["NO_BACK_TO_BACK"] {
		// Get slot number
		var slotNum int
		fmt.Sscanf(variable, "Slot%d", &slotNum)

		// Check previous slot
		if slotNum > 1 {
			prevSlot := fmt.Sprintf("Slot%d", slotNum-1)
			if prevBot, exists := csp.Assignment[prevSlot]; exists {
				if prevBot == value {
					return false // Same bot in consecutive slots
				}
			}
		}

		// Check next slot
		if slotNum < 4 {
			nextSlot := fmt.Sprintf("Slot%d", slotNum+1)
			if nextBot, exists := csp.Assignment[nextSlot]; exists {
				if nextBot == value {
					return false // Same bot in consecutive slots
				}
			}
		}
	}

	return true
}

// Check if minimum coverage can still be satisfied
func (csp *CSP) canSatisfyMinimumCoverage(remainingVars []string, remainingDomains map[string][]string) bool {
	if !csp.Constraints["MINIMUM_COVERAGE"] {
		return true
	}

	// Get all used bots
	usedBots := make(map[string]bool)
	for _, bot := range csp.Assignment {
		usedBots[bot] = true
	}

	// Get all available bots in remaining domains
	availableBots := make(map[string]bool)
	for _, domain := range remainingDomains {
		for _, bot := range domain {
			availableBots[bot] = true
		}
	}

	// Check if we can still assign all required bots
	allBots := []string{"A", "B", "C"}
	for _, bot := range allBots {
		if !usedBots[bot] && !availableBots[bot] {
			return false // Required bot cannot be assigned anymore
		}
	}

	return true
}

// Minimum Remaining Values (MRV) heuristic
func (csp *CSP) selectUnassignedVariable(domains map[string][]string) string {
	minSize := int(^uint(0) >> 1) // Max int
	var selected string

	for _, variable := range csp.Variables {
		if _, assigned := csp.Assignment[variable]; !assigned {
			domainSize := len(domains[variable])
			if domainSize < minSize {
				minSize = domainSize
				selected = variable
			}
		}
	}

	return selected
}

// Forward checking
func (csp *CSP) forwardCheck(variable string, value string, domains map[string][]string) map[string][]string {
	newDomains := make(map[string][]string)
	for k, v := range domains {
		newDomains[k] = make([]string, len(v))
		copy(newDomains[k], v)
	}

	// Get slot number
	var slotNum int
	fmt.Sscanf(variable, "Slot%d", &slotNum)

	// Update next slot domain (remove assigned value to prevent back-to-back)
	if csp.Constraints["NO_BACK_TO_BACK"] && slotNum < 4 {
		nextSlot := fmt.Sprintf("Slot%d", slotNum+1)
		if _, assigned := csp.Assignment[nextSlot]; !assigned {
			filtered := []string{}
			for _, bot := range newDomains[nextSlot] {
				if bot != value {
					filtered = append(filtered, bot)
				}
			}
			newDomains[nextSlot] = filtered
		}
	}

	// Update previous slot domain (if not yet assigned)
	if csp.Constraints["NO_BACK_TO_BACK"] && slotNum > 1 {
		prevSlot := fmt.Sprintf("Slot%d", slotNum-1)
		if _, assigned := csp.Assignment[prevSlot]; !assigned {
			filtered := []string{}
			for _, bot := range newDomains[prevSlot] {
				if bot != value {
					filtered = append(filtered, bot)
				}
			}
			newDomains[prevSlot] = filtered
		}
	}

	return newDomains
}

// Backtracking search with MRV and Forward Checking
func (csp *CSP) backtrackingSearch(domains map[string][]string, attempts *int, backtracks *int) bool {
	*attempts++

	// Check if assignment is complete
	if len(csp.Assignment) == len(csp.Variables) {
		// Check minimum coverage constraint
		if csp.Constraints["MINIMUM_COVERAGE"] {
			usedBots := make(map[string]bool)
			for _, bot := range csp.Assignment {
				usedBots[bot] = true
			}
			if len(usedBots) < 3 {
				return false // Not all bots used
			}
		}
		return true
	}

	// Select variable using MRV heuristic
	variable := csp.selectUnassignedVariable(domains)
	if variable == "" {
		return false
	}

	// Try each value in domain
	for _, value := range domains[variable] {
		if csp.isConsistent(variable, value) {
			// Make assignment
			csp.Assignment[variable] = value

			// Forward checking
			newDomains := csp.forwardCheck(variable, value, domains)

			// Get remaining variables
			remainingVars := []string{}
			for _, v := range csp.Variables {
				if _, assigned := csp.Assignment[v]; !assigned {
					remainingVars = append(remainingVars, v)
				}
			}

			// Check if domains are not empty and minimum coverage can be satisfied
			allDomainsValid := true
			for _, v := range remainingVars {
				if len(newDomains[v]) == 0 {
					allDomainsValid = false
					break
				}
			}

			if allDomainsValid && csp.canSatisfyMinimumCoverage(remainingVars, newDomains) {
				// Recursive call
				if csp.backtrackingSearch(newDomains, attempts, backtracks) {
					return true
				}
			}

			// Backtrack
			delete(csp.Assignment, variable)
			*backtracks++
		}
	}

	return false
}

// Solve CSP
func solveCSP(filename string) *CSPResult {
	start := time.Now()

	csp, err := parseCSPInput(filename)
	if err != nil {
		fmt.Println("Error reading input:", err)
		return nil
	}

	// Collect constraint descriptions
	constraints := []string{}
	if csp.Constraints["NO_BACK_TO_BACK"] {
		constraints = append(constraints, "No Back-to-Back: Bot cannot work two consecutive slots")
	}
	if csp.Constraints["BOT_C_NO_SLOT_4"] {
		constraints = append(constraints, "Maintenance Break: Bot C cannot work in Slot 4")
	}
	if csp.Constraints["MINIMUM_COVERAGE"] {
		constraints = append(constraints, "Minimum Coverage: Every bot must be used at least once")
	}

	attempts := 0
	backtracks := 0

	// Solve using backtracking with MRV and Forward Checking
	success := csp.backtrackingSearch(csp.Domains, &attempts, &backtracks)

	result := &CSPResult{
		Success:         success,
		Heuristic:       "Minimum Remaining Values (MRV)",
		Inference:       "Forward Checking",
		Constraints:     constraints,
		FinalAssignment: make(map[string]string),
		Attempts:        attempts,
		Backtracks:      backtracks,
		TimeTaken:       time.Since(start),
	}

	if success {
		for k, v := range csp.Assignment {
			result.FinalAssignment[k] = v
		}
	}

	return result
}

// Demonstrate backtracking steps
func demonstrateBacktrackingSteps(filename string) {
	fmt.Println("\n======== DEMONSTRATING FIRST 3 STEPS WITH MRV ========\n")

	csp, _ := parseCSPInput(filename)

	fmt.Println("Initial Domains (after unary constraint):")
	for _, slot := range csp.Variables {
		fmt.Printf("  %s: %v\n", slot, csp.Domains[slot])
	}

	fmt.Println("\n--- Step 1: Choose variable with MRV ---")
	selected := csp.selectUnassignedVariable(csp.Domains)
	fmt.Printf("Selected: %s (Domain size: %d)\n", selected, len(csp.Domains[selected]))
	fmt.Printf("Try %s = %s\n", selected, csp.Domains[selected][0])
	
	csp.Assignment[selected] = csp.Domains[selected][0]
	step1Domains := csp.forwardCheck(selected, csp.Domains[selected][0], csp.Domains)
	
	fmt.Println("\nDomains after forward checking:")
	for _, slot := range csp.Variables {
		if _, assigned := csp.Assignment[slot]; assigned {
			fmt.Printf("  %s: %s ✓\n", slot, csp.Assignment[slot])
		} else {
			fmt.Printf("  %s: %v\n", slot, step1Domains[slot])
		}
	}

	fmt.Println("\n--- Step 2: Choose variable with MRV ---")
	selected2 := csp.selectUnassignedVariable(step1Domains)
	fmt.Printf("Selected: %s (Domain size: %d)\n", selected2, len(step1Domains[selected2]))
	fmt.Printf("Try %s = %s\n", selected2, step1Domains[selected2][0])
	
	csp.Assignment[selected2] = step1Domains[selected2][0]
	step2Domains := csp.forwardCheck(selected2, step1Domains[selected2][0], step1Domains)
	
	fmt.Println("\nDomains after forward checking:")
	for _, slot := range csp.Variables {
		if _, assigned := csp.Assignment[slot]; assigned {
			fmt.Printf("  %s: %s ✓\n", slot, csp.Assignment[slot])
		} else {
			fmt.Printf("  %s: %v\n", slot, step2Domains[slot])
		}
	}

	fmt.Println("\n--- Step 3: Choose variable with MRV ---")
	selected3 := csp.selectUnassignedVariable(step2Domains)
	fmt.Printf("Selected: %s (Domain size: %d)\n", selected3, len(step2Domains[selected3]))
	fmt.Printf("Try %s = %s\n", selected3, step2Domains[selected3][0])
	
	csp.Assignment[selected3] = step2Domains[selected3][0]
	step3Domains := csp.forwardCheck(selected3, step2Domains[selected3][0], step2Domains)
	
	fmt.Println("\nDomains after forward checking:")
	for _, slot := range csp.Variables {
		if _, assigned := csp.Assignment[slot]; assigned {
			fmt.Printf("  %s: %s ✓\n", slot, csp.Assignment[slot])
		} else {
			fmt.Printf("  %s: %v\n", slot, step3Domains[slot])
		}
	}
}

// Visualize constraint graph
func visualizeConstraintGraph() {
	fmt.Println("\n======== CONSTRAINT GRAPH ========\n")
	fmt.Println("Nodes: Slot1, Slot2, Slot3, Slot4")
	fmt.Println("\nEdges (No Back-to-Back constraints):")
	fmt.Println("  Slot1 -------- Slot2")
	fmt.Println("  Slot2 -------- Slot3")
	fmt.Println("  Slot3 -------- Slot4")
	fmt.Println("\nUnary Constraint:")
	fmt.Println("  Slot4: Domain {A, B} (C excluded)")
	fmt.Println("\nGlobal Constraint:")
	fmt.Println("  All bots {A, B, C} must be used at least once")
	fmt.Println("\nVisualization:")
	fmt.Println("```")
	fmt.Println("     Slot1 -------- Slot2 -------- Slot3 -------- Slot4")
	fmt.Println("       |              |              |              |")
	fmt.Println("    {A,B,C}        {A,B,C}        {A,B,C}        {A,B}")
	fmt.Println("       |              |              |")
	fmt.Println("       └──────────────┴──────────────┴────> Minimum Coverage")
	fmt.Println("```")
}

func main() {
	fmt.Println("======== QUESTION 2: CSP - SECURITY BOT SCHEDULING ========")

	// Visualize constraint graph
	visualizeConstraintGraph()

	// Demonstrate backtracking steps
	demonstrateBacktrackingSteps("input.txt")

	// Solve the CSP
	fmt.Println("\n\n======== SOLVING CSP ========")
	result := solveCSP("input.txt")
	if result != nil {
		result.Print()
	}
}
