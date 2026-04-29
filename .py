import random
import time
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# ======================
# 1. ALGORITHM IMPLEMENTATIONS
# ======================

def knapsack_greedy(capacity, weights, values):
    """Solves the 0/1 knapsack problem using the Greedy (Value/Weight Ratio) heuristic."""
    items = list(zip(weights, values))
    # Calculate ratio (value/weight) for each item
    items_with_ratio = [(w, v, v/w) for w, v in items]
    # Sort by ratio descending
    items_with_ratio.sort(key=lambda x: x[2], reverse=True)
    
    total_value = 0
    total_weight = 0
    for w, v, ratio in items_with_ratio:
        if total_weight + w <= capacity:
            total_weight += w
            total_value += v
    return total_value

def knapsack_dp(capacity, weights, values):
    """Solves the 0/1 knapsack problem optimally using Dynamic Programming."""
    n = len(weights)
    # Create a DP table with (n+1) rows and (capacity+1) columns
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Build the table bottom-up
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                # Max of (including item i-1) vs (excluding it)
                dp[i][w] = max(values[i-1] + dp[i-1][w - weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]

# ======================
# 2. PROFILING & DATA COLLECTION
# ======================

def measure_performance(func, capacity, weights, values):
    """Measures the runtime of a given knapsack algorithm."""
    start_time = time.perf_counter_ns()
    result = func(capacity, weights, values) # This runs the algorithm
    end_time = time.perf_counter_ns()
    duration_ms = (end_time - start_time) / 1_000_000.0 # Convert to milliseconds
    return duration_ms, result

def generate_problem_sizes():
    """Generates a range of problem sizes (number of items) for testing.
       Adjust the max_items based on your computer's processing power.
       WARNING: DP time is O(N*W). Use a relatively small max_capacity (e.g., 200-500)
       to keep runtime reasonable for large item counts."""

    # Vary the number of items, keep capacity moderate to avoid O(N*W) explosion
    sizes = list(range(10, 210, 20))  # 10, 30, 50 ... 190
    max_capacity = 300
    results = {'size': [], 'greedy_time': [], 'dp_time': [], 'greedy_value': [], 'dp_value': []}
    
    print("Running benchmarks... (This may take a minute or two for the largest sizes)")
    for n in sizes:
        # Generate consistent random data for each test iteration
        random.seed(42)
        weights = [random.randint(5, 50) for _ in range(n)]
        values = [random.randint(10, 100) for _ in range(n)]
        
        # 1. Test Dynamic Programming (The slower one)
        dp_time, dp_val = measure_performance(knapsack_dp, max_capacity, weights, values)
        
        # 2. Test Greedy (The fast one)
        greedy_time, greedy_val = measure_performance(knapsack_greedy, max_capacity, weights, values)
        
        results['size'].append(n)
        results['greedy_time'].append(greedy_time)
        results['dp_time'].append(dp_time)
        results['greedy_value'].append(greedy_val)
        results['dp_value'].append(dp_val)
        
        # Calculate optimality of greedy
        optimality_pct = (greedy_val / dp_val) * 100 if dp_val > 0 else 0
        print(f"  n={n:3d} | Greedy: {greedy_time:6.2f}ms | DP: {dp_time:8.2f}ms | Greedy Optimality: {optimality_pct:.1f}%")
    
    return results

# ======================
# 3. VISUALIZATION & OUTPUT
# ======================

def plot_results(results):
    """Creates publication-quality plots for runtime and solution quality."""
    sizes = results['size']
    
    # --- Plot 1: Runtime Comparison (Log Scale is best for this disparity) ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sizes, results['greedy_time'], 'o-', color='green', label='Greedy Algorithm', linewidth=2)
    plt.plot(sizes, results['dp_time'], 's-', color='red', label='Dynamic Programming', linewidth=2)
    plt.xlabel('Number of Items (n)', fontsize=12)
    plt.ylabel('Average Runtime (milliseconds)', fontsize=12)
    plt.title('Algorithm Runtime Scaling', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    # Use log scale if the difference is huge, else keep linear
    if max(results['dp_time']) / (max(results['greedy_time']) + 1) > 100:
        plt.yscale('log')
        plt.ylabel('Average Runtime (milliseconds - Log Scale)', fontsize=12)

    # --- Plot 2: Greedy Optimality (% of DP value) ---
    plt.subplot(1, 2, 2)
    optimality_pct = [ (g/d)*100 if d>0 else 0 for g,d in zip(results['greedy_value'], results['dp_value']) ]
    plt.plot(sizes, optimality_pct, 'd-', color='blue', linewidth=2)
    plt.axhline(y=100, color='black', linestyle='--', label='100% Optimal (DP)')
    plt.xlabel('Number of Items (n)', fontsize=12)
    plt.ylabel('Greedy Solution Quality (% of Optimal)', fontsize=12)
    plt.title('Greedy Algorithm: Performance vs. Optimal', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(80, 105)  # Focus on the relevant range

    plt.tight_layout()
    plt.savefig('knapsack_analysis.png', dpi=300)
    plt.show()

def print_table(results):
    """Prints a formatted table of the results."""
    table_data = []
    for i in range(len(results['size'])):
        opt_pct = (results['greedy_value'][i] / results['dp_value'][i]) * 100
        table_data.append([
            results['size'][i],
            f"{results['greedy_time'][i]:.2f}",
            f"{results['dp_time'][i]:.2f}",
            f"{results['greedy_time'][i]/results['dp_time'][i]:.2f}x",
            f"{opt_pct:.1f}%"
        ])
    
    headers = ["Items (n)", "Greedy (ms)", "DP (ms)", "Speedup Factor", "Greedy Optimality"]
    print("\n" + "="*80)
    print("KNAPSACK ALGORITHM PERFORMANCE ANALYSIS")
    print("="*80)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    print("Starting Knapsack Algorithm Analysis...")
    print("This script compares the Greedy heuristic vs. Dynamic Programming.\n")
    
    # 1. Run the benchmarks
    analysis_data = generate_problem_sizes()
    
    # 2. Display the results in a table
    print_table(analysis_data)
    
    # 3. Generate the plots for your presentation
    plot_results(analysis_data)
    
    print("\nAnalysis complete! 'knapsack_analysis.png' has been saved to your current directory.")
