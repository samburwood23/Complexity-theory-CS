"""
Simplified Chaos Engineering Example: Finding the Percolation Threshold

This script demonstrates how distributed systems exhibit phase transitions
in failure propagation, similar to percolation in physics.

Below a critical threshold, failures remain localized.
Above it, system-wide cascades become possible.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Set, List


def simulate_cascade(graph: nx.Graph, 
                    initial_failure: int, 
                    propagation_prob: float) -> Set[int]:
    """
    Simulate a cascade failure starting from one node.
    
    Each failed node has 'propagation_prob' chance of failing its neighbors.
    This models how bugs, overload, or misconfigurations spread.
    """
    failed = {initial_failure}
    newly_failed = {initial_failure}
    
    while newly_failed:
        next_failures = set()
        for node in newly_failed:
            for neighbor in graph.neighbors(node):
                if neighbor not in failed:
                    # Probabilistic failure propagation
                    if np.random.random() < propagation_prob:
                        next_failures.add(neighbor)
        
        failed.update(next_failures)
        newly_failed = next_failures
    
    return failed


def find_percolation_threshold(graph: nx.Graph, 
                              num_trials: int = 100) -> tuple:
    """
    Empirically find the percolation threshold where cascades go from
    localized to system-wide.
    
    Theory predicts: p_critical ≈ 1 / average_degree
    """
    probabilities = np.linspace(0, 0.5, 30)
    avg_cascade_sizes = []
    
    total_nodes = len(graph.nodes())
    
    for p in probabilities:
        cascade_sizes = []
        
        for _ in range(num_trials):
            # Start from random node
            start_node = np.random.choice(list(graph.nodes()))
            cascade = simulate_cascade(graph, start_node, p)
            cascade_sizes.append(len(cascade) / total_nodes)
        
        avg_cascade_sizes.append(np.mean(cascade_sizes))
    
    # Theoretical prediction
    avg_degree = np.mean([d for _, d in graph.degree()])
    theoretical_threshold = 1 / avg_degree if avg_degree > 0 else 0
    
    return probabilities, avg_cascade_sizes, theoretical_threshold


def main():
    """
    Demonstrate percolation phase transition in three different network types
    """
    print("=" * 60)
    print("CHAOS ENGINEERING: PERCOLATION THRESHOLD DEMONSTRATION")
    print("=" * 60)
    
    # Create three different network topologies
    networks = [
        ("Random Network", nx.erdos_renyi_graph(100, 0.05)),
        ("Small World", nx.watts_strogatz_graph(100, 6, 0.1)),
        ("Scale-Free", nx.barabasi_albert_graph(100, 3))
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (name, graph) in enumerate(networks):
        ax = axes[idx]
        
        print(f"\n{name}:")
        print(f"  Nodes: {len(graph.nodes())}")
        print(f"  Edges: {len(graph.edges())}")
        
        avg_degree = np.mean([d for _, d in graph.degree()])
        print(f"  Average degree: {avg_degree:.2f}")
        
        # Find percolation threshold
        probs, cascade_sizes, theoretical = find_percolation_threshold(graph, num_trials=50)
        
        print(f"  Theoretical threshold: {theoretical:.3f}")
        
        # Find empirical threshold (where cascade size jumps)
        gradient = np.gradient(cascade_sizes)
        empirical_idx = np.argmax(gradient)
        empirical_threshold = probs[empirical_idx]
        print(f"  Empirical threshold: {empirical_threshold:.3f}")
        
        # Plot results
        ax.plot(probs, cascade_sizes, 'b-', linewidth=2, label='Empirical')
        ax.axvline(theoretical, color='r', linestyle='--', 
                  label=f'Theory: {theoretical:.3f}')
        ax.axvline(empirical_threshold, color='g', linestyle=':', 
                  label=f'Empirical: {empirical_threshold:.3f}')
        
        ax.set_xlabel('Propagation Probability')
        ax.set_ylabel('Avg Cascade Size (fraction of system)')
        ax.set_title(f'{name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 1])
    
    plt.suptitle('Percolation Phase Transitions in Different Network Topologies', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('percolation_thresholds.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization: percolation_thresholds.png")
    
    # Practical implications
    print("\n" + "=" * 60)
    print("PRACTICAL IMPLICATIONS FOR CHAOS ENGINEERING:")
    print("=" * 60)
    print("""
1. BLAST RADIUS PREDICTION:
   - Below threshold: Failures affect O(1) nodes
   - Above threshold: Failures affect O(n) nodes
   
2. RESILIENCE STRATEGIES:
   - Reduce coupling (lower average degree)
   - Add circuit breakers (limit propagation probability)
   - Isolate critical services (graph partitioning)
   
3. CHAOS EXPERIMENT DESIGN:
   - Start with p << p_critical for safety
   - Gradually increase to find actual threshold
   - Different topologies have different resilience
   
4. MONITORING METRICS:
   - Track propagation probability (error rates, timeout cascades)
   - Monitor average service dependencies
   - Alert when approaching critical threshold
    """)


if __name__ == "__main__":
    main()
