"""
Chaos Engineering Theory: From Lyapunov Stability to Distributed System Resilience

This module demonstrates the theoretical foundations of chaos engineering,
including stability analysis, perturbation theory, and failure propagation models.
It bridges abstract mathematical concepts with practical chaos testing strategies.

Author: Infrastructure Complexity Theory Project
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import networkx as nx
from scipy.integrate import odeint
import random
import json
import time
from collections import defaultdict, deque


class SystemState(Enum):
    """Represents possible states in our system model"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    RECOVERING = "recovering"
    FAILED = "failed"


@dataclass
class SystemMetrics:
    """Metrics for measuring system stability"""
    latency_ms: float
    error_rate: float
    throughput: float
    cpu_utilization: float
    memory_utilization: float
    
    def to_vector(self) -> np.ndarray:
        """Convert metrics to state vector for stability analysis"""
        return np.array([
            self.latency_ms / 1000,  # Normalize to seconds
            self.error_rate,
            self.throughput / 10000,  # Normalize
            self.cpu_utilization,
            self.memory_utilization
        ])


class LyapunovStabilityAnalyzer:
    """
    Implements Lyapunov stability theory for distributed systems.
    
    In chaos engineering, we want to prove that our system returns to 
    equilibrium after perturbations. Lyapunov functions help us:
    1. Define stability boundaries
    2. Predict recovery time
    3. Identify critical failure modes
    """
    
    def __init__(self, equilibrium_state: SystemMetrics):
        self.equilibrium = equilibrium_state.to_vector()
        self.stability_threshold = 0.3  # Maximum deviation before instability
        
    def lyapunov_function(self, state: np.ndarray) -> float:
        """
        Compute Lyapunov function V(x) = x^T * P * x
        where x is deviation from equilibrium
        
        For a system to be stable, V(x) > 0 for x ≠ 0 and dV/dt < 0
        """
        deviation = state - self.equilibrium
        # Simple quadratic Lyapunov function
        P = np.diag([2.0, 5.0, 1.0, 3.0, 3.0])  # Weight matrix
        return deviation.T @ P @ deviation
    
    def is_stable(self, current_state: SystemMetrics) -> Tuple[bool, float]:
        """
        Check if system is within stability region
        Returns: (is_stable, lyapunov_value)
        """
        state_vector = current_state.to_vector()
        V = self.lyapunov_function(state_vector)
        return V < self.stability_threshold, V
    
    def predict_recovery_time(self, perturbed_state: SystemMetrics, 
                            decay_rate: float = 0.1) -> float:
        """
        Estimate time to return to equilibrium using exponential decay model
        T = -ln(ε/V₀) / λ where λ is decay rate
        """
        V0 = self.lyapunov_function(perturbed_state.to_vector())
        epsilon = 0.01  # Target proximity to equilibrium
        if V0 < epsilon:
            return 0
        return -np.log(epsilon / V0) / decay_rate


class CascadeFailureModel:
    """
    Models cascading failures in distributed systems using percolation theory
    and graph dynamics. Demonstrates how local failures can lead to 
    system-wide collapse.
    """
    
    def __init__(self, num_nodes: int, connection_probability: float = 0.3):
        # Generate random graph representing service dependencies
        self.graph = nx.erdos_renyi_graph(num_nodes, connection_probability)
        self.node_states = {i: SystemState.HEALTHY for i in range(num_nodes)}
        self.node_load = {i: 0.5 for i in range(num_nodes)}  # Initial load
        self.failure_threshold = 0.9  # Load threshold for failure
        self.propagation_probability = 0.4
        
    def inject_failure(self, node: int) -> Set[int]:
        """
        Inject failure at specific node and compute cascade
        Returns set of all failed nodes
        """
        failed_nodes = {node}
        self.node_states[node] = SystemState.FAILED
        self.node_load[node] = 0
        
        # BFS to propagate failure
        queue = deque([node])
        
        while queue:
            current = queue.popleft()
            neighbors = list(self.graph.neighbors(current))
            
            if not neighbors:
                continue
                
            # Redistribute load from failed node
            load_redistribution = 0.5 / len(neighbors) if neighbors else 0
            
            for neighbor in neighbors:
                if self.node_states[neighbor] == SystemState.HEALTHY:
                    # Increase load on healthy neighbors
                    self.node_load[neighbor] += load_redistribution
                    
                    # Check if overload causes failure
                    if self.node_load[neighbor] > self.failure_threshold:
                        self.node_states[neighbor] = SystemState.FAILED
                        self.node_load[neighbor] = 0
                        failed_nodes.add(neighbor)
                        queue.append(neighbor)
                    # Probabilistic failure propagation (models software bugs, etc.)
                    elif random.random() < self.propagation_probability:
                        self.node_states[neighbor] = SystemState.DEGRADED
                        
        return failed_nodes
    
    def calculate_resilience_metrics(self) -> Dict[str, float]:
        """Calculate system resilience metrics based on current state"""
        total_nodes = len(self.graph.nodes())
        healthy_nodes = sum(1 for s in self.node_states.values() 
                          if s == SystemState.HEALTHY)
        failed_nodes = sum(1 for s in self.node_states.values() 
                         if s == SystemState.FAILED)
        
        # Calculate various resilience metrics
        metrics = {
            "availability": healthy_nodes / total_nodes,
            "failure_rate": failed_nodes / total_nodes,
            "mean_load": np.mean(list(self.node_load.values())),
            "load_variance": np.var(list(self.node_load.values())),
            "largest_component_size": len(max(nx.connected_components(
                self.graph.subgraph([n for n, s in self.node_states.items() 
                                   if s != SystemState.FAILED])), 
                key=len)) / total_nodes if healthy_nodes > 0 else 0
        }
        
        return metrics


class PerturbationStrategies:
    """
    Theoretical framework for chaos experiment design based on 
    perturbation theory and system identification
    """
    
    @staticmethod
    def calculate_perturbation_complexity(system_size: int, 
                                        failure_modes: int) -> Dict[str, float]:
        """
        Calculate the complexity of exhaustive failure testing
        
        For a system with n components and k failure modes per component:
        - State space size: k^n
        - Full test coverage: O(k^n)
        - Random sampling needed for tractability
        """
        state_space_size = failure_modes ** system_size
        
        # Calculate sampling requirements for different confidence levels
        # Using Hoeffding bound for sample complexity
        confidence_levels = [0.90, 0.95, 0.99]
        sample_sizes = {}
        
        for confidence in confidence_levels:
            epsilon = 0.05  # Error margin
            # Hoeffding bound: n >= ln(2/δ) / (2ε²)
            delta = 1 - confidence
            sample_size = np.log(2/delta) / (2 * epsilon**2)
            sample_sizes[f"samples_for_{int(confidence*100)}%_confidence"] = int(sample_size)
        
        return {
            "state_space_size": state_space_size,
            "exhaustive_test_infeasible": state_space_size > 1e6,
            "log_state_space": np.log10(state_space_size),
            **sample_sizes
        }
    
    @staticmethod
    def generate_failure_injection_strategy(graph: nx.Graph, 
                                          budget: int) -> List[Tuple[str, List[int]]]:
        """
        Generate optimal failure injection strategy given limited budget
        Uses graph centrality measures from complex network theory
        """
        strategies = []
        
        # Strategy 1: Target high-centrality nodes (hubs)
        betweenness = nx.betweenness_centrality(graph)
        top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        strategies.append(("high_betweenness", [n for n, _ in top_betweenness[:budget]]))
        
        # Strategy 2: Target nodes that would maximally disconnect the graph
        if len(graph.nodes()) < 100:  # Only for small graphs due to complexity
            articulation_points = list(nx.articulation_points(graph))
            strategies.append(("articulation_points", articulation_points[:budget]))
        
        # Strategy 3: Random sampling (baseline)
        random_nodes = random.sample(list(graph.nodes()), min(budget, len(graph.nodes())))
        strategies.append(("random", random_nodes))
        
        # Strategy 4: Target based on clustering (community boundaries)
        if len(graph.nodes()) > 5:
            communities = nx.community.greedy_modularity_communities(graph)
            boundary_nodes = []
            for i, comm1 in enumerate(communities):
                for j, comm2 in enumerate(communities):
                    if i < j:
                        # Find nodes at community boundaries
                        for node in comm1:
                            if any(neighbor in comm2 for neighbor in graph.neighbors(node)):
                                boundary_nodes.append(node)
            strategies.append(("community_boundaries", boundary_nodes[:budget]))
        
        return strategies


class ChaosExperimentOrchestrator:
    """
    Orchestrates chaos experiments with theoretical guarantees on 
    coverage and system impact
    """
    
    def __init__(self, system_model: CascadeFailureModel):
        self.system = system_model
        self.experiment_history = []
        self.stability_analyzer = None
        
    def run_controlled_experiment(self, failure_nodes: List[int], 
                                 duration_seconds: float = 10.0) -> Dict:
        """
        Run chaos experiment with controlled blast radius
        Implements theoretical bounds on failure propagation
        """
        start_time = time.time()
        initial_metrics = self.system.calculate_resilience_metrics()
        
        # Calculate theoretical blast radius
        blast_radius = self._calculate_blast_radius(failure_nodes)
        
        # Inject failures
        affected_nodes = set()
        for node in failure_nodes:
            affected_nodes.update(self.system.inject_failure(node))
        
        # Measure impact
        post_failure_metrics = self.system.calculate_resilience_metrics()
        
        # Simulate recovery (simplified)
        recovery_start = time.time()
        self._simulate_recovery()
        recovery_time = time.time() - recovery_start
        
        final_metrics = self.system.calculate_resilience_metrics()
        
        experiment_result = {
            "failure_nodes": failure_nodes,
            "affected_nodes": list(affected_nodes),
            "theoretical_blast_radius": blast_radius,
            "actual_blast_radius": len(affected_nodes),
            "initial_availability": initial_metrics["availability"],
            "min_availability": post_failure_metrics["availability"],
            "final_availability": final_metrics["availability"],
            "recovery_time": recovery_time,
            "mean_load_increase": post_failure_metrics["mean_load"] - initial_metrics["mean_load"],
            "largest_component_degradation": initial_metrics["largest_component_size"] - 
                                           post_failure_metrics["largest_component_size"]
        }
        
        self.experiment_history.append(experiment_result)
        return experiment_result
    
    def _calculate_blast_radius(self, failure_nodes: List[int]) -> int:
        """
        Calculate theoretical upper bound on failure propagation
        using percolation theory
        """
        # Simplified model: expected cascade size
        p = self.system.propagation_probability
        avg_degree = np.mean([d for _, d in self.system.graph.degree()])
        
        # Critical threshold from percolation theory
        critical_threshold = 1 / avg_degree if avg_degree > 0 else 1
        
        if p < critical_threshold:
            # Sub-critical: localized failure
            expected_cascade_size = len(failure_nodes) * (1 + p * avg_degree)
        else:
            # Super-critical: potential system-wide failure
            # Use giant component size as upper bound
            expected_cascade_size = len(max(nx.connected_components(self.system.graph), key=len))
        
        return int(expected_cascade_size)
    
    def _simulate_recovery(self):
        """Simulate system recovery after failure"""
        for node, state in self.system.node_states.items():
            if state == SystemState.FAILED:
                # Simple recovery model
                self.system.node_states[node] = SystemState.RECOVERING
                self.system.node_load[node] = 0.3
            elif state == SystemState.DEGRADED:
                self.system.node_states[node] = SystemState.HEALTHY
                self.system.node_load[node] = 0.5


class ComplexityVisualizer:
    """Visualize the complexity and theoretical aspects of chaos engineering"""
    
    @staticmethod
    def plot_stability_basin(analyzer: LyapunovStabilityAnalyzer):
        """
        Visualize the basin of attraction around stable equilibrium
        Shows the region where system naturally recovers
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Create 2D projection of stability basin
        latency_range = np.linspace(0, 2000, 50)
        error_range = np.linspace(0, 1, 50)
        
        stability_map = np.zeros((len(error_range), len(latency_range)))
        lyapunov_map = np.zeros((len(error_range), len(latency_range)))
        
        base_metrics = SystemMetrics(500, 0.01, 5000, 0.5, 0.5)
        
        for i, error in enumerate(error_range):
            for j, latency in enumerate(latency_range):
                test_metrics = SystemMetrics(latency, error, 5000, 0.5, 0.5)
                is_stable, V = analyzer.is_stable(test_metrics)
                stability_map[i, j] = 1 if is_stable else 0
                lyapunov_map[i, j] = min(V, 1.0)  # Cap for visualization
        
        # Plot stability region
        im1 = ax1.contourf(latency_range, error_range, stability_map, 
                          levels=[0, 0.5, 1], colors=['red', 'green'], alpha=0.6)
        ax1.contour(latency_range, error_range, lyapunov_map, levels=10, colors='black', alpha=0.3)
        ax1.set_xlabel('Latency (ms)')
        ax1.set_ylabel('Error Rate')
        ax1.set_title('System Stability Basin\n(Green = Stable, Red = Unstable)')
        ax1.grid(True, alpha=0.3)
        
        # Plot Lyapunov function
        im2 = ax2.contourf(latency_range, error_range, lyapunov_map, levels=20, cmap='RdYlGn_r')
        ax2.set_xlabel('Latency (ms)')
        ax2.set_ylabel('Error Rate')
        ax2.set_title('Lyapunov Function V(x)\n(Lower values = closer to equilibrium)')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_cascade_dynamics(cascade_model: CascadeFailureModel, 
                            failure_history: List[Set[int]]):
        """
        Visualize cascade failure propagation through the system
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Network with failure propagation
        ax = axes[0, 0]
        pos = nx.spring_layout(cascade_model.graph, k=2, iterations=50)
        
        node_colors = []
        for node in cascade_model.graph.nodes():
            if cascade_model.node_states[node] == SystemState.HEALTHY:
                node_colors.append('green')
            elif cascade_model.node_states[node] == SystemState.DEGRADED:
                node_colors.append('yellow')
            elif cascade_model.node_states[node] == SystemState.FAILED:
                node_colors.append('red')
            else:
                node_colors.append('orange')
        
        nx.draw(cascade_model.graph, pos, node_color=node_colors, 
               node_size=300, ax=ax, with_labels=True, font_size=8)
        ax.set_title('System State After Failure Injection\n(Red=Failed, Yellow=Degraded, Green=Healthy)')
        
        # Plot 2: Cascade size distribution
        ax = axes[0, 1]
        cascade_sizes = [len(cascade) for cascade in failure_history]
        if cascade_sizes:
            ax.hist(cascade_sizes, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(cascade_sizes), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(cascade_sizes):.1f}')
            ax.set_xlabel('Cascade Size (# nodes affected)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Cascade Sizes')
            ax.legend()
        
        # Plot 3: Load distribution
        ax = axes[1, 0]
        loads = list(cascade_model.node_load.values())
        ax.hist(loads, bins=20, edgecolor='black', alpha=0.7, color='blue')
        ax.axvline(cascade_model.failure_threshold, color='red', linestyle='--', 
                  label=f'Failure Threshold: {cascade_model.failure_threshold}')
        ax.set_xlabel('Node Load')
        ax.set_ylabel('Number of Nodes')
        ax.set_title('Load Distribution Across System')
        ax.legend()
        
        # Plot 4: Percolation analysis
        ax = axes[1, 1]
        component_sizes = [len(c) for c in nx.connected_components(
            cascade_model.graph.subgraph([n for n, s in cascade_model.node_states.items() 
                                         if s != SystemState.FAILED]))]
        if component_sizes:
            ax.bar(range(len(component_sizes)), sorted(component_sizes, reverse=True))
            ax.set_xlabel('Component Rank')
            ax.set_ylabel('Component Size')
            ax.set_title('Connected Component Sizes\n(Shows system fragmentation)')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_complexity_scaling(max_nodes: int = 20):
        """
        Visualize how chaos engineering complexity scales with system size
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        node_counts = range(2, max_nodes + 1)
        failure_modes = [2, 3, 5]
        
        # Plot 1: State space growth
        ax = axes[0, 0]
        for modes in failure_modes:
            state_spaces = [modes ** n for n in node_counts]
            ax.semilogy(node_counts, state_spaces, marker='o', 
                       label=f'{modes} failure modes/node')
        ax.set_xlabel('Number of System Components')
        ax.set_ylabel('State Space Size (log scale)')
        ax.set_title('Exponential Growth of System State Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Testing coverage requirements
        ax = axes[0, 1]
        confidence_levels = [0.90, 0.95, 0.99]
        epsilon = 0.05
        
        for confidence in confidence_levels:
            delta = 1 - confidence
            sample_size = np.log(2/delta) / (2 * epsilon**2)
            ax.axhline(sample_size, linestyle='--',
                      label=f'{int(confidence*100)}% confidence ({int(sample_size)} samples)')

        ax.set_xlabel('Number of System Components')
        ax.set_ylabel('Required Test Samples')
        ax.set_title('Sample Complexity for Chaos Testing\n(Hoeffding Bound, ε=0.05)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Expected cascade size vs propagation probability
        ax = axes[1, 0]
        propagation_probs = np.linspace(0, 0.6, 50)
        avg_degree = 4  # Example average degree

        cascade_sizes_subcritical = []
        cascade_sizes_critical = []

        for p in propagation_probs:
            critical_threshold = 1 / avg_degree
            if p < critical_threshold:
                # Sub-critical regime: localized failure
                expected_size = 1 + p * avg_degree
            else:
                # Super-critical regime: giant component
                # Using mean-field approximation
                expected_size = 1 / (1 - p * avg_degree) if p * avg_degree < 1 else 50

            cascade_sizes_subcritical.append(min(expected_size, 50))

        ax.plot(propagation_probs, cascade_sizes_subcritical, 'b-', linewidth=2)
        ax.axvline(1/avg_degree, color='r', linestyle='--',
                  label=f'Critical threshold: {1/avg_degree:.3f}')
        ax.axhspan(0, 5, alpha=0.2, color='green', label='Localized failures')
        ax.axhspan(5, 50, alpha=0.2, color='red', label='System-wide cascades')
        ax.set_xlabel('Propagation Probability')
        ax.set_ylabel('Expected Cascade Size')
        ax.set_title('Phase Transition in Failure Propagation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Recovery time vs system size
        ax = axes[1, 1]
        system_sizes = np.array(list(node_counts))

        # Different recovery models
        recovery_linear = system_sizes * 0.5
        recovery_log = np.log(system_sizes) * 5
        recovery_sqrt = np.sqrt(system_sizes) * 2

        ax.plot(system_sizes, recovery_linear, 'r-', marker='o', label='O(n) - Sequential')
        ax.plot(system_sizes, recovery_log, 'g-', marker='s', label='O(log n) - Hierarchical')
        ax.plot(system_sizes, recovery_sqrt, 'b-', marker='^', label='O(√n) - Parallel')

        ax.set_xlabel('Number of System Components')
        ax.set_ylabel('Recovery Time (arbitrary units)')
        ax.set_title('Recovery Time Complexity')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# Demonstration and Example Usage
def demonstrate_chaos_engineering_theory():
    """
    Complete demonstration of the chaos engineering theory toolkit
    """
    print("=" * 80)
    print("CHAOS ENGINEERING THEORY DEMONSTRATION")
    print("=" * 80)

    # Part 1: Lyapunov Stability Analysis
    print("\n1. LYAPUNOV STABILITY ANALYSIS")
    print("-" * 80)

    equilibrium = SystemMetrics(
        latency_ms=500,
        error_rate=0.01,
        throughput=5000,
        cpu_utilization=0.5,
        memory_utilization=0.5
    )

    analyzer = LyapunovStabilityAnalyzer(equilibrium)

    # Test different perturbations
    test_states = [
        ("Minor latency spike", SystemMetrics(600, 0.01, 5000, 0.5, 0.5)),
        ("Error rate increase", SystemMetrics(500, 0.05, 5000, 0.5, 0.5)),
        ("Severe degradation", SystemMetrics(1500, 0.15, 3000, 0.8, 0.8)),
    ]

    for name, state in test_states:
        is_stable, V = analyzer.is_stable(state)
        recovery_time = analyzer.predict_recovery_time(state)

        print(f"\n{name}:")
        print(f"  Lyapunov value: {V:.4f}")
        print(f"  Stable: {is_stable}")
        print(f"  Predicted recovery time: {recovery_time:.2f} seconds")

    # Part 2: Cascade Failure Modeling
    print("\n\n2. CASCADE FAILURE MODELING")
    print("-" * 80)

    cascade_model = CascadeFailureModel(num_nodes=50, connection_probability=0.3)

    print(f"System initialized:")
    print(f"  Nodes: {len(cascade_model.graph.nodes())}")
    print(f"  Edges: {len(cascade_model.graph.edges())}")
    print(f"  Average degree: {np.mean([d for _, d in cascade_model.graph.degree()]):.2f}")

    # Inject failure at a random node
    initial_failure = 5
    affected = cascade_model.inject_failure(initial_failure)

    print(f"\nFailure injection at node {initial_failure}:")
    print(f"  Total nodes affected: {len(affected)}")
    print(f"  Cascade amplification: {len(affected)}x")

    metrics = cascade_model.calculate_resilience_metrics()
    print(f"\nResilience metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Part 3: Perturbation Strategies
    print("\n\n3. PERTURBATION STRATEGY ANALYSIS")
    print("-" * 80)

    complexity = PerturbationStrategies.calculate_perturbation_complexity(
        system_size=10,
        failure_modes=3
    )

    print("Chaos testing complexity:")
    for key, value in complexity.items():
        print(f"  {key}: {value}")

    strategies = PerturbationStrategies.generate_failure_injection_strategy(
        cascade_model.graph,
        budget=5
    )

    print("\nFailure injection strategies:")
    for strategy_name, nodes in strategies:
        print(f"  {strategy_name}: {nodes[:5]}")

    # Part 4: Controlled Experiment
    print("\n\n4. CONTROLLED CHAOS EXPERIMENT")
    print("-" * 80)

    orchestrator = ChaosExperimentOrchestrator(cascade_model)
    result = orchestrator.run_controlled_experiment([3, 7, 12])

    print("Experiment results:")
    for key, value in result.items():
        if isinstance(value, list):
            print(f"  {key}: {value[:5]}{'...' if len(value) > 5 else ''}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("Demonstration complete. Generating visualizations...")
    print("=" * 80)

    return analyzer, cascade_model, orchestrator


if __name__ == "__main__":
    # Run demonstration
    analyzer, cascade_model, orchestrator = demonstrate_chaos_engineering_theory()

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Stability basin plot
    fig1 = ComplexityVisualizer.plot_stability_basin(analyzer)
    fig1.savefig('stability_basin.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: stability_basin.png")

    # Cascade dynamics plot
    failure_history = [cascade_model.inject_failure(i) for i in range(5)]
    fig2 = ComplexityVisualizer.plot_cascade_dynamics(cascade_model, failure_history)
    fig2.savefig('cascade_dynamics.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: cascade_dynamics.png")

    # Complexity scaling plot
    fig3 = ComplexityVisualizer.plot_complexity_scaling(max_nodes=20)
    fig3.savefig('complexity_scaling.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: complexity_scaling.png")

    print("\nAll visualizations generated successfully!")
    print("Run with Python to see the chaos engineering theory in action.")
