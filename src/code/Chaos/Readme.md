

---

# Chaos Engineering Theory Toolkit

A Python module that connects mathematical stability analysis with practical chaos engineering. It provides tools for studying Lyapunov stability, cascading failures, perturbation complexity, and experiment orchestration for distributed systems.

## Features

* **Lyapunov Stability Analysis**
  Evaluate system resilience, compute Lyapunov functions, and estimate recovery times.

* **Cascade Failure Modeling**
  Simulate percolation-driven failure propagation across service dependency graphs.

* **Perturbation & Experiment Design**
  Estimate failure-space complexity, generate failure-injection strategies based on graph metrics, and apply sampling theory to chaos testing.

* **Chaos Experiment Orchestration**
  Run controlled experiments, track impact, compute blast-radius bounds, and simulate recovery.

* **Visualization Tools**
  Plot stability basins, cascade dynamics, load distributions, component fragmentation, and complexity scaling.

## Installation

```
pip install numpy scipy matplotlib networkx
```

## Quick Start

**Stability analysis**

```python
eq = SystemMetrics(500, 0.01, 5000, 0.5, 0.5)
analyzer = LyapunovStabilityAnalyzer(eq)

stable, v = analyzer.is_stable(eq)
recovery = analyzer.predict_recovery_time(eq)
```

**Cascade failures**

```python
model = CascadeFailureModel(num_nodes=50, connection_probability=0.3)
failed = model.inject_failure(3)
metrics = model.calculate_resilience_metrics()
```

**Chaos experiments**

```python
orchestrator = ChaosExperimentOrchestrator(model)
result = orchestrator.run_controlled_experiment([3, 7])
print(result)
```

**Visualization**

```python
fig = ComplexityVisualizer.plot_stability_basin(analyzer)
fig.show()
```

## Concepts Covered

* Lyapunov functions and basins of attraction
* Percolation theory and load redistribution
* Failure propagation bounds
* Graph-theoretic targeting strategies (betweenness, articulation points, community boundaries)
* Sampling complexity (Hoeffding bounds)
* Component fragmentation and resilience metrics

## Project Structure

* `SystemMetrics` — Normalized state vectors for stability analysis
* `LyapunovStabilityAnalyzer` — Stability tests and recovery predictions
* `CascadeFailureModel` — Failure injection and cascade simulation
* `PerturbationStrategies` — Failure-injection planning and test-space analytics
* `ChaosExperimentOrchestrator` — Controlled experiments with telemetry
* `ComplexityVisualizer` — Matplotlib-based exploration of system behavior

## License

MIT License.

---
