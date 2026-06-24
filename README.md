# EasyDNR

A Python library for Distribution Network Reconfiguration (DNR) problems. EasyDNR provides ready-to-use benchmark networks, power-flow-based evaluation, and multiple solvers (random search, simulated annealing, brute-force, and classical heuristics via [DNRlib](https://github.com/)) built on top of [pandapower](https://www.pandapower.org/).

## Installation

```bash
pip install git+https://github.com/baptistechev/EasyDNR-lib.git
```

### Dependencies

Core: `pandapower`, `networkx`, `numpy`, `matplotlib`, [`troma`](https://github.com/baptistechev/TrOMA)

Optional (for `solve_with_dnrlib` / `local_search`): `GridCalEngine`, `pyomo`

## Quick start

```python
from easy_dnr import DNR_Network

# Load a benchmark network
dnr = DNR_Network("33_bus")

# Evaluate the default (all-tie-open) configuration
loss = dnr.evaluate([1]*dnr.n_closed_switches + [0]*dnr.n_dits)
print(f"Default loss: {loss:.4f} MW")

# Random search
best_loss, best_config = dnr.naive_random_search_min_loss(n_samples=1000, seed=42)
print(f"Best loss found: {best_loss:.4f} MW")

# Simulated annealing
best_config = dnr.solve_via_simulated_annealing(T0=0.1, alpha=0.995, seed=42)
print(f"SA loss: {dnr.evaluate(best_config):.4f} MW")

# Classical heuristic (requires DNRlib + GridCalEngine)
best_config = dnr.solve_with_dnrlib(method="baran")
print(f"Baran loss: {dnr.evaluate(best_config):.4f} MW")
```

## Benchmark networks

| Name | Buses | Switches | Tie switches | Source |
|------|-------|----------|--------------|--------|
| `9_bus` | 9 | 10 | 2 | case33bw-derived |
| `12_bus` | 12 | 14 | 3 | case33bw-derived (QPU benchmark) |
| `15_bus` | 15 | 17 | 3 | case33bw-derived |
| `26_bus` | 26 | 30 | 5 | case33bw-derived |
| `33_bus` | 33 | 37 | 5 | pandapower `case33bw` |

All networks are loaded with every line replaced by a switch, so the DNR problem reduces to finding a binary switch-state vector.

## API overview

### `DNR_Network(network_name)`

Main class wrapping a pandapower network for DNR.

**Properties:** `n_switches`, `n_buses`, `n_closed_switches`, `n_dits`

**Evaluation:**
- `evaluate(config)` -- returns total power loss (MW), or `inf` if infeasible
- `evaluate_mcco(config)` -- returns maximisation-oriented score (offset minus loss)
- `evaluate_lagrangian_mcco(config, ...)` -- Lagrangian-relaxed score with penalty weights for constraint violations

**Solvers:**
- `naive_random_search_min_loss(n_samples, seed)` -- uniform random search over feasible switch configurations
- `solve_via_simulated_annealing(T0, alpha, max_iter, patience, seed)` -- simulated annealing with swap-move neighbourhood
- `solve_with_dnrlib(method, initial_config, **kwargs)` -- classical DNR heuristics (`baran`, `merlin`, `salkuti`, `mstgreedy`, `jakus`, `khalil`, `morton`, `taylor`)
- `local_search(config)` -- shortcut for `solve_with_dnrlib(method='baran', initial_config=config)`
- `bruteforce(number_of_one, n_jobs)` -- parallel exhaustive enumeration, writes results to CSV

**Topology checks:**
- `check_radiality(config)` -- True if the configuration forms a tree
- `check_connectivity(config)` -- True if all buses are connected

**Visualisation:**
- `plot_network()` -- plot the full network topology
- `plot_configuration(config)` -- plot with switches applied
- `plot_tree(config)` -- hierarchical tree layout rooted at the slack bus

**Analysis:**
- `analyse_results(results, ...)` -- post-process solver outputs (supports `MatchingPursuitResults` from troma)
- `evaluate_quality(config_or_value, folder)` -- rank a solution against the brute-force CSV (requires prior `bruteforce()` run)

**Representation helpers:**
- `dit_representation_to_switch_vector(dit_repr)` -- convert open-switch positions to a full binary vector
- `switch_vector_to_dit_representation(switch_vector)` -- inverse conversion

## Configuration format

Configurations can be provided in two formats:

- **Switch vector** (length `n_switches`): binary list where `1` = closed, `0` = open
- **Dit representation** (length `n_dits`): list of switch indices that are open

Methods automatically detect the format based on length.

## License

MIT
