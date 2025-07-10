# Simulation Results

This folder contains the CSV files summarizing the output metrics from all simulation runs conducted in the two experiments described in Section 2.3 of the thesis. Each row in these files corresponds to a single stochastic simulation run, including both the experimental setup and the resulting metrics. These results were later used for statistical analysis and figure generation.

---

## Global Simulation Parameters

The following simulation settings were constant across all runs in both experiments:

- Initial number of infections: 200  
- Latency period (days): 20  
- Simulation length: 120 days  

---

## Files

### `experiment1_results.csv`

**Purpose**: Evaluate whether an RL model can learn to apply adaptive interventions in a complex, stochastic environment, compared with static policies.

**Structure**:  
- Cities: Holon and Bene Beraq  
- Intervention strategies:
  1. No intervention  
  2. Full lockdown with Neighborhood Vaccination  
  3. RL-based adaptive intervention policy  
- Compliance rate: fixed at 0.75
- Vaccination strategies: fixed as **Descending** (older age groups prioritized)  
- Simulation runs: 30 stochastic runs per intervention per city  
- Total simulations: 2 cities × 3 intervention strategies × 30 runs = **180**

**Columns in the `experiment1_results.csv` file:**
- `city_name`  
- `intervention_name`  
- `economic_index_cumulative`  
- `total_hospitalizations`  
- `total_deceased`  
- `Rt` (final reproduction number)

---

### `experiment2_results.csv`

**Purpose**: Assess how public compliance levels and vaccination prioritization affect outcomes across two demographically distinct cities.

**Structure**:  
- Cities: Holon and Bene Beraq  
- Compliance levels: 0.35, 0.75, 1.0  
- Vaccination strategies: Ascending (younger first), Descending (older first)  
- Simulation runs: 60 stochastic runs per configuration  
- Total configurations: 2 cities × 3 compliance levels × 2 strategies = 12  
- Total simulations: 12 × 60 = **720**

**Columns in the file:**
- `city_name`  
- `vaccination_strategy` (Ascending / Descending)  
- `compliance_level`  
- `total_hospitalizations`  
- `total_deceased`  
- `cumulative_economic_index`  
- `action_frequencies` (number of days each intervention was applied during the simulation)  
- `Rt` (final reproduction number)

---

These files provide the basis for all quantitative analyses and support reproducibility of the study’s findings.
