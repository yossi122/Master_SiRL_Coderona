`This repository contains the official codebase accompanying the research article:`

**`“Insights from Reinforcement Learning and Individual-Based Model Simulations on Population Compliance and Policy Optimization during COVID-19”`**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15715520.svg)](https://doi.org/10.5281/zenodo.15715520) ![Python package](https://github.com/yossi122/Master_SiRL_Coderona/actions/workflows/python-package.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Top Language](https://img.shields.io/github/languages/top/yossi122/Master_SiRL_Coderona)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen)](https://github.com/yossi122/Master_SiRL_Coderona#readme)
[![version](https://img.shields.io/badge/version-1.0.0-blue)](https://doi.org/10.5281/zenodo.15715520)
![Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-yes-blue)




**Abstract**

We aim to develop and evaluate a novel framework that integrates Reinforcement Learning (RL) with existing Individual-Based Model (IBM) that have been previously developed and validated for simulating the spread of COVID-19 (Coderona). Our primary objective is to assess whether an RL agent can learn adaptive, data-driven intervention policies that maximize economic outcomes while simultaneously reducing hospitalization and mortality rates.

As a case study, we simulate outbreaks under the demographic conditions of two Israeli cities:

	•	Holon: ~25% under 18 years old, average household size: 2.8
 
	•	Bene Beraq: ~50% under 18 years old, average household size: 4.5

Our model supports the simulation of heterogeneous populations and enables the evaluation of intervention strategies within specific demographic contexts. This stands in contrast to deterministic compartmental models, which assume homogeneous behavior.


**Requirements**

Install dependencies using:

`pip install -r requirements.txt`

Required packages include:

	•	aiofiles==24.1.0
 
	•	Flask==1.1.2
 
	•	Flask_AutoIndex==0.6.6
 
	•	Jinja2==2.11.2
 
	•	keras==2.13.1
 
	•	matplotlib==3.4.3
 
	•	numpy==1.23.5
 
	•	pandas==1.3.4
 
	•	pyfunctional==1.5.0
 
	•	pytest==6.2.4
 
	•	sanic==24.6.0
 
	•	scipy==1.10.1
 
	•	seaborn==0.11.2
 
	•	tabulate==0.9.0
 
	•	tensorflow==2.13.1
 
	•	tqdm==4.64.1
 
	•	xlrd==2.0.1



How to Run and Train the Model

**Step 1** – Clone the Repository

Clone or download this repository to your local machine or runtime environment.

**Step 2** – Install Dependencies

`pip install -r requirements.txt`

**Step 3** – Configure Simulation Parameters

Edit the `config.json` file to specify the simulation and pandemic-related parameters:

Pandemic and Simulation Parameters:


Below is a description of all major configuration options in the config.json file:

 **Epidemic & Intervention Setup**
 
	•	latency (int)
Number of days before the first intervention is applied.
This is the natural spread phase (e.g., 20 or 30).

	•	days_n (int)
Total number of simulation days, including both latency and policy phases (e.g., 170).

	•	intervention_duration (int)
Number of days each selected intervention remains active before the agent chooses a new one (e.g., 7, 14, 30).

	•	days_bound (int)
Maximum number of days allowed per episode. Caps the full simulation run time.

 **RL Training Parameters**
 
	•	train_epochs (int)
Number of training episodes (i.e., full simulation loops where the agent interacts and learns).

 Vaccination Strategy
 
	•	vaccinate_per_day_persons (int)
Number of individual persons vaccinated per day (typical range: 0 to 300).

	•	vaccinate_per_day_household (int)
Number of entire households vaccinated per day (typical range: 0 to 100).

 Initial Conditions & Population Structure
 
	•	initial_num_infected (int)
Number of people who start infected at the beginning of the simulation.

	•	linked_immune_age (object: min_age, max_age)
Age range for individuals considered immune.

Example:

"linked_immune_age": {
  "min_age": 12,
  "max_age": 99
}


	•	compliance (float, range 0.0–1.0)
Population compliance rate with interventions (e.g., 0.75 = 75% compliance).

	•	order (list: [ASCENDING, DESCENDING])
Determines priority order for vaccination or interventions (e.g., age-based prioritization).




**Step 4** – Run Training

`python src/_run.py --mode train --config config.json`

**Step 5** – Check Output Directory

After training completes, a new folder is created under outputs/ with a timestamp in the format:

`YYYY_MM_DD_HH_MM_SS`

The folder contains:

	1.	Trained model weights:
	       •	Bene_Beraq_model.weight.h5
	       •	Holon_model.weight.h5
 
	2.	Training metadata:
	       •	params.json containing:
	
```
{
  “episode”: <last_episode>,
  “epsilon”: <final_epsilon_value>,
  “action_select_counter”: {
  “0”: ,
  “1”: ,
  …
  “10”: 
  }
}
```





Continuing Training from a Checkpoint

To continue training:

	1.	Set the checkpoint_date field in config.json:

"checkpoint_date": `"YYYY_MM_DD_HH_MM_SS"`

	2.	Update the train_epochs field to reflect the new total:

"train_epochs": previous_value + additional_episodes

	3.	Re-run the training script. Training will resume from the saved checkpoint.

**Note:** The trained model file (model.h5) is not included in this repository due to its large binary size and GitHub’s file hosting limitations. However, the model can be fully re-trained using the included training scripts and configuration files.

Colab Notebooks and Statistical Analysis

This repository includes three Google Colab notebooks used to generate the complete set of statistical analyses and visualizations referenced in the article:

	`•	Statistical1_Analysis_SiRL_Coderona-2.ipynb`
 
	`•	Statistical2_Analysis_SiRL_Coderona.ipynb`
 
	`•	Statistic_evaluation_RL.ipynb`

These notebooks contain:

	•	Reproduction of all figures included in the published article
	•	Descriptive and inferential statistical evaluations of simulation outcomes
	•	Performance comparisons between demographic settings and policy interventions

Supplementary Visualizations (Not Included in the Published Article):

	•	Additional visual summaries of policy effects over time
	•	Detailed epidemic and economic indicators
	•	Extended sensitivity and exploratory analysis beyond the main scope of the paper

Features

	•	Age-specific SEIR dynamics via individual-based simulation (includes custom “Coderona-virus” model)
 
	•	DQN agent with LSTM architecture for sequential decision making
 
	•	Demographic-aware policy interventions via self.act_dict, including:
 
	•	Social distancing
	•	Vaccination (household/global)
	•	Curfews, lockdowns, school/workplace closures
	•	Daily economic index computed using model-specific parameters and health states
	•	Adjustable compliance rates and age-specific immunity

**Example Visualizations from Article**

A dedicated folder `examples/` contains:

	•	Two CSV data files:
 
	•	bene-baraq_metrics_output.csv
 
	•	holon_metrics_output.csv
 
These contain preprocessed simulation metrics for the cities of Bene Beraq and Holon, used in visualizations.

	•	One Google Colab notebook:
	•	Analysis_and_Visualization_Holon_Bene_Beraq.ipynb


**Training Environment**

Training was conducted on:
	•	NVIDIA A100 80GB PCIe GPU

The code may run on smaller GPUs or CPUs, but performance will be significantly slower.

**Branch Protection**

The main branch is protected via GitHub’s classic branch protection rules to ensure the reproducibility and integrity of the published results.



Tagline

Optimal policy, fighting COVID-19 — one model at a time.




