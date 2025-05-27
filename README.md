**Insights from Reinforcement Learning and Individual-Based Model Simulations on Population Compliance and Policy Optimization during COVID-19**

This repository contains the official codebase accompanying the research article:

“Insights from Reinforcement Learning and Individual-Based Model Simulations on Population Compliance and Policy Optimization during COVID-19”

**Abstract**

We aim to develop and evaluate a novel framework that integrates RL with existing IBMs that have been previously developed and validated for simulating the spread of COVID-19.
Our primary objective is to assess whether an RL agent can learn adaptive, data-driven intervention policies that maximize economic outcomes while simultaneously reducing hospitalization and mortality rates.

As a case study, we simulate outbreaks under the demographic conditions of two Israeli cities:
Holon, where approximately a quarter of the population is under 18 and the average household size is 2.8; and
Bene Beraq, a younger and denser city, where nearly half the population is under 18 and the average household size is 4.5.

Our model supports the simulation of heterogeneous populations and enables the evaluation of intervention strategies within specific demographic contexts. This stands in contrast to deterministic compartmental models, which divide the population into homogeneous compartments and assume identical behavior within each group.

**Requirements**

Install with:

pip install -r requirements.txt

Required packages:

aiofiles==24.1.0
Flask==1.1.2
Flask_AutoIndex==0.6.6
Jinja2==2.11.2
keras==2.13.1
matplotlib==3.4.3
numpy==1.24.3
pandas==1.3.4
pyfunctional==1.5.0
pytest==6.2.4
sanic==24.6.0
scipy==1.4.1
seaborn==0.11.2
tabulate==0.9.0
tensorflow==2.13.1
tqdm==4.64.1
xlrd==2.0.1

How to Run

Make sure you’re in the project root:

cd Master_SiRL_Coderona

Train:

python src/_run.py --mode train --config config.json

Evaluate:

Set in config.json:

"checkpoint_date": "YYYY_MM_DD_HH_MM_SS"

Then run:

python src/_run.py --mode test --config config.json



Features
	•	Age-specific SEIR dynamics via individual-based simulation (includes custom Coderona-virus model)
	•	DQN agent with LSTM architecture for sequential decision making
	•	Multiple policy interventions via self.act_dict:

{
  0: scenarios.Empty_scenario(),          # No action
  
  1: house_interventions,                 # Household vaccination + lockdown
  
  2: global_interventions,                # General vaccination + curfew
  
  3: upd_2_3,                             # Gen. vaccination + symptomatic isolation
  
  4: [act.social_distance()],            # Social distancing
  
  5: upd_1_5,                             # Household vac + school closure
  
  6: upd_1_6,                             # Household vac + workplace closure
  
  7: upd_1_7,                             # Household vac + elderly quarantine
  
  8: [act.household_isolation()],        # Household-level isolation
  
  9: [act.curfew()],                      # Curfew
  
  10: [act.lockdown(city_name='all')]    # Full lockdown
}

**Daily economic index calculated by:**

vac_decrease_coef = decrease_coef_[0]  # VPRF
decrease_coef = decrease_coef_[1]      # PPF
vsr = (S + V * vac_decrease_coef + R) * decrease_coef
economic_index += delta * vsr - alpha * I - beta * H - gamma * D

Configurable population compliance levels



**Training Environment**

Training was conducted on:

NVIDIA A100 80GB PCIe GPU

This GPU is optimized for deep learning and large-scale individual-based simulations.

Note: The code may run on smaller GPUs or CPUs, but expect slower performance.


Citation (BibTeX)

@article{your2025paper,
  title={Insights from Reinforcement Learning and Individual-Based Model Simulations on Population Compliance and Policy Optimization during COVID-19},
  author={Your Name},
  year={2025},
  url={https://github.com/yossi122/Master_SiRL_Coderona}
}

**Branch Protection**

The main branch is protected using GitHub’s classic branch protection rules, ensuring the reproducibility and integrity of this research.

Optimal policy fighting covid-19 one model at a time

config.json

Pandemic parameters

Tune the following:
- `vaccinate_per_day_persons`
- `vaccinate_per_day_household`
- `initial_num_infected`
- `linked_immune_age`:
  - `min_age`
  - `max_age`
- `compliance`
- `"order"`: [ASCENDING, DESCENDING]

Simulation parameters

Tune the following:
- `latency days_n`
- `intervention_duration`
- `days_bound`

Model Saving Process (.h5 Format)

During the training process, after the model has been compiled and trained, it is important to save the model's architecture, weights, and training configuration to allow for reuse or further training later.

In this project, the model is saved in HDF5 (.h5) format, which is widely supported by machine learning frameworks such as TensorFlow and Keras.

### The saving process involves:

1. **Model Training**:  
   The model learns from the dataset using a specified optimizer, loss function, and evaluation metrics.

2. **Checkpointing**:  
   The model is saved during or after training. This includes both the model's structure and its weights.

3. **Saving the Model**:  
   Saves the following:
   - The model architecture
   - The learned weights (parameters)
   - The optimizer's state (if needed)
   - The training configuration

4. **Restoring the Model**:  
   The saved model can be loaded later for inference or continued training.

This process ensures reproducibility and cross-platform usability of the trained model.


