## LIST OF CHANGES:

Main rules of this DQN appendix for CODERONA project:
* all totally new scripts would starts with _ symbol. Like _run.py, _DQN.py
* all new functionality like DQN model, economic/statistic classes would exists only in such new scripts with no addition of any code to existing scripts
* some reformation of existing function were provided with addition function with same name. Old function currently leaved unchanged with _old prefix. Example: run_simulation_old()


### Precise list of changes:
* **_run.py**: reformatted version of main.py
  * expanded list of stored parameters in **generate_scenario_name()**. From totally full list (basically it saved as experiment folder name) of parameters agent will get some parameters later
  * all experiment/job parameters moved into **config.json** file
       ```python
         { 
           "ExtensionsNamelst": ["ImmuneByAgeExtension"], 
           "target_immune_percentage": [0.8], 
         }
  * all parameter changes handlers were added. Now all changes can be done in json file only 
  * added **wandb** login 
  * current main nested loops looks more complicated but no need to change anything in this place
  * TO BE CONTINUED
  
  * **run_utils.py**: main changes in function create_and_run_simulation() of class SimpleJob. This script uses classes from the new _DQN file, which stores all service classes.
    * The new structure create_and_run_simulation() involves the following steps:
      - loading global parameters of the calculation from the config - the number of epochs of training and conditions of stopping each epoch: 
        - train_epochs: the larger this number is, the more chances the agent has to explore all possible combinations of states and find a suitable final strategy in theory
        - stop_condition: in the current form this is a cumulative check of a set of conditions; if the result is positive, the agent should continue studying the states within the current episode. We can play around changing percents of  
         __{"infected_population": 0.01, "vaccinated_population": 0.01}__
    *  The next step is to create an object of the __Runner__ class. This is a new class with both new and old functions (wrapped in this form).
      * definition of global parameters of interventions is realized through parsing of the directory name, which contains everything necessary
      * population initialization. This should be done for each epoch of learning - so that the initial values of the experiment are as similar as possible for each epoch and learning progresses. 
    * The next object of the class is __EpidemicStats__. The main purpose is to compute statistics for each DiseaseState + vaccinated to form a vector of states for each training step. It can store these statistics either in absolute number or as a percentage of the total population.
    * __Agent__ object is the main class aimed at training in the current environment (states obtained from CODERONA modeling). 

  Thus the structure of create_and_run_simulation lends itself to the following logic:
* initialize all objects of all classes needed for:
  * statistic counting,
  * interacting with CODERONA (e.g. initializing a population, running a simulation to get a new batch of statistics)

* main changes here - split of code for population init purposes each train episode (method Runner.population_init())

  
## DQN (to be updated) 


The basic functions have been kept unchanged, so that you can revert to the basic calculation if needed, running from the “main” script. However, it is possible to calculate in the basic way, but in the new code by selecting “experiment_type”. If “experiment_type” = “DQN”, the experiment always starts with an empty script and the RL model is connected. If “experiment_type” = “Basic”, the basic code is started with calculations within the updated code structure.



## Config description
* __"experiment_type"__ - styles of code running. "Basic" - just run CODERONA as was, without RL. "DQN" - starts RL
* __"experiment_mode"__ - style of calculation parameters usage. If 'test' - use only first pair of linked parameters, like {"immune_source": "BY_NEIGHBORHOOD", "min_age": 18} while config can contains all possible desired combinations 
* __"linked_immune_compliance"__ - subfield for all paired age parameters [{"target_immune_percentage": 0.8, "immune_compliance": 1}]
* __"people_per_day"__ - list of options for vaccinated per day  [X,Y,Z],
* __"latency days_n"__ - scalar value for epidemic latency period,
* __"linked_immune_age"__: subfield for all paired immune source [{"immune_source": "BY_NEIGHBORHOOD", "min_age": 18}]
* __"initial_num_infected"__ list of options [700],
* __"linked_city_scale"__ [
       {"city_name": "Holon", "scale": 1},
       {"city_name": "Bene Beraq", "scale": 0.8}
  ],
* __"compliance"__ [0.7],
* __"order"__: ["ASCENDING"],
* __"ci_delay"__: [4],
* __"hi_delay"__: [4],
* __"minimum_infectiousness_age"__: [0],
* __"symptomatic_probs_scale"__: [1],
* __"scenarios"__: For "Basic" experiment type use all except Empty. For RL experiment type - starts only from Empty 
    {
    "Empty_scenario": "Empty_scenario",
    "reality1" : "scenario_reality1",
    "paper_1" : "paper_1"
   },
* __"train_epochs"__: Number of agent training epochs , 
* __"stop_condition"__: "rt_flag" - do we use rt value as stop condition, "rt" - value of rt, in which we're stopping, else we use all other values, like % of infected_population
  {
        "rt_flag": 1,
        "rt": 0.75,
        "infected_population": 0.01,
        "vaccinated_population": 0.01
      },
  * __"wandb"__: {"usage_flag":  1,
                "experiment_name": "coderona_dqn",
                "team_name": "__",
                "api_key": "__",
                "log_vars": [
                  "infected_population",
                  "vaccinated_population",
                  "infected_population",
                  "vaccinated_population",
                  "rt",
                  "r0",
                  "reward",
                  "economic_index",
                  "action",
                  "epsilon"
                ]
      },
* __"dqn_agent"__: Upper bound of actions number is 10. We can play with number of actions {
      "actions": {
        "start_key": 0,
        "end_key": 4
      }
