# Reinforcement Learning Coderona simulation
Optimal policy fighting covid-19 one model at a time

### config.json
#### Pandemic parameters 
* Tune the : vaccinate_per_day_persons , vaccinate_per_day_household , initial_num_infected , linked_immune_age : min_age , max_age , compliance , "order" : [ASCENDING , DESCENDING , NONE]

### Simulation parameters
* Tune the  : latency days_n , intervention_duration , days_bound

## Model Saving Process (.h5 Format)
During the training process, after the model has been compiled and trained, it is important to save the model's architecture, weights, and training configuration to allow for reuse or further training later. In this project, the model is saved in HDF5 (.h5) format, which is widely supported by machine learning frameworks such as TensorFlow and Keras.
The saving process typically occurs after the model training using the following approach:
1. Model Training: The model undergoes a training phase where it learns from the provided dataset using a specified optimizer, loss function, and metrics.
2. Checkpointing: The model is saved during or after training. This allows us to store both the model's architecture and learned parameters (weights).
3. Saving the Model:
   The model architecture.
   The learned weights (parameters).
   The optimizer's state (if any).
   Training configuration.
4. Restoring the Model: The saved model can be easily loaded later for inference or further training.

This approach ensures that the trained model can be used across different environments or resumed from the last saved state for continued training or evaluation
