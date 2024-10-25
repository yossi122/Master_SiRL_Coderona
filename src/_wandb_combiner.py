import wandb
import json
import os
import pandas as pd

def get_config_param(config_name="people_per_day"):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = f'{parent_dir}/src/config.json'
    with open(config_path) as json_data_file:
        ConfigData = json.load(json_data_file)

    return ConfigData[config_name]

wandb_ = get_config_param("wandb")
print(wandb_.keys())
api = wandb.Api()

entity = wandb_['team_name']
project = wandb_['experiment_name']

# Get all runs in the project
runs = api.runs(f'{entity}/{project}')

# Extract run IDs and other information into a list
run_ids = [run.id for run in runs][11:13] # get latest N runs


# If you want more details, you can retrieve a list of dictionaries
run_info = [{
    'id': run.id,
    'name': run.name,
    'created_at': run.created_at,
    'state': run.state,
    'summary': run.summary._json_dict
} for run in runs]

# Initialize API


# Load runs by their ID
run_log = []
df_log = []
all_records = []
batch_size = 10
current_step = 0

for idx, i in enumerate(run_ids):
    run_log.append(api.run(f'{entity}/{project}/{i}'))
    print('+>',run_log[idx].name,'<+')
    print('step: ', run_log[idx].summary)


    x = api.run(f'{entity}/{project}/{i}')

    while True:
        # Retrieve the next batch of records
        history_batch = run_log[idx].scan_history(page_size=batch_size, min_step=current_step)

        # Convert batch to list and check if it's empty
        batch_list = list(history_batch)
        if not batch_list:
            break  # Exit loop if no more records

        # Append the batch to all_records
        all_records.extend(batch_list)

        # Update the current step to fetch the next batch
        current_step = max(record['_step'] for record in batch_list) + 1
    '''
    y = run_log[idx].scan_history(keys=['epsilon', '_step', 'economic_index',
                                        'reward', 'memory_access', 'loss_'],
                                  page_size=1000, min_step=None, max_step=None
                                  )
    '''
    #df = pd.DataFrame(list(y))
    df = pd.DataFrame(all_records)

    print(df.shape)
    print('='*10)
    df_log.append(df)



# Combine the dataframes
combined_df = pd.DataFrame()
for i, df in enumerate(df_log):
    if i > 0:
        # Adjust epochs based on the last epoch of the previous run
        last_epoch = combined_df['_step'].max()
        df['_step'] += last_epoch + 1  # Ensure continuity
    combined_df = pd.concat([combined_df, df], ignore_index=True)

combined_df = combined_df.sort_values(by='_step')
combined_df = combined_df.rename(columns={'_step': 'day', 'loss_': 'loss'})

# Plotting the combined data
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(combined_df['day'], combined_df['loss'], label='Loss')
#plt.plot(combined_df['day'], combined_df['epsilon'], label='Epsilon')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Combined Metrics Across Multiple Runs')
plt.legend()
plt.show()