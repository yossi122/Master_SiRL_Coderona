import wandb
import numpy as np
from datetime import timedelta, date, datetime
import time
import random

wandb_ = {"experiment_name": "coderona_dqn",
          "team_name": "karazin-gushchin",
          "api_key": "9ea2363660dd70e3e5fa2aff335b14f71c34bfd1"
          }
'''

def wandb_routine(job_id, episodes_data, project_name="coderona_dqn", team_name='mine'):
    # Initialize a new run for each job
    run_name = f"job_{job_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    wandb.init(
        project=wandb_['experiment_name'],
        entity=wandb_['team_name'],
        name=run_name,
        reinit=True
    )
    for episode_id, episode_data in enumerate(episodes_data):
        for step, value in episode_data:
            wandb.log({f"Episode {episode_id}": value}, step=step)
    wandb.finish()


# Example usage
jobs_data = [
    [(np.arange(150), np.sin(np.arange(150) / 10)) for _ in range(5)],  # Job 1 with 5 episodes
    [(np.arange(150), np.cos(np.arange(150) / 10)) for _ in range(5)]  # Job 2 with 5 episodes
]

# Login once
wandb.login(key=wandb_['api_key'])

# Process each job
for job_id, episodes_data in enumerate(jobs_data, start=1):
    # Transform episodes_data into the format [(step, value), (step, value), ...]
    formatted_episodes_data = [[(step, value) for step, value in zip(episode[0], episode[1])] for episode in
                               episodes_data]
    wandb_routine(job_id, formatted_episodes_data)
'''


def wandb_routine(job_id, project_name="coderona", team_name='mine'):
    run_name = f"job_{job_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    wandb.init(
        project=wandb_['experiment_name'],
        entity=wandb_['team_name'],
        name=run_name,
        reinit=True
    )

    data = {f"Episode {episode_id}": [] for episode_id in range(5)}

    # Simulate heavy calculation for 5 episodes with 150 steps each

    all_data = []
    for episode_id in range(5):
        for step in range(15):
            # Generate some data (replace this with your actual data generation)
            value = np.sin((step * random.random()) / 10 + episode_id)

            # Append data to the all_data list
            all_data.append([episode_id, step, value])

    # Create a wandb Table with episode identifiers
    table = wandb.Table(data=all_data, columns=["episode_id", "step", "height"])

    # Create a line series plot from the table
    line_plot = wandb.plot_table(
        vega_spec_name="wandb/line/v0",
        data_table=table,
        fields={"x": "step", "y": "height", "series": "episode_id"},
        string_fields={"title": "Parameter A over Episodes"}
    )
    wandb.log({"Parameter A": line_plot})
    wandb.finish()


# Example usage
jobs_data = [1, 2]  # Representing job IDs

# Login once
wandb.login(key=wandb_['api_key'])

# Process each job
for job_id in jobs_data:
    pass
    # wandb_routine(job_id)
import math
import wandb

N_STEPS = 100

# wandb.init(project="someProject", name="testMultipleLines")
run_name = f"job_{1}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
wandb.init(
    project=wandb_['experiment_name'],
    entity=wandb_['team_name'],
    name=run_name,
    reinit=True
)

table = wandb.Table(columns=["metric", "value", "step"])
'''
log={}
for ep in range(5): # epochs
    other_metrics = {}
    for step in range(15+random.randint(-2,2)): # model
        log[f'other_metrics/metric_{step}'] = math.sin(step * math.pi * (ep /(step+1)))
        table.add_data(f'other_metrics/metric_{step}',
                       log[f'other_metrics/metric_{step}'],
                       wandb.run.step)
    wandb.log(log)

wandb.log({"multiline": wandb.plot_table(
    "wandb/line/v0", table, {"x": "step",
                             "y": "value",
                             "groupKeys": "metric"},
    {"title": "Multiline Plot"})
})
'''

def foo(id):
    run_name = f"job_{id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    wandb.init(
        project=wandb_['experiment_name'],
        entity=wandb_['team_name'],
        name=run_name,
        reinit=True
    )

    table = wandb.Table(columns=["metric", "value", "step"])
    for epoch in range(150+random.randint(-2,2)):
        log = {}
        #log['main/metric'] = epoch / N_STEPS  # some main metric

        # some other metrics I want to have all on 1 plot
        other_metrics = {}
        for j in range(3):
            log[f'other_metrics/metric_{j}'] = random.randint(0,4)
            table.add_data(f'other_metrics/metric_{j}',
                           log[f'other_metrics/metric_{j}'],
                           wandb.run.step)
        wandb.log(log)

    wandb.log({"multiline": wandb.plot_table(
        "wandb/line/v0", table, {"x": "step",
                                 "y": "value",
                                 "groupKeys": "metric"},
        {"title": "Multiline Plot"})
    })
    '''
    line_plot = wandb.plot.line_series(
        xs=[table.get_column("step") for _ in range(3)],
        ys=[table.get_column("value") for _ in range(3)],
        keys=table.get_column("metric"),
        title="Multiline Plot",
        xname="Step",
        yname="Value"
    )
    wandb.log({"multiline": line_plot})
    '''
    wandb.finish()
'''
x_values = np.arange(1, 100)
y_values = np.random.randn(100)
run_name = f"job_{1}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
wandb.init(
    project=wandb_['experiment_name'],
    entity=wandb_['team_name'],
    name=run_name,
    reinit=True
)
'''
'''
import wandb
import random
import math

offset = random.random()

# At each time step in the model training loop
for run_step in range(20):

    # Log basic experiment metrics, which show up as standard line plots in the UI
    wandb.log({
        "acc": math.log(1 + random.random() + run_step) + offset,
        "val_acc": math.log(1 + random.random() + run_step) + offset * random.random(),
    }, commit=False)

    # Set up data to log in custom charts
    data = []
    for i in range(100):
        data.append([i, random.random() + math.log(1 + i) + offset + random.random()])

    # Create a table with the columns to plot
    table = wandb.Table(data=data, columns=["step", "height"])

    # Use the table to populate various custom charts
    line_plot = wandb.plot.line(table, x='step', y='height', title='Line Plot')
    histogram = wandb.plot.histogram(table, value='height', title='Histogram')
    scatter = wandb.plot.scatter(table, x='step', y='height', title='Scatter Plot')

    # Log custom tables, which will show up in customizable charts in the UI
    wandb.log({'line_1': line_plot,
               'histogram_1': histogram,
               'scatter_1': scatter})

# Finally, end the run. We only need this ine in Jupyter notebooks.
#run.finish()
'''
def doo(id):
    run_name = f"job_{id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    wandb.init(
        project=wandb_['experiment_name'],
        entity=wandb_['team_name'],
        name=run_name,
        reinit=True
    )

    table = wandb.Table(columns=["episode", "metric", "value", "step"])
    N_EPISODES = 5
    N_STEPS = 150

    for episode in range(N_EPISODES):
        num_steps = N_STEPS + random.randint(-2, 2)
        for step in range(num_steps):
            log = {}
            # some other metrics I want to have all on 1 plot
            other_metrics = {}
            for j in range(3):
                metric_value = random.random() * math.sin(j * math.pi * (step / N_STEPS))
                log[f'other_metrics/metric_{j}'] = metric_value
                table.add_data(episode, f'other_metrics/metric_{j}', metric_value, step)
            wandb.log(log)

    wandb.log({"multiline": wandb.plot_table(
        "wandb/line/v0", table, {"x": "step",
                                 "y": "value",
                                 "groupKeys": ["episode", "metric"]},
        {"title": "Multiline Plot"})
    })

    wandb.finish()

for run in range(1):
    doo(run)
