from sanic import Sanic, response
import aiofiles
import jinja2

import os
import string
import inspect
import time
import json
import sys
import enum
import shutil
import pickle
from datetime import datetime
from collections import OrderedDict
import multiprocessing as mp

# Hack to import sibling module
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from simulation.interventions import *

from world.city_data import get_city_list_from_dem_xls
from world.population_generation import PopulationLoader
from simulation.simulation import Simulation
from simulation.params import Params
from seir import DiseaseState
from logs import make_age_and_state_datas_to_plot

def intervention_title(inter):
    if type(inter) != str:
        inter = inter.__name__
    new = ''
    for c in inter.replace('Intervention', ''):
        if c in string.ascii_uppercase and new:
            new += ' '
        new += c
    return new

def intervention_params(inter):
    sig = inspect.signature(inter.__init__)
    params = list(sig.parameters.values())[1:]
    ps = {p.name: p.annotation for p in params}
    if inspect._empty in ps.values():
        raise ValueError('Unable to find type hint for ' + inter.__name__)
    return ps

INTERVENTIONS = [
        ElderlyQuarantineIntervention,
        SymptomaticIsolationIntervention, SocialDistancingIntervention
]

INTERVENTION_BY_NAME = {i.__name__: i for i in INTERVENTIONS}

POPULATIONS = get_city_list_from_dem_xls()
POPULATION_NAMES = ['All'] + sorted([city.english_name for city in POPULATIONS])

DEFAULT_DISEASE_STATES_GROUPS = OrderedDict([
        ("Susceptible", [DiseaseState.SUSCEPTIBLE]),
        ("Infected", [DiseaseState.SYMPTOMATICINFECTIOUS, DiseaseState.ASYMPTOMATICINFECTIOUS,
                DiseaseState.INCUBATINGPOSTLATENT, DiseaseState.LATENT])
])
DEFAULT_AGE_GROUPS = [(0, 19), (20, 39), (40, 59), (60, 79), (80, 99)]

# TODO: These should be GUI parameters really
Params.load_from(os.path.join(PARENT_DIR, 'simulation', 'params.json'))
INITIAL_INFECTED_NUM = 20

DAYS = 250
TASKS_DIR = 'tasks'

class TaskStatus(enum.IntEnum):
    CREATED = 1
    STARTED = 2
    FINISHED = 3

# This is used in create_and_run_simulation, which is a sync function, so no aiofiles
def change_task_status(name, status, also_add={}):

    status_file = os.path.join(TASKS_DIR, name, 'status')
    with open(status_file, 'r') as f:
        j = json.load(f)

    j['status'] = status
    j.update(also_add)

    with open(status_file, 'w') as f:
        json.dump(j, f)

def create_and_run_simulation(name, city_name, scale, city_list, interventions):
    change_task_status(name, TaskStatus.STARTED,
            {'started_at': int(time.time())}
    )

    Params.load_from(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'simulation', 'params.json'))
    population_loader = PopulationLoader()
    world = population_loader.get_world(city_name=city_name, scale=scale)
    sim = Simulation(world, 0, interventions,
                     False, outdir=os.path.join(TASKS_DIR, name))
    sim.infect_random_set(INITIAL_INFECTED_NUM, "N/A", None)
    sim.run_simulation(DAYS, name, datas_to_plot=[])

    change_task_status(name, TaskStatus.FINISHED,
            {'finished_at': int(time.time())}
    )

# On windows, multiprocessing import the main module
# So we need to mask all this away
if __name__ == '__main__': 
    # It sucks that the loader is not async, but it's not improtant
    my_dir = os.path.dirname(os.path.abspath(__file__))
    templates_path = os.path.join(my_dir, 'templates')
    loader = jinja2.FileSystemLoader(templates_path)
    jinja = jinja2.Environment(loader=loader, enable_async=True)

    jinja.globals['intervention_title'] = intervention_title
    jinja.globals['intervention_params'] = intervention_params
    jinja.globals['int'] = int
    jinja.globals['float'] = float
    jinja.globals['interventions'] = INTERVENTIONS
    jinja.globals['population_names'] = POPULATION_NAMES
    jinja.globals['TaskStatus'] = TaskStatus
    jinja.globals['datetime'] = datetime
    jinja.globals['DEFAULT_AGE_GROUPS'] = DEFAULT_AGE_GROUPS
    jinja.globals['DEFAULT_DISEASE_STATES_GROUPS'] = DEFAULT_DISEASE_STATES_GROUPS
    jinja.globals['DiseaseState'] = DiseaseState

    app = Sanic(__name__)
    app.static('/static', os.path.join(os.path.dirname(__file__), 'static'))
    app.static('/tasks', TASKS_DIR)

    TASK_POOL = mp.Pool(max(mp.cpu_count() - 2, 1))

    try:
        os.mkdir(TASKS_DIR)
    except FileExistsError:
        pass

    @app.listener('after_server_stop')
    def terminate_pool(app, loop):
        TASK_POOL.terminate()

    async def render_template(template, **kwargs):
        html = await jinja.get_template(template).render_async(**kwargs)
        return response.html(html)

    @app.route('/')
    async def index(request):
        return response.redirect(app.url_for('add_task_page'))

    @app.route('/add_task_page')
    async def add_task_page(request):
        return await render_template('add_task_page.html')

    @app.route('/tasks_page')
    async def tasks_page(request):
        tasks = []
        for filename in os.listdir(TASKS_DIR):
            fullname = os.path.join(TASKS_DIR, filename, 'status')
            async with aiofiles.open(fullname, 'r') as f:
                tasks.append(json.loads(await f.read()))
        tasks.sort(key=lambda t: t['created_at'], reverse=True)
        return await render_template('tasks_page.html', tasks=tasks)

    @app.route('/add_task', methods=['POST'])
    async def add_task(request):
        req = request.json

        name = req['name']

        dirname = os.path.join(TASKS_DIR, name)
        if os.path.isdir(dirname):
            return response.text('A task with that name already exists',
                    status=404);
        os.mkdir(dirname)

        status_name = os.path.join(dirname, 'status')

        interventions = [
                INTERVENTION_BY_NAME[inter['name']](**inter['params'])
                for inter in req['interventions']
        ]

        city_name = req['population'].lower()

        async with aiofiles.open(status_name, 'w') as f:
            await f.write(json.dumps({
                'status': TaskStatus.CREATED,
                'created_at': int(time.time()),
                'task_data': req
            }))

        scale = 1.0
        if city_name.lower() == 'all':
            scale = 0.1
        TASK_POOL.apply_async(create_and_run_simulation,
                args=(name, city_name, scale, POPULATIONS, interventions))
       
        return response.text('OK')

    @app.route('/delete_task/<name>')
    async def delete_task(request, name):
        try:
            shutil.rmtree(os.path.join(TASKS_DIR, name))
        except FileNotFoundError:
            pass
        return response.text('OK')

    @app.route('/plot_graph/<task_name>', methods=['POST'])
    async def plot_graph(request, task_name):
        req = request.json
        age_groups = tuple([tuple(g) for g in req['age_groups']])
        disease_states_groups = [
                (key, [DiseaseState(int(v)) for v in value])
                for key, value in req['disease_states_groups'].items()
        ]

        task_dir = os.path.join(TASKS_DIR, task_name)
        async with aiofiles.open(os.path.join(task_dir, 'statistics.pkl'), 'rb') as f:
            stats_data = await f.read()
        stats = pickle.loads(stats_data)
        data = make_age_and_state_datas_to_plot(age_groups, disease_states_groups)
        stats.plot_daily_sum('tmp', data)
        return response.text('OK')

    @app.route('/save_graph/<task_name>/<graph_name>')
    async def save_graph(request, task_name, graph_name):
        task_dir = os.path.join(TASKS_DIR, task_name)
        src = os.path.join(task_dir, 'tmp.svg')
        dst = os.path.join(task_dir, graph_name + '.svg')
        shutil.copy(src, dst)
        return response.text('OK')

    @app.route('/graphs/<task_name>')
    async def graphs(request, task_name):
        files = []
        for filename in os.listdir(os.path.join(TASKS_DIR, task_name)):
            if filename.endswith('.svg') and filename != 'tmp.svg':
                files.append(filename.replace('.svg', ''))

        return response.json(files)

    app.run()
