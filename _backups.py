
class ReplayBuffer_old:
    MEMORY_SIZE = 700000000

    def __init__(self):
        self.memory = deque(maxlen=self.MEMORY_SIZE)  # STORAGE
        self.reward_memory_total = deque(maxlen=self.MEMORY_SIZE)
        self.terminal_memory_total = deque(maxlen=self.MEMORY_SIZE)

        self.day_range = get_config_param("latency days_n")  # initial period for get historical dynamics
        self.intervention_dur = get_config_param("intervention_duration")


class AgentLSTM_old(ReplayBuffer_old):
    def __init__(self, runner, economic, job, model=None):
        super().__init__()
        self.name = runner.city_name
        self.model, self.target = model, None
        self.discount_factor = 0.95
        self.lr = get_config_param("dqn_agent")['train_params']["learning_rate"]
        self.q_equation = get_config_param('dqn_agent')["q_equation"]  # original / classical
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0)
        self.loss_fn = losses.MeanSquaredError()
        self.time_steps = self.day_range  #self.day_range - 1
        self.n_outputs = None
        self.input_shape = None

        self.cum_reward_graph = []
        self.model_history = []
        self.mean_action = []
        self.reward = 0
        self.mean_reward = 0
        self.epsilon = None
        self.seen_states = {}

        self.update_counter = 0
        self.target_update_frequency =100

        rew_type = get_config_param("dqn_agent")['train_params']["reward_type"]
        if rew_type['parametrized'] and not rew_type['penalty']:
            self.rew_type = 'parametrized'
        elif not rew_type['parametrized'] and rew_type['penalty']:
            self.rew_type = 'penalty'

        hyper = get_config_param(config_name="dqn_agent")['train_params']["hyper_params"]
        if hyper["constant_flag"]:
            self.lambda_ = hyper['lambda']
            self.mu_ = hyper['mu']
            self.ro_ = hyper['ro']
            self.pi_ = hyper['pi']
        else:
            self.lambda_ = 1  # economic aspect first
            self.mu_ = 1  # infected count aspect first
            self.ro_ = 1  # deaths aspect first
            self.pi_ = 1  #
        # self.set_model()

        # full dict of possible actions:
        self.update_act_dict(runner, economic, job)
        get_log_history()

    def update_act_dict(self, runner, economic, job):
        compliance = runner.compliance
        ci_delay = runner.ci_delay
        hi_delay = runner.hi_delay
        #INITIAL_DATE = date(year=2020, month=2, day=27)

        x = economic.age_groups
        duration = timedelta(get_config_param("intervention_duration"))
        houses_per_day = get_config_param("vaccinate_per_day_household")
        vac_per_day = get_config_param("vaccinate_per_day_persons")

        house_interventions = []
        global_interventions = []

        if runner.cur_day == None:
            start_date = job.initial_date + timedelta(days=0)
        else:
            start_date = job.initial_date + timedelta(days=runner.cur_day)

        if len(x) > 0:
            for i in x:
                house_interventions.append(ImmuneByHouseholdIntervention(start_date=start_date,
                                                                         duration=duration,
                                                                         compliance=compliance,
                                                                         houses_per_day=houses_per_day,
                                                                         min_age=i[0],
                                                                         max_age=i[1],
                                                                         immune_type="vaccine"
                                                                         ))
                global_interventions.append(ImmuneGeneralPopulationIntervention(compliance=compliance,
                                                                                start_date=start_date,
                                                                                duration=duration,
                                                                                people_per_day=vac_per_day,
                                                                                min_age=i[0],
                                                                                max_age=i[1],
                                                                                immune_type="vaccine"
                                                                                ))
        else:
            house_interventions.append(ImmuneByHouseholdIntervention(start_date=start_date,
                                                                     duration=duration,
                                                                     compliance=compliance,
                                                                     houses_per_day=houses_per_day,
                                                                     min_age=0,
                                                                     max_age=99,
                                                                     immune_type="vaccine"
                                                                     ))
            global_interventions.append(ImmuneGeneralPopulationIntervention(compliance=compliance,
                                                                            start_date=start_date,
                                                                            duration=duration,
                                                                            people_per_day=vac_per_day,
                                                                            min_age=0,
                                                                            max_age=99,
                                                                            immune_type="vaccine"
                                                                            ))

        ''' 6
                                     
        '''
        self.act_dict = {0: scenarios.Empty_scenario(),
                         1: house_interventions,
                         2: global_interventions,
                         3: [SymptomaticIsolationIntervention(start_date=start_date,
                                                              duration=duration,
                                                              compliance=compliance,
                                                              delay=ci_delay
                                                              )],
                         4: [SocialDistancingIntervention(start_date=start_date,
                                                          duration=duration,
                                                          compliance=compliance,
                                                          age_range=(0, 99)
                                                          )],
                         5: [SchoolClosureIntervention(start_date=start_date,
                                                       duration=duration,
                                                       compliance=compliance,
                                                       proportion_of_envs=1.0,
                                                       city_name='all',
                                                       age_segment=(3, 22)
                                                       )],
                         6: [WorkplaceClosureIntervention(start_date=start_date,
                                                          duration=duration,
                                                          compliance=compliance
                                                          )],
                         7: [ElderlyQuarantineIntervention(start_date=start_date,
                                                           duration=duration,
                                                           compliance=compliance,
                                                           min_age=70
                                                           )],
                         8: [HouseholdIsolationIntervention(start_date=start_date,
                                                            duration=duration,
                                                            compliance=compliance,
                                                            delay_on_enter=hi_delay
                                                            )],
                         9: [CityCurfewIntervention('jerusalem',
                                                    start_date=start_date,
                                                    duration=duration,  # daysdelta(120),
                                                    compliance=compliance)],
                         10: [LockdownIntervention(start_date=start_date + timedelta(0.0),
                                                   duration=duration,  # daysdelta(4 * 7),
                                                   compliance=compliance,
                                                   city_name='all'
                                                   )]
                         }

    def set_model(self, economic):
        '''
        Bi-LSTM model for train on historical data
        :return:
        '''
        # TODO: as option change condition (alpha, beta +)
        stats = [attr for attr in dir(economic) if
                 'population' in attr and not callable(getattr(economic, attr)) and not attr.startswith("__")
                 and not 'total' in attr]
        stats.remove('exposed_population')

        if get_config_param("age_stats"):
            new_stats = []
            for a in get_config_param("age_groups"):
                for s in stats:
                    new_stats.append(f'{s}_{a}')
            stats = new_stats
            state_size = len(stats)
        else:
            state_size = sum(
                len(getattr(economic, i).keys()) if isinstance(getattr(economic, i), dict) else 1 for i in stats)

        start_key = get_config_param("dqn_agent")['train_params']["actions"]["start_key"]
        end_key = get_config_param("dqn_agent")['train_params']["actions"]["end_key"]
        sub_dict = {key: self.act_dict[key] for key in range(start_key, end_key + 1)}
        self.input_shape = state_size + 1  # SEIRDE + Action
        self.n_outputs = len(sub_dict.keys())
        self.seen_states = {i: 0 for i in range(self.n_outputs)}

        model = keras.Sequential([
            keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True),
                                       input_shape=(self.time_steps, self.input_shape)),
            keras.layers.Dropout(0.2),
            keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
            keras.layers.Dropout(0.2),
            keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.n_outputs, activation='softmax')
        ])
        # TODO: fix fo loss increase?
        ''' 
        self.model = model
        self.model.compile(loss=self.loss_fn, optimizer=self.optimizer)
        self.target = model
        '''
        model.compile(loss=self.loss_fn, optimizer=self.optimizer)
        self.model = model
        self.target = keras.models.clone_model(model)
        self.target.set_weights(model.get_weights())

        get_log_history()

    def load_model(self, file_path):
        if os.path.exists(file_path):
            self.model = keras.models.load_model(file_path)
            self.target = self.model
            print(f"Model loaded from {file_path}")
        else:
            raise FileNotFoundError(f"The file {file_path} does not exist.")

    def get_reward(self, env, sim):
        '''
        maximize the following multi-objective reward function for
        state θ := [Et, S, I, H, D, R, V1].

        :return:
        '''
        if isinstance(env.infected_population, dict):
            infect = sum(env.infected_population.values())
            dead = sum(env.deceased_population.values())
            sus = sum(env.susceptible_population.values())
            vac = sum(env.vaccinated_population.values())
            hos = sum(env.hospitalized_population.values())
            rec = sum(env.recovered_population.values())
            exp = sum(env.exposed_population.values())
            tot = sum(env.total_population.values())
        else:
            infect = env.infected_population
            dead = env.deceased_population
            sus = env.susceptible_population
            vac = env.vaccinated_population
            hos = env.hospitalized_population
            rec = env.recovered_population
            exp = env.exposed_population
            tot = env.total_population
            # x = sum(env.infected_population.values())

        if self.rew_type == 'parametrized':
            # PREVIOUS VERSION OF REWARD EQUATION
            self.reward = (self.lambda_ * env.economic_index -
                           self.mu_ * infect -
                           self.ro_ * dead +
                           self.pi_ * (sus + vac))
        elif self.rew_type == 'penalty':
            # -- PENALTY PART
            rt, r0, r_smoothed, r_estimated = sim.get_rt()

            print(f'\t >> RT = {rt}')
            if 0 < rt < 1.25 and env.economic_index < 0:
                print('#### ODD CASE:', rt, env.economic_index)

            if rt > 2.0:  #or rt == 0:  #!@ TODO: rt==0 is equal to rt is high
                self.reward = get_config_param("dqn_agent")["train_params"]["penalty_reward_params"][
                                  "rt>2.0"] * env.economic_index  # Severe penalty for high rates
            elif 1.5 < rt <= 2.0:
                self.reward = get_config_param("dqn_agent")["train_params"]["penalty_reward_params"][
                                  "1.5 < rt <= 2.0"] * env.economic_index  # Mild penalty for moderate rates
            elif 1.25 <= rt <= 1.5:
                self.reward = get_config_param("dqn_agent")["train_params"]["penalty_reward_params"][
                                  "1.25 <= rt <= 1.5"] * env.economic_index  # Reward for relatively stable condition
            elif 1.0 < rt < 1.25:
                self.reward = get_config_param("dqn_agent")["train_params"]["penalty_reward_params"][
                                  "rt < 1.25"] * env.economic_index  # High reward for optimal condition
            elif 0 < rt < 1.0:
                self.reward = get_config_param("dqn_agent")["train_params"]["penalty_reward_params"][
                                  "rt < 1.0"] * env.economic_index

            print('%%%%', self.reward)
            infection_ratio = infect / tot
            print(f'\t >> infection_ratio = {infection_ratio}')
            infect_rate = get_config_param("dqn_agent")["train_params"]["penalty_reward_params"]["infect_rate"]
            if infection_ratio > 0.01:
                self.reward -= get_config_param("dqn_agent")["train_params"]["penalty_reward_params"][
                                   "ratio > infect_rate"] * (infection_ratio / infect_rate)
            else:
                self.reward += get_config_param("dqn_agent")["train_params"]["penalty_reward_params"][
                                   "ratio < infect_rate"] * (infect_rate - infection_ratio)
            print('%%%%', self.reward)
            if self.memory[-1][-1] < 0:  # if action < 0, or not action
                self.reward = -1000000
            print(f'\t >> REWARD = {self.reward}')
        else:
            raise f"You must specify ony one reward type to be active. Current settings is {rew_type}"
        get_log_history()

    def epsilon_greedy_policy(self, state, num_mini_batches=10, mini_batch_size=11):
        """
        Epsilon-greedy policy with random mini-batch selection.
        """
        if np.random.rand() < self.epsilon:
            min_seen = min(self.seen_states.values())
            min_seen_keys = [key for key, value in self.seen_states.items() if value == min_seen]
            x = random.choice(min_seen_keys)
            print('==========> RAND EPS:', min_seen_keys, x)
            return x, "EPS"
        else:
            state_array = state
            print('^'*50)
            print(state_array)
            print('^' * 50)
            Q_values = self.model.predict(state_array)[0]
            print('^' * 50)
            print(Q_values)
            print('^' * 50)

            # Ensure the number of unique indices does not exceed the length of Q_values
            total_indices_needed = num_mini_batches * mini_batch_size
            available_indices = len(Q_values)
            if total_indices_needed > available_indices:
                # Adjust mini-batch size and number of mini-batches if necessary
                mini_batch_size = max(1, available_indices // num_mini_batches)
                num_mini_batches = available_indices // mini_batch_size
                total_indices_needed = num_mini_batches * mini_batch_size

            if total_indices_needed <= available_indices:
                unique_indices = np.random.choice(available_indices, total_indices_needed, replace=False)
            else:
                unique_indices = np.random.choice(available_indices, available_indices, replace=False)
                mini_batch_size = available_indices
                num_mini_batches = 1

            mini_batch_indices = unique_indices.reshape(num_mini_batches, mini_batch_size)
            print('^' * 50)
            print(mini_batch_indices)
            print('^' * 50)
            mini_batch_max_values = []

            top_k = 3
            for mini_batch in mini_batch_indices:
                mini_batch_Q_values = Q_values[mini_batch]
                # Get top k actions instead of just one max value
                top_k_indices = np.argsort(mini_batch_Q_values, axis=1)[:, -top_k:]
                top_k_values = np.take_along_axis(mini_batch_Q_values, top_k_indices, axis=1)
                best_actions = top_k_indices.flatten()  # Flatten to get a single list of actions
                mini_batch_max_values.append((np.max(top_k_values), best_actions))

            log.info(mini_batch_max_values)
            # Select the maximum action from the mini-batches
            #selected_max_value, selected_best_actions = max(mini_batch_max_values, key=lambda x: x[0])
            '''
            all_actions = [actions for _, actions in mini_batch_max_values]
            flattened_actions = [action for sublist in all_actions for action in sublist]
            unique_actions = list(set(flattened_actions))
            x = np.random.choice(unique_actions)
            print(f'==========> MODEL EPS: -> {selected_max_value} -> {selected_best_actions} -> {x}')
            '''
            # Collect all unique actions
            unique_actions = set()
            for _, actions in mini_batch_max_values:
                unique_actions.update(actions)
            print('&&&', unique_actions)
            # Find the maximum value and corresponding actions
            max_value = max(mini_batch_max_values, key=lambda x: x[0])[0]
            max_actions = [actions for value, actions in mini_batch_max_values if value == max_value]

            # Flatten the list of actions and remove duplicates
            max_actions_flat = set([action for sublist in max_actions for action in sublist])

            # Select a random action from the unique actions
            x = np.random.choice(list(max_actions_flat))

            print(f'==========> MODEL EPS: -> {max_value} -> {list(unique_actions)} -> {x}')
            assert max(unique_actions) <= self.n_outputs - 1
            self.seen_states[x] += 1
            if x >= self.n_outputs:
                raise ValueError(f"Predicted action {x} is out of bounds!")

            return x, "MEMORY"

    def epsilon_greedy_policy_old(self, state):
        # small probability of exploring - epsilon is probability of exploration, np.random.rand() gives # between 0,1
        if np.random.rand() < self.epsilon:
            min_seen = min(self.seen_states.values())
            min_seen_keys = [key for key, value in self.seen_states.items() if value == min_seen]

            #x = random.randint(0, self.n_outputs - 1)
            x = random.choice(min_seen_keys)
            print('==========> RAND EPS:', min_seen_keys, x)
            #x = 2
            return x, "EPS"
        # predict next action using model
        else:
            state_array = state
            Q_values = self.model.predict(state_array)[0]
            # print(f'>>>>> {state_array}\t>>>>{Q_values}')

            max_value = np.max(Q_values)
            #prediction = np.argmax(Q_values[0][0])
            best_actions = np.where(Q_values == max_value)[1]
            x = np.random.choice(best_actions)
            print(f'==========> MODEL EPS: -> {max_value} -> {best_actions} -> {x}')

            assert max(best_actions) <= self.n_outputs - 1
            self.seen_states[x] += 1
            if x >= self.n_outputs:
                raise ValueError(f"Predicted action {x} is out of bounds!")
            #x =2
            return x, "MEMORY"
            #print('prediction', prediction)
            #return prediction

    def update_model(self, current_states, next_states):
        if self.q_equation == 'original':
            self.update_model_original(current_states, next_states)
        elif self.q_equation == 'classical':
            self.update_model_classical(current_states, next_states)

    def update_model_original(self, current_states, next_states):
        rewards = []
        terminals = []

        for i in range(1, self.day_range + 1):
            rewards.append(self.reward_memory_total[-i])  # last N days
            terminals.append(self.terminal_memory_total[-i])

        next_Q_values = (self.model.predict(next_states))[0]  # predict next
        all_Q_values = (self.model.predict(current_states))[0]  # predict i-1
        prediction = (self.target.predict(next_states))[0]  # use target to find next
        # ALL ARE (1,30,4)

        ACTION_HISTORY = []

        for i in range(self.day_range):# - 1):
            ACTION_HISTORY.append(current_states[0][i][-1])  # retrieve actions

        best_next_actions = []
        for i in range(len(next_Q_values)):
            best_next_actions.append(np.argmax(next_Q_values[i]))

        next_mask = tf.one_hot(best_next_actions, self.n_outputs).numpy()

        next_best_Q_values = (prediction * next_mask).sum(axis=1)

        target_Q_values = []

        for i in range(self.day_range):# - 1):
            target_Q_values.append(rewards[i] + (1 - terminals[i]) * self.discount_factor * next_best_Q_values[i])
            action = ACTION_HISTORY[i]  # action taken at every step

            if int(action) < len(all_Q_values[i]):
                all_Q_values[i][int(action)] = target_Q_values[i]  # this is like "informal" mask

        all_Q_values = [all_Q_values.tolist()]
        all_Q_values = np.asarray(all_Q_values)

        history = self.model.fit(current_states, all_Q_values, verbose=0)  # calculate loss, update weights
        self.model_history.append(history.history['loss'])
        get_log_history()

    def update_model_classical(self, prev_states, cur_states):
        rewards = []
        terminals = []

        print('#'*50)
        print(self.reward_memory_total)
        print(self.terminal_memory_total)
        print('#' * 50)
        for i in range(1, self.day_range + 1):
            rewards.append(self.reward_memory_total[-i])  # last N days
            terminals.append(self.terminal_memory_total[-i])

        print('#' * 50)
        print(prev_states)
        print()
        print(cur_states)
        print('#' * 50)
        next_Q_values = self.model.predict(cur_states)  # predict next (should be (batch_size, time_steps, n_outputs))
        print(next_Q_values)
        all_Q_values = self.model.predict(
            prev_states)  # predict current (should be (batch_size, time_steps, n_outputs))
        best_next_actions = np.argmax(next_Q_values,
                                      axis=2)  # best actions for next states (should be (batch_size, time_steps))

        target_Q_values = np.copy(all_Q_values)

        # Iterate over batch size and time steps correctly
        batch_size = len(prev_states)
        time_steps = len(prev_states[0])
        n_outputs = self.n_outputs

        for batch in range(batch_size):
            for t in range(time_steps - 1):
                action = prev_states[batch][t][-1]
                best_next_Q_value = next_Q_values[batch][t + 1][best_next_actions[batch][t + 1]]
                target_Q_values[batch][t][int(action)] = (all_Q_values[batch][t][int(action)] +
                                                          self.lr * (rewards[t] + (1 - terminals[t]) *
                                                                     self.discount_factor * best_next_Q_value -
                                                                     all_Q_values[batch][t][int(action)]))

        all_Q_values = np.asarray(target_Q_values)

        history = self.model.fit(prev_states, all_Q_values, verbose=0)  # calculate loss, update weights
        self.model_history.append(history.history['loss'])
        get_log_history()
        if self.update_counter % self.target_update_frequency == 0:
            self.target.set_weights(self.model.get_weights())
        self.update_counter += 1

    def get_current_states(self, custom_mem_len=None):
        current_states = []

        to = custom_mem_len if custom_mem_len else self.day_range+1
        for i in range(1, to):  #range(1, self.day_range + 1):
            current_states.append(self.memory[-i])
        #current_states = self.memory[-to:]

        current_states = np.array(current_states[::-1])
        current_states = current_states[np.newaxis, ...]
        if isinstance(self.input_shape, int):
            assert current_states.shape[
                       -1] == self.input_shape, f"Shape mismatch: {current_states.shape[-1]} != {self.input_shape}"
        else:
            assert current_states.shape[-1] == self.input_shape[
                -1], f"Shape mismatch: {current_states.shape[-1]} != {self.input_shape[-1]}"
        return current_states

    def get_previous_states(self):
        previous_states = []
        for i in range(2, self.day_range+2):
            previous_states.append(self.memory[-i])
        previous_states = np.array(previous_states[::-1])
        previous_states = previous_states[np.newaxis, ...]
        assert previous_states.shape[
                   -1] == self.input_shape, f"Shape mismatch: {previous_states.shape[-1]} != {self.input_shape}"

        return previous_states

    def get_previous_states_old(self):
        """
        Get the previous states for the model.
        """
        previous_states = []
        dummy_state = [0 for _ in range(len(self.memory[0]))]

        previous_states.append(dummy_state)
        for i in range(self.day_range-1):
            if len(self.memory) > i:
                previous_states.append(self.memory[i])
            else:
                previous_states.append(dummy_state)
        '''
        # Collect previous states, handling cases where memory length is less than intervention duration
        for i in range(self.day_range):
            if i == 0:
                previous_states.append(dummy_state)
            elif len(self.memory) < i + 1:
                previous_states.append(dummy_state)
            else:
                previous_states.append(self.memory[-i - 1])
        '''
        # Reverse the order of states to have the most recent state last
        #previous_states = np.array(previous_states[::-1])
        previous_states = np.array(previous_states)
        previous_states = previous_states[np.newaxis, ...]

        # Ensure the shape matches the input shape
        assert previous_states.shape[
                   -1] == self.input_shape, f"Shape mismatch: {previous_states.shape[-1]} != {self.input_shape}"

        return previous_states

    def get_previous_states_old(self):
        # TODO: add condition exact for 0 epoch
        previous_states = []
        if len(self.memory) <= self.intervention_dur:
            dummy_state = [0 for _ in range(len(self.memory[0]))]

            for i in range(self.day_range):
                if i == 0:
                    previous_states.append(dummy_state)
                else:
                    if len(self.memory) >= i:
                        previous_states.append(self.memory[-i])
                    else:
                        previous_states.append(dummy_state)
        else:
            for i in range(self.day_range):
                previous_states.append(self.memory[-i])
        '''
        for i in range(2, self.day_range+2):
            previous_states.append(self.memory[-i])
            #previous_states.append(self.memory[-(i + 1)])
        '''
        previous_states = np.array(previous_states[::-1])
        previous_states = previous_states[np.newaxis, ...]
        assert previous_states.shape[
                   -1] == self.input_shape, f"Shape mismatch: {previous_states.shape[-1]} != {self.input_shape}"
        return previous_states

    def execute_action(self, sim, action, economic, runner, simple_job_obj):
        '''
        Change hyperparameters of the E and Rw equations:
        alpha_ = 1
        beta_ = 1
        lambda_ = 1  # economic aspect first
        mu_ = 1  # infected count aspect first
        ro_ = 1  # deaths aspect first
        pi_ = 1  # vaccination aspect first

        Each of these depends on actions
        :param sim:
        :param action:
        :return:
        '''

        '''
        factor = 0.5
        sim.beta = beta * factor
        sim.sigma = sigma * factor
        sim.gamma = gamma * factor
        sim.mu = mu * factor
        sim.economic_impact_from_actions = factor

        '''
        sim.interventions = self.act_dict[action]
        simple_job_obj.interventions = self.act_dict[action]
        hyper = get_config_param(config_name="dqn_agent")['train_params']["hyper_params"]
        if hyper["constant_flag"] == 1:
            self.lambda_ = hyper['lambda']
            self.mu_ = hyper['mu']
            self.ro_ = hyper['ro']
            self.pi_ = hyper['pi']
            economic.alpha = hyper['alpha']
            economic.beta = hyper['beta']
            economic.gamma = hyper['gamma']
            economic.delta = hyper['delta']
        else:
            self.lambda_ = hyperparam_dict[action]  # economic aspect first # *=
            self.mu_ = hyperparam_dict[action]  # infected count aspect first
            self.ro_ = hyperparam_dict[action]  # deaths aspect first
            self.pi_ = hyperparam_dict[action]  #
            economic.alpha = hyperparam_dict[action]
            economic.beta = hyperparam_dict[action]
            economic.gamma = hyperparam_dict[action]
            economic.delta = hyperparam_dict[action]
        get_log_history()
        return simple_job_obj
        
        
        
#=============================================================================
#=======================================================================
#=================================================================================
    def run_simulation_old(self, num_days, name, datas_to_plot=None, run_simulation=None, extensionsList=None):
        """
        This main loop of the simulation.
        It advances the simulation day by day and saves,
        and after it finishes it saves the output data to the relevant files.
        :param num_days: int - The number of days to run
        :param name: str - The name of this simulation, will determine output
        directory path and filenames.
        :param datas_to_plot: Indicates what sort of data we wish to plot
        and save at the end of the simulation.
        :param Extension: user's class that contains function that is called at the end of each day
        """
        assert self.num_days_to_run is None
        self.num_days_to_run = num_days
        if datas_to_plot is None:
            datas_to_plot = dict()
        log.info("Starting simulation " + name)

        extensions = []
        if extensionsList != None:
            for ExtName in extensionsList:
                mod = __import__('src.extensions.' + ExtName, fromlist=[ExtName])
                ExtensionType = getattr(mod, ExtName)
                extensions = extensions + [ExtensionType(self)]

        # self.print_hh_statistics()

        for day in range(num_days):
            for ext in extensions:
                ext.start_of_day_processing()

            self.simulate_day()
            # Call Extension function at the end of the day
            for ext in extensions:
                ext.end_of_day_processing()

            if self.stats.is_static() or self.first_people_are_done():
                if self._verbosity:
                    log.info('simulation stopping after {} days'.format(day))
                print(f'run_simulation() simulation statistics stopped changing after {day} days')
                # break

        self.stats.mark_ending(self._world.all_people())

        # self.stats.calc_r0_data(self._world.all_people(), self.num_r_days, min_age=19, max_age=100)
        # if self.stats._r0_data:
        #     self.stats.plot_r0_data('r0_data_age_19_100_' + name)

        self.stats.calc_r0_data(self._world.all_people(), self.num_r_days)
        self.stats.dump('statistics.pkl')
        for name, data_to_plot in datas_to_plot.items():
            self.stats.plot_daily_sum(name, data_to_plot)
        # return this function
        # self.stats.write_daily_delta('dailydelta')
        self.stats.write_summary_file('summary')
        self.stats.write_summary_file('summary_long', shortened=False)
        if self.stats._r0_data:
            self.stats.plot_r0_data('r0_data_' + name)
        self.stats.write_params()
        self.stats.write_daily_delta('daily_delta')
        self.stats.write_inputs(self)
        self.stats.write_interventions_inputs_csv()

    # ===== NEW BLOCK ===
    
 def run_simulation_middle(self, num_days, name, datas_to_plot=None, run_simulation=None,
                              extensionsList=None, info=()):
        # self.print_hh_statistics()

        # ===== NEW BLOCK ===
        runner, agent, economic, sim, extensions, simple_job_obj, outdir, ext_param = info
        day = 0
        log.info('=' * 100)
        get_log_history()
        if runner.simulation_type == 'basic':  # "standard" simulation
            for day in range(num_days):
                runner.day_sim(self, extensions)
                # economic.get_population_statistic(self, vaccinated_seq)
                # economic.get_economic_index()

                # print(self.stats._r0_data)
                # self.stats.mark_ending(self._world.all_people())
                # self.stats.calc_r0_data(self._world.all_people(), self.num_r_days)
                # x = self.stats._r0_data
                # if x['estimated_r0'] != []:
                #    f=4
                # print(self.stats._r0_data)
                if self.stats.is_static() or self.first_people_are_done():
                    if self._verbosity:
                        log.info('simulation stopping after {} days'.format(day))
                    print(f'run_simulation() simulation statistics stopped changing after {day} days')
                    # break
        elif runner.simulation_type == 'reinforcement':
            runner.cum_reward = 0
            runner.cur_day = 0
            runner.incubation_day_counter = 0  #!@
            window = get_config_param("dqn_agent")['train_params']["actions_window"]
            #print('!!!!!!', runner.cur_day, economic.infected_population)
            # !!! decay
            is_init_period = True

            agent.epsilon = max(0.7 - (runner.episode / get_config_param("train_epochs")), 0.01)
            # ==== NEW EPSILON UPDATE RULE
            epsilon_start = 1.0
            epsilon_min = 0.1
            k = 0.01
            t = runner.episode
            #agent.epsilon = epsilon_min + (epsilon_start - epsilon_min) * math.exp(-k * t)
            #if agent.epsilon < 0.15:
            #    agent.epsilon = 0.15
            # ======
            log.info(f'=== EP# {runner.episode} ==== eps {agent.epsilon}======')
            action_call_counter = 0
            if runner.episode == 0:
                #agent.seen_states[0] += 1
                action_call_counter += 1
                for runner.cur_day in range(agent.day_range):
                    regular_immune_seq = runner.day_sim(self, extensions)

                    # TODO: first N days don't calculate economic and reward
                    economic.get_population_statistic(self, regular_immune_seq, 0, is_init_period,
                                                      aged_flag=get_config_param("age_stats"))

                    agent = runner.memorize(agent, economic, runner.cur_action, is_init_period)
                    agent.get_reward(economic, sim)  #!@ FIRST N days just epidemic raises
                    # runner.cum_reward += agent.reward  #!@ FIRST N days just epidemic raises
                    #agent.reward = -1000.0
                    agent.reward_memory_total.append(agent.reward)  #!@ FIRST N days just epidemic raises
                    agent.terminal_memory_total.append(0)
                    wandb_routine(log_params=(economic, sim, agent, runner.cur_action, runner),
                                  name=(runner.city_name, True))

                    log.info(
                        f'EP:{runner.episode}\tday:{runner.cur_day + 1}\treward:{agent.reward:.5f}\taction:{runner.cur_action}\n')
                is_init_period = False

            if runner.episode == 0:
                runner.cur_day += 1

            while runner.cur_day < runner.strategy['days_bound']:
                if action_call_counter == 0:
                    if runner.incubation_day_counter == agent.day_range:
                        runner.incubation_day_counter = 0
                        action_call_counter += 1
                else:
                    if runner.incubation_day_counter == agent.intervention_dur:
                        runner.incubation_day_counter = 0
                        action_call_counter += 1
                    '''
                    elif runner.incubation_day_counter <= agent.intervention_dur:
                        if runner.cur_action >= 0:
                            print(runner.history_actions_counter[runner.cur_action])
                            runner.history_actions_counter[runner.cur_action] += 1
                    '''
                '''
                if runner.cur_day == 0 and runner.incubation_day_counter == 0:
                    print('==>')
                    print(f'prev acion is {runner.cur_action}', end='\t')
                    runner.cur_action = 0

                    print(f'new action is {runner.cur_action}')
                    # runner.cur_action = 1
                    economic.chosen_action = runner.cur_action
                    simple_job_obj = agent.execute_action(self, runner.cur_action, economic, runner, simple_job_obj)
                    agent.update_act_dict(runner, economic, simple_job_obj)
                    sim.update_events(simple_job_obj.interventions, ext_param)
                '''
                current_states = agent.get_current_states()
                if runner.episode == 0:
                    pass
                else:
                    #if runner.episode == 0 and runner.cur_day + 1 == agent.day_range + 1:  # ep = 0 just run some action
                    #    pass
                    #else:
                    previous_states = agent.get_previous_states()
                    agent.update_model(previous_states, current_states)

                if runner.incubation_day_counter == 0 and runner.cur_day > 0:
                    print('==>')
                    print(f'prev acion is {runner.cur_action}', end='\t')
                    runner.cur_action, action_type = agent.epsilon_greedy_policy(current_states)

                    if action_type.lower() == 'eps':
                        runner.history_memory_access = 0
                        #runner.history_memory_access['MEMORY'] = False
                    elif action_type.lower() == 'memory':
                        runner.history_memory_access = 1
                        #runner.history_memory_access['MEMORY'] = True

                    agent.update_act_dict(runner, economic, simple_job_obj)

                    print(f'new action is {runner.cur_action}')
                    # runner.cur_action = 1
                    # Increment the history_actions_counter for the current action
                    if runner.cur_action >= 0:
                        runner.history_actions_counter[runner.cur_action] += 1

                    simple_job_obj = agent.execute_action(self, runner.cur_action, economic, runner, simple_job_obj)

                    sim.update_events(simple_job_obj.interventions, ext_param)
                    agent.seen_states[runner.cur_action] += 1

                regular_immune_seq = runner.day_sim(self, extensions)
                # economic.get_population_statistic(self, vaccinated_seq, 0, aged_flag=get_config_param("age_stats"))
                economic.get_population_statistic(self, regular_immune_seq, 0, is_init_period,
                                                  aged_flag=get_config_param("age_stats"))
                economic.chosen_action = runner.cur_action
                print(runner.cur_day + 1, runner.incubation_day_counter)
                runner.incubation_day_counter += 1
                runner.cur_day += 1
                agent = runner.memorize(agent, economic, runner.cur_action, is_init_period)
                agent.get_reward(economic, sim)

                if runner.cur_action >= 0:
                    runner.history_actions_reward[runner.cur_action].append(agent.reward)

                    #runner.cum_reward += agent.reward
                agent.reward_memory_total.append(agent.reward)

                wandb_routine(log_params=(economic, sim, agent, runner.cur_action, runner),
                              name=(runner.city_name, True))

                if runner.cur_day < runner.strategy['days_bound']:
                    agent.terminal_memory_total.append(0)

                is_init_period = False  # TODO: add init_period flag from simple_job_obj
                print({key: max(values) if len(values) > 0 else 0 for key, values in
                       runner.history_actions_reward.items()})
                #print(runner.history_actions_counter)
                log.info(
                    f'EP:{runner.episode}\tday:{runner.cur_day}\treward:{agent.reward:.5f}\taction:{runner.cur_action}\n')
                print('%' * 50)

            agent.terminal_memory_total.append(1)
            runner.cum_reward_graph.append(runner.cum_reward * (1 / runner.cur_day))
            return extensions, simple_job_obj    


 '''
        while runner.cur_day < runner.strategy['days_bound']:
            # Step 1: Simulate the current day to get cur_state

            print(len(cur_state), cur_state.shape)

            # Step 2: Choose an action based on cur_state
            action = agent.act(cur_state.reshape(1, cur_state.shape[1], 1))

            # Step 3: Simulate the next day to get next_state
            runner.cur_day += 1
            next_immune_seq = runner.day_sim(self, extensions)
            economic.get_population_statistic(self, next_immune_seq, 0, is_init_period,
                                                           aged_flag=get_config_param("age_stats"))
            agent = runner.memorize(agent, economic, runner.cur_action, is_init_period)
            next_state = np.array(agent.replay_buffer.memory[-1])

            # Step 4: Calculate the reward (e.g., negative of new infections)
            agent.get_reward(economic, sim)
            # Step 5: Store the experience in the replay buffer
            agent.remember(cur_state, action, agent.reward, next_state)

            # Step 6: Train the agent
            agent.replay()

            # Move to the next day in the loop
            runner.cur_day += 1

        '''
        '''
        for time in range(simulation_period):
            action = agent.act(state)
            next_state, reward = simulate_environment(action)
            next_state = np.reshape(next_state, [1, state_size, 1])

            agent.remember(state, action, reward, next_state)
            state = next_state

            agent.replay()
        '''
        
        
     def create_and_run_simulation_old(self, outdir, stop_early, with_population_caching=True, verbosity=False):
        """
        The main function that handles the run of the simulation by the task.
        It updated the params changes, loads or creates the population, initializes the simulation and runs it.
        :param outdir: the output directory for the task
        :param stop_early: only relevant to R computation, see Simulation doc
        :param with_population_caching: bool, if False generates the population, else - tries to use the cache and save time.
        :param verbosity: bool, if it's True then additional output logs will be printed to the screen
        """
        seed.set_random_seed()
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path) as json_data_file:
            ConfigData = json.load(json_data_file)
            citiesDataPath = ConfigData['CitiesFilePath']
            paramsDataPath = ConfigData['ParamsFilePath']
            Extensionslst = ConfigData['ExtensionsNamelst']
        Params.load_from(os.path.join(os.path.dirname(__file__), paramsDataPath), override=True)
        for param, val in self.params_to_change.items():
            Params.loader()[param] = val
        DiseaseState.init_infectiousness_list()

        citiesDataPath = citiesDataPath

        population_loader = PopulationLoader(
            citiesDataPath,
            added_description=Params.loader().description(),
            with_caching=with_population_caching,
            verbosity=verbosity
        )

        world = population_loader.get_world(city_name=self.city_name, scale=self.scale, is_smart=True)

        ExtensionType = None

        sim = Simulation(world, self.initial_date, self.interventions,
                         verbosity=verbosity, outdir=outdir, stop_early=stop_early,
                         extension_params={"ImmuneByAgeExtension": self.infection_params})
        self.infection_params.infect_simulation(sim, outdir)
        if len(Extensionslst) > 0:
            sim.run_simulation(self.days, self.scenario_name, datas_to_plot=self.datas_to_plot,
                               extensionsList=Extensionslst)
        else:
            sim.run_simulation(self.days, self.scenario_name, datas_to_plot=self.datas_to_plot, extensionsList=None)
