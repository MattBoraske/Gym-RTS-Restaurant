import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import norm
import random

class Order:
    def __init__(self, items, arrival_time):
        self.items = items
        self.arrival_time = arrival_time
        self.submission_time = None
        self.submitted = False

class FoodItem:
    def __init__(self, item_id, creation_time):
        self.item_id = item_id
        self.creation_time = creation_time

class RestaurantActionSpace(spaces.Box):
    def __init__(self, num_employees, num_tasks):
        super().__init__(low=0, high=num_employees, shape=(num_tasks,), dtype=int)
        self.num_employees = num_employees
        self.num_tasks = num_tasks

    def sample(self):
        # Generate a valid action that satisfies the constraint of the number of assigned workers not exceeding the total amount
        action = np.zeros(self.num_tasks, dtype=int)
        remaining_employees = self.num_employees

        # Randomly shuffle the task IDs to mitigate task order bias
        task_ids = np.arange(self.num_tasks)
        np.random.shuffle(task_ids)

        # Assign workers to the first n-1 tasks
        for i in range(self.num_tasks - 1):
            task_id = task_ids[i]
            max_workers = min(remaining_employees, self.num_employees - self.num_tasks + i + 1)
            action[task_id] = np.random.randint(0, max_workers + 1)
            remaining_employees -= action[task_id]

        # Assign the remaining employees to the last task
        last_task_id = task_ids[-1]
        action[last_task_id] = remaining_employees

        return action

class RTS_Restaurant_Env(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is an env that simulates a fast-food restaurant.
    The agent learns to allocate employees to different tasks to minimize customer wait time, which consequently maximizes revenue. 
    Please note that this env abstracts away the payment process and solely focuses on minimizing customer wait time.
    """

    def __init__(self, total_workers, tasks, max_new_customers, max_total_customers, render_mode="console"):
        """
        Args:
            total_workers (int): Total number of employees
            tasks (list): List of dictionaries containing information about each task
                Each dictionary should contain the following keys:
                    'task_type' (str): Type of task (either 'BOH' or 'FOH')
                    'yield_per_worker' (float): Number of items produced by a single worker in a unit of time
                    'item_ordering_weight' (float): Weight of the item in the ordering frequency distribution
                    'storage_capacity' (int): Maximum number of items that can be stored for the task
            max_new_customers (int): Maximum number of new customers that can arrive at a FOH station at a given time
            max_total_customers (int): Maximum number of customers that can be in the restaurant at a given time
            render_mode (str): Rendering mode. Currently supports 'console' only
        """
        super(RTS_Restaurant_Env, self).__init__()
        self.render_mode = render_mode

        if total_workers < 1:
            raise ValueError('total_workers must be greater than 0')
        
        # check that tasks has 'task_type' and 'yield_per_worker' keys
        if not all('task_type' in task and 'yield_per_worker' in task and 'item_ordering_weight' in task and 'storage_capacity' in task for task in tasks):
            raise ValueError('tasks must contain keys "task_type", "yield_per_worker", "item_ordering_weight", and "storage_capacity"')
        
        # check that task_type is either 'BOH' or 'FOH'
        if not all(task['task_type'] in ['BOH', 'FOH'] for task in tasks):
            raise ValueError('task_type must be either "BOH" (back of house) or "FOH" (front of house)')
        
        # check that yield_per_worker is a positive number
        if not all(task['yield_per_worker'] > 0 for task in tasks):
            raise ValueError('yield_per_worker must be greater than 0')
        
        # check that max_new_customers is a positive number
        if max_new_customers < 1:
            raise ValueError('max_new_customers must be greater than 0')
        
        # check that item_ordering_weight is a positive number for the BOH tasks
        if not all(task['item_ordering_weight'] > 0 for task in tasks if task['task_type'] == 'BOH'):
            raise ValueError('item_ordering_weight must be greater than 0 for BOH tasks')
        
        # check that item_ordering_weight is 0 for the FOH tasks
        if not all(task['item_ordering_weight'] == 0 for task in tasks if task['task_type'] == 'FOH'):
            raise ValueError('item_ordering_weight must be 0 for FOH tasks')

        # check that storage_capacity is a positive number for the BOH tasks
        if not all(task['storage_capacity'] > 0 for task in tasks if task['task_type'] == 'BOH'):
            raise ValueError('storage_capacity must be greater than 0 for BOH tasks')
    
        # check that the storage capacity is 0 for the FOH tasks
        if not all(task['storage_capacity'] == 0 for task in tasks if task['task_type'] == 'FOH'):
            raise ValueError('storage_capacity must be 0 for FOH tasks')
        
        # Assign environment parameters
        self.total_workers = total_workers # Total number of employees
        self.tasks = tasks # Tasks dictionary
        self.max_new_customers = max_new_customers # Maximum number of new customers that can arrive at the restaurant at a given time
        self.max_total_customers = max_total_customers # Maximum number of customers that can be in the restaurant at a given time
        self.task_types = ['BOH', 'FOH'] # Two types of tasks - back of house (BOH) and front of house (FOH)
        self.current_time = 0 # Time in minutes from the start of the restaurant day (0 to 1080 minutes)
        self.customer_orders = [] # List of customer orders
        self.food_inventory = [] # List of food items in the inventory
        self.items_for_agent_to_consider = 1000 # Number of most recent ordered items for the agent to consider
        self.rejected_customers = 0 # Number of customers that were unable to be served

        # worker assignments dictionary
        self.worker_assignments = {
            i: {
                'num_workers': 0, # starts at 0 for all tasks
                'task_type': '',              
                'in_progress_inventory': 0, # starts at 0 for all tasks
                'storage_capacity': 0,
                'yield_per_worker': 0,
            } for i in range(len(self.tasks))
        }

        # populate worker_assignment dictionary with task information
        for i, task in enumerate(self.tasks):
            self.worker_assignments[i]['task_type'] = task['task_type']
            self.worker_assignments[i]['yield_per_worker'] = task['yield_per_worker']
            self.worker_assignments[i]['storage_capacity'] = task['storage_capacity']

        # reward parameters
        self.reward_params = {
            'fulfilled_order': 10,
            'unfulfilled_order': -10,
            'idle_worker': -3,
            'submission_to_fulfillment_time': -1,
            'arrival_to_submission_time': -2,
            'extra_items_penalty': -3,
            'unable_to_serve_customer': -5,
        }

        # create item ordering frequencies using the task item ordering weights
        self.item_ordering_frequencies = {}
        for i,task in enumerate(tasks):
            if task['task_type'] == 'BOH':
                self.item_ordering_frequencies[i] = task['item_ordering_weight']
        total_weight = sum(self.item_ordering_frequencies.values())
        for i in self.item_ordering_frequencies:
            self.item_ordering_frequencies[i] /= total_weight

        # Action space is the number of employees that can be assigned to each task
        self.action_space = RestaurantActionSpace(self.total_workers, len(self.tasks))

        # Observation space is a dictionary containing sequences for customer orders and food inventory
        self.num_BOH_tasks = sum(1 for task in tasks if task['task_type'] == 'BOH')
        self.observation_space = spaces.Dict({
            i: spaces.Dict({
                '1_assigned workers': spaces.Box(low=0, high=self.total_workers, shape=(1,), dtype=int),
                '2_ordered_item_count': spaces.Box(low=0, high=self.items_for_agent_to_consider, shape=(1,), dtype=int),
                '3_created_item_count': spaces.Box(low=0, high=self.worker_assignments[i]['storage_capacity'], shape=(1,), dtype=int),
                '4_customers_waiting_for_serving': spaces.Box(low=0, high=self.max_new_customers, shape=(1,), dtype=int),
                '5_customers_waiting_for_fulfillment': spaces.Box(low=0, high=self.max_new_customers, shape=(1,), dtype=int),
            }) for i in range(len(self.tasks))
        })

        # Initialize the observation dictionary
        self.observation = {
            i: {
                '1_assigned workers': 0,
                '2_ordered_item_count': 0,
                '3_created_item_count': 0,
                '4_customers_waiting_for_serving': 0,
                '5_customers_waiting_for_fulfillment': 0
            } for i in range(len(self.tasks))
        }

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
            :return: (np.array)
        """
        super().reset(seed=seed, options=options)

        # reset the observation dictionary
        self.observation = {
            i: {
                '1_assigned workers': 0,
                '2_ordered_item_count': 0,
                '3_created_item_count': 0,
                '4_customers_waiting_for_serving': 0,
                '5_customers_waiting_for_fulfillment': 0
            } for i in range(len(self.tasks))
        }

        # reset the current time and rejected
        self.current_time = 0
        self.rejected_customers = 0
        info = {
            'current_time': self.current_time
        }

        return self.observation, info

    def step(self, action):
        # initialize reward and rejected customers for the current timestep
        reward = 0
        self.rejected_customers = 0

        # get the number of workers for each task and assign them
        for i, num_workers in enumerate(action.astype(int)):
            self.worker_assignments[i]['num_workers'] = num_workers

        # apply reward penalty for idle workers if they exist
        idle_workers = self.total_workers - sum(self.worker_assignments[assignment]['num_workers'] for assignment in self.worker_assignments)
        if idle_workers > 0:
            reward += self.reward_params['idle_worker'] * idle_workers
        
        # simulate customer arrivals
        self.get_new_orders(self.current_time)

        # apply reward penalty for customers that were unable to be served
        if self.rejected_customers > 0:
            reward += self.reward_params['unable_to_serve_customer'] * self.rejected_customers
      
        # simulate restaurant operations
        for assignment in self.worker_assignments:
            if self.worker_assignments[assignment]['task_type'] == 'BOH':
                # simulate food production
                self.worker_assignments[assignment]['in_progress_inventory'] += self.worker_assignments[assignment]['yield_per_worker'] * self.worker_assignments[assignment]['num_workers']
                if self.worker_assignments[assignment]['in_progress_inventory'] >= 1:
                    self.food_inventory.append(FoodItem(item_id=0, creation_time=self.current_time))
                    self.worker_assignments[assignment]['in_progress_inventory'] -= 1

            elif self.worker_assignments[assignment]['task_type'] == 'FOH':
                # submit orders if there are any to be submitted
                submission_opportunities = int(self.worker_assignments[assignment]['num_workers'] * self.worker_assignments[assignment]['yield_per_worker'])
                for order in self.customer_orders:
                    if not order.submitted and submission_opportunities > 0:
                        order.submitted = True
                        order.submission_time = self.current_time
                        submission_opportunities -= 1

                    if submission_opportunities == 0:
                        break

                # Fulfill orders if there are any that can be fulfilled with the current food inventory
                fulfillment_opportunities = int(self.worker_assignments[assignment]['num_workers'] * self.worker_assignments[assignment]['yield_per_worker'])
                for order in self.customer_orders:
                    if order.submitted and fulfillment_opportunities > 0:
                        # Count the occurrences of each item in the order
                        order_item_counts = {}
                        for item in order.items:
                            order_item_counts[item.item_id] = order_item_counts.get(item.item_id, 0) + 1

                        # Check if all items in the order are available in the food inventory
                        order_fulfilled = True
                        for item_id, count in order_item_counts.items():
                            if sum(1 for food_item in self.food_inventory if food_item.item_id == item_id) < count:
                                order_fulfilled = False
                                break

                        if order_fulfilled:
                            # Remove the items from the food inventory
                            for item_id, count in order_item_counts.items():
                                for _ in range(count):
                                    self.food_inventory.remove(next(food_item for food_item in self.food_inventory if food_item.item_id == item_id))

                            # calculate reward for fulfilling the order
                            reward += self.reward_params['fulfilled_order']
                            reward += self.reward_params['submission_to_fulfillment_time'] * (self.current_time - order.submission_time)
                            reward += self.reward_params['arrival_to_submission_time'] * (order.submission_time - order.arrival_time)
                            
                            # Remove the order from the customer orders
                            self.food_inventory.remove(order)
                            fulfillment_opportunities -= 1

                        if fulfillment_opportunities == 0:
                            break

        # update the observation dictionary
        # update the observation dictionary
        for i, assignment in enumerate(self.worker_assignments):
            # assigned workers
            self.observation[i]['1_assigned workers'] = self.worker_assignments[assignment]['num_workers']
            
            # ordered item count - the total amount of this item across all customer orders
            item_counts = {}
            for order in self.customer_orders:
                for item in order.items:
                    if item not in item_counts:
                        item_counts[item] = 0
                    item_counts[item] += 1
            self.observation[i]['2_ordered_item_count'] = item_counts.get(i, 0)
            
            # created item count - the amount of this item in the food inventory
            self.observation[i]['3_created_item_count'] = self.worker_assignments[assignment]['in_progress_inventory']
            
            # customers waiting for serving - the amount of customers that have not yet submitted their orders (submitted=False)
            self.observation[i]['4_customers_waiting_for_serving'] = sum(1 for order in self.customer_orders if not order.submitted)
            
            # customers waiting for fulfillment - the amount of customers that have submitted their orders but are not yet fulfilled (submitted=True)
            self.observation[i]['5_customers_waiting_for_fulfillment'] = sum(1 for order in self.customer_orders if order.submitted)

        ## edge case of episode ending (timestep == 1079)
        terminated = bool(self.current_time == 1079)

        if terminated:
            # penalize unfulfilled orders
            reward += self.reward_params['unfulfilled_order'] * len(self.customer_orders)
            # penalize extra items in the food inventory
            reward += self.reward_params['extra_items_penalty'] * len(self.food_inventory)
            
        # increment time
        self.current_time += 1

        # pass back current time as extra info
        info = {'current_time': self.current_time}

        return (
            self.observation,
            reward,
            terminated,
            False, # using terminated to signal end of episode (aka current_time == 1079)
            info,
        )
    
    def get_new_orders(self, time):
        """
        Sample the number of customers that arrive at a given time
        """
        
        # sample the number of new customers
        customer_probability = self.get_customer_arrival_probability(time)
        new_customers = np.random.choice(np.clip(np.random.poisson(lam=customer_probability * self.max_new_customers, size=1000), 0, self.max_new_customers))
        
        # create new customer orders
        for _ in range(int(new_customers)):
            # check if the restaurant has reached the maximum number of customers and reject if so
            if len(self.customer_orders) >= self.max_total_customers:
                self.rejected_customers += 1
            else:
                # sample the number of items ordered by the customer
                num_items_ordered = np.random.choice(np.clip(np.random.poisson(lam=4, size=1000), 1, 10)) # avg of 4, 1 to 10 items - possibly parameterize this later
                new_order_items = random.choices(list(self.item_ordering_frequencies.keys()), 
                                    weights=list(self.item_ordering_frequencies.values()), 
                                    k=num_items_ordered)
                
                # create a new customer order and add them to observation
                new_order = Order(items=new_order_items, arrival_time=time)
                self.customer_orders.append(new_order)


    def get_customer_arrival_probability(self, time):
        """
        Get the probability of a customer arriving at a given time

        :param time: (int) Time in minutes from the start of the restaurant day (0 to 1080 minutes)
        :return: (float) Probability of a customer arriving at the given time
        """

        if time > 1080:
            return ValueError('Time must be between 0 and 1080 minutes (18 hours)')

        # spikes for breakfast and dinner - https://squareup.com/us/en/press/square-q2-restaurant-industry-report-post-pandemic-spending-shifts-from-friday
        spike_width = 100  # Increase the spike width for more pronounced spikes

        # generate a normal distribution
        mean = 0
        std_dev = 0.5
        num_points = 1000
        x = np.linspace(-1, 1, num_points)
        y = norm.pdf(x, mean, std_dev)

        # calculate spike positions
        breakfast_spike = norm.ppf(0.125, mean, std_dev)
        dinner_spike = norm.ppf(0.875, mean, std_dev)

        # add meal spikes to curve
        max_y = np.max(y)
        # breakfast spike 
        breakfast_spike_height = 1 * max_y
        y += breakfast_spike_height * np.exp(-spike_width * (x - breakfast_spike) ** 2)
        # dinner spike
        dinner_spike_height = 1.5 * max_y
        y += dinner_spike_height * np.exp(-spike_width * (x - dinner_spike) ** 2)

        # normalize frequencies so they range from 0 to 1
        y /= np.max(y)

        # use hanning window to smooth curve so that start and end are zero
        window = np.hanning(num_points)
        y *= window

        # return the customer frequency at the given time
        return y[0] 

    def render(self):
        if self.render_mode == "console":
            print('---------------------------------')
            print('CURRENT STEP')
            print(f'\tOrders: {self.observation["customer_orders"]}')
            print(f'\tItems in food inventory: {self.observation["food_inventory"]}')
            print(f'\tWorker Assignments: {self.observation["worker_assignments"]}')
            print('---------------------------------')

    def close(self):
        pass