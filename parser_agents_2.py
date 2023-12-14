from itertools import product

class Domain(object):

    def __init__(self, actions=(), agents={}):
        self.actions = tuple(actions)
        self.agents = agents

    def grounding(self, objects):
        actions = list()
        agents = self.agents.keys()
        for action in self.actions:
            parameters = [objects[t] for t in action.types]
            combinations = set()
            for p in product(product(*parameters), agents):
                parameters_set = frozenset(p)
                if action.unique and len(parameters_set) != len(p):
                    continue
                if action.same and parameters_set in combinations:
                    continue
                combinations.add(parameters_set)
                actions.append(action.grounding(*p))
        return actions
    
class Action(object):
    
    def __init__(self, name, parameters=(), preconditions=(), effects=(), unique=False, same=False):
        self.name = name
        if len(parameters) > 0:
            self.types, self.arg_names = zip(*parameters)
        else:
            self.types = tuple()
            self.arg_names = tuple()
        self.preconditions = preconditions
        self.effects = effects
        self.unique = unique
        self.same = same

    def grounding(self, *args):
        return GroundedAction(self, *args)

    def __str__(self):
        arglist = ', '.join(['%s - %s' % pair for pair in zip(self.arg_names, self.types)])
        return '%s(%s)' % (self.name, arglist)

def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list):
            rt.extend(flatten(i))
        else:
            rt.append(i)
    return rt

def create_domain(data):
    agents = dict()
    for agent, weight in data["agent"].items():
        agents[agent] = weight
    
    actions = list()
    
    for action, params in data["action"].items():
        param = list()
        for _type, variable_list in params["parameters"].items():
            for variable in variable_list:
                param.append((_type, variable))
        
        precon = list()
        for name, predicates in params["precondition"].items():
            precon.append(tuple(flatten([name,predicates])))
        
        eff = list()
        for name, predicates in params["effect"].items():
            eff.append(tuple(flatten([name,predicates])))
        
        actions.append(Action(action, param, precon, eff))
    return Domain(actions, agents)

class Task(object):

    def __init__(self, domain, objects, init=(), goal=()):
        self.grounded_actions = domain.grounding(objects)

        predicates = list()
        functions = dict()
        for predicate in init:
            predicates.append(predicate)
        self.init = State(predicates, functions)

        self.goal = list()
        for g in goal:
            self.goal.append(g)
        self.weights = objects
        self.agents = domain.agents
            
def apply_grounding(arg_names, args):
    names = dict()
    for name, arg in zip(arg_names, args[0]):
        names[name] = arg
    def tmp(predicate):
        return predicate[0:1] + tuple(names.get(arg, arg) for arg in predicate[1:]) + (args[1],)
    return tmp

class GroundedAction(object):

    def __init__(self, action, *args):
        self.name = action.name
        ground = apply_grounding(action.arg_names, args)

        self.written = ground((self.name,) + action.arg_names)

        self.preconditions = list()
        for pre in action.preconditions:
                self.preconditions.append(ground(pre))
        self.effects = list()
        for effect in action.effects:
            self.effects.append(ground(effect))

    def __str__(self):
        arglist = ', '.join(map(str, self.written[1:]))
        return '%s(%s)' % (self.written[0], arglist)

    
def create_problem(domain, data):
    obj = dict()
    for _type, variables in data["objects"].items():
        obj[_type] = variables
      
    init = list()
    for name, objects in data["init"].items():
        for object_ in objects:
            init.append(tuple(flatten([name, object_])))
            
    goal = list()
    for name, objects in data["goal"].items():
        for object_ in objects:
            goal.append(tuple(flatten([name, object_])))
        
    return Task(domain, obj, init, goal)

class State(object):

    def __init__(self, predicates, functions, predecessor=None):
        self.predicates = frozenset(predicates)
        self.functions = tuple(functions.items())
        self.f_dict = functions
        self.predecessor = predecessor

    def apply(self, action):
        no_agent_eff = [p[:-1] for p in action.effects]
        no_agent_precon = [p[:-1] for p in action.preconditions]
        new_preds = set(self.predicates)
        new_preds |= set(no_agent_eff)
        new_preds -= set(no_agent_precon)
        new_functions = dict()
        new_functions.update(self.functions)
        return State(new_preds, new_functions, (self, action))

    def plan(self):
        plan = list()
        n = self
        while n.predecessor is not None:
            plan.append(n.predecessor[1])
            n = n.predecessor[0]
        plan.reverse()
        return plan

    def __hash__(self):
        return hash((self.predicates, self.functions))

    def __eq__(self, other):
        return ((self.predicates, self.functions) ==
                (other.predicates, other.functions))

    def __lt__(self, other):
        return hash(self) < hash(other)

import heapq

def null_heuristic(state, new_state=None, goal=None):
    return 0

def is_true(state, predicates, problem):
    no_agent_pred = [p[:-1] for p in predicates]
    flag = all(p in state.predicates for p in no_agent_pred) and sum(['holding' in i[0] for i in state.predicates]) < 3
    a_w = problem.agents.get(predicates[0][-1])
    b_w = [list(problem.weights.values())[0].get(p[1]) for p in predicates]
    flag2 = all(a_w[0] <= b and a_w[1] >= b for b in b_w if b is not None)
    return flag + flag2

def equal(state, predicates):
    return all(p in state.predicates for p in predicates)

def planner(problem, heuristic=None, state0=None, goal=None):
    if heuristic is None:
        heuristic = null_heuristic
    if state0 is None:
        state0 = problem.init
    if goal is None:
        goal = tuple(problem.goal)

    closed = set()
    fringe = [(heuristic(state0), state0)]
    heapq.heapify(fringe)
    while True:
        if len(fringe) == 0:
            return None

        h, node = heapq.heappop(fringe)

        if is_true(node, goal):
            plan = node.plan()
            return plan

        if node not in closed:
            closed.add(node)
            successors = set(node.apply(action)
                             for action in problem.grounded_actions
                             if is_true(node, action.preconditions))

            for successor in successors:
                if successor not in closed:
                    heapq.heappush(fringe, (0, successor))

def h_g_heuristic(state, new_state, goal):
    h_g = 0
    for s in new_state.predicates:
        if s in state.predicates:
            h_g += 1
        if s in goal:
            h_g += 1
    print(h_g)
    return h_g

def Astar_planner(problem, heuristic=null_heuristic, state0=None, goal=None):
    if heuristic is None:
        heuristic = null_heuristic
    if state0 is None:
        state0 = problem.init
    if goal is None:
        goal = tuple(problem.goal)
        
    equal(state0, state0.predicates)

    closed = set()
    fringe = [(heuristic(state0, State(list(),dict()), goal), state0)]
    heapq.heapify(fringe)
    while True:
        if len(fringe) == 0:
            return None
        
        h, node = heapq.heappop(fringe)
        tmp = [p[:-1] for p in node.predicates]

        if equal(node, goal):
            plan = node.plan()
            return plan

        if node not in closed:
            closed.add(node)
            successors = set(node.apply(action)
                             for action in problem.grounded_actions
                             if is_true(node, action.preconditions, problem))
            for successor in successors:
                if successor not in closed:
                    heapq.heappush(fringe, (heuristic(node, successor, goal), successor))

import json 

with open('domain_weight.json', 'r') as domain_file:
    domain_data = json.load(domain_file)
domain = create_domain(domain_data)

with open('task_weight.json', 'r') as task_file:
    task_data = json.load(task_file)
problem = create_problem(domain, task_data)

print("Actions:")
[print(i.__str__()) for i in domain.actions]

print("\nAction combinations:")
[print(i.__str__()) for i in problem.grounded_actions]

#print("\nPlan BFS:")
#[print(i.__str__()) for i in planner(problem)]

print("\nPlan Astar:")
[print(i.__str__()) for i in Astar_planner(problem)];




