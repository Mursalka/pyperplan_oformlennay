from itertools import product


class Domain(object):

    def __init__(self, actions=()):
        self.actions = tuple(actions)

    def grounding(self, objects):
        actions = list()
        for action in self.actions:
            parameters = [objects[t] for t in action.types]
            combinations = set()
            for p in product(*parameters):
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
        if isinstance(i, list):
            rt.extend(flatten(i))
        else:
            rt.append(i)
    return rt


def create_domain(data):
    actions = list()

    for action, params in data["action"].items():
        param = list()
        for _type, variable_list in params["parameters"].items():
            for variable in variable_list:
                param.append((_type, variable))

        precon = list()
        for name, predicates in params["precondition"].items():
            precon.append(tuple(flatten([name, predicates])))

        eff = list()
        for name, predicates in params["effect"].items():
            eff.append(tuple(flatten([name, predicates])))

        actions.append(Action(action, param, precon, eff))
    return Domain(actions)


class Task(object):

    def __init__(self, domain, objects, init=(), goal=()):
        self.grounded_actions = domain.grounding(objects)

        predicates = list()
        functions = dict()
        for predicate in init:
            predicates.append(predicate)
        self.init = (predicates, functions)

        self.goal = list()
        for g in goal:
            self.goal.append(g)


def apply_grounding(arg_names, args):
    names = dict()
    for name, arg in zip(arg_names, args):
        names[name] = arg

    def tmp(predicate):
        return predicate[0:1] + tuple(names.get(arg, arg) for arg in predicate[1:])

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


import json

with open('domain.json', 'r') as domain_file:
    domain_data = json.load(domain_file)
domain = create_domain(domain_data)

with open('task.json', 'r') as task_file:
    task_data = json.load(task_file)
problem = create_problem(domain, task_data)

print("Actions:")
[print(i.__str__()) for i in domain.actions]

print("\nAction combinations:")
[print(i.__str__()) for i in problem.grounded_actions]
