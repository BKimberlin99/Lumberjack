"""
Brandon Kimberlin
OpenAI gym environment: gym/envs/toy_text/frozen_lake.py
Attempting 16x16 size environment
The agent uses an epsilon-greedy approach with Q-learning to
    decide what action to take.
"""

"""
Question Answers:

1) The size of the grade does not alter the difficulty of the programming
    but it does take longer for the agent to learn the correct way to get to
    the goal

2) I've set the agent to use an epsilon-greedy approach with Q-learning
    to decide what action to take.

3) 
"""
import sys
from contextlib import closing

import numpy as np
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

epsilon = 0.9
discount = 0.96
# learning rate
lr = 0.9
MAX_EPISODES = 100

MAX_STEPS = 1000
WEST = 0
SOUTH = 1
EAST = 2
NORTH = 3

""" Randomly choose a number between 0 and 1, if that number is smaller
    than epsilon, a random action is chosen.
    If it's greater than epsilon, choose the action having the highest value
    in the Q-table"""
def choose_action(cur_state):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[cur_state, :])

    return action

""" Updates Q-table with expected rewards based on states and learning rate"""
def learn(cur_state, new_state, reward, action):
    predict = Q[cur_state, action]
    target = reward + discount * np.max(Q[new_state, :])
    Q[cur_state, action] = Q[cur_state, action] + lr * (target - predict)
    
def generate_random_map(size=16, p=0.9):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0,0))
        while frontier:
            r, c = frontier.pop()
            if not (r,c) in discovered:
                discovered.add((r,c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] not in '#B'):
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'B'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


class ForestEnv(discrete.DiscreteEnv):
    """
    You're a lumberjack. While you're out chopping wood, you lay your axe down and decide to walk
    around and take a break from chopping. While walking, you black out and when you come to,
    you have no idea where in the forest you are or where your axe is. While searching for your
    axe, you notice prints in the dirt, large prints belonging to what can only be a bear.
    Upon seeing this, you start searching for your axe quicker but as you search you see more
    tracks of varying sizes, all of them being bear tracks. You realize that you are lost, alone and
    unarmed in the woods with a family of bears.
    If you travel into the same area as a bear, you know the bear is going to maul you to death.
    You need to find your axe to have any hope of surviving this forest.   
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : forest, safe
    B : bear, become a snack for the Berenstain Bears
    G : goal, where the axe is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        desc = generate_random_map()
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == WEST:
                col = max(col-1,0)
            elif a == SOUTH:
                row = min(row+1,nrow-1)
            elif a == EAST:
                col = min(col+1,ncol-1)
            elif a == NORTH:
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GB':
                        li.append((1.0, s, 0, True))
                    else:
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]
                        done = bytes(newletter) in b'GB'
                        rew = float(newletter == b'G')
                        li.append((1.0, newstate, rew, done))


        super(ForestEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["West","South","East","North"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
            
# Generate starting environment
env = ForestEnv()

# Run multiple episodes of the forest
for episode in range(MAX_EPISODES):
    cur_state = env.reset()
    env.render()

    """Set up Q table filled with zeros to later be filled with max expected rewards
    for different states"""
    Q = np.zeros((env.observation_space.n, env.action_space.n))


    # Agent moves based on Q-learning, dying if it finds a bear, winning if it reaches
    # the axe    
    for i in range(MAX_STEPS):
        action = choose_action(cur_state)
        new_state, reward, done, info = env.step(action)
    
        # call learn function so agent can use Q-learning to learn the
        # correct path
        learn(cur_state, new_state, reward, action)
    
        # environment is rendered after action, showing what action was made and
        # the new state
        cur_state = new_state
        env.render()
        if done:
            print("Reward = ", reward, "\n")
            break




        
