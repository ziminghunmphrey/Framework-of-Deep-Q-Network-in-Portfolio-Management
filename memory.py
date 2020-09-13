import numpy as np

#This class is the structure of memory pool
class SumTree:

    def __init__(self, capacity):
        self.capacity = capacity      #capacity is the number of memories that the memory pool can contain
        self.tree = np.zeros(2 * capacity - 1)     #the number of nodes of sumtree
        self.data = {}
        self.data_pointer=0

    #The method is to add new memory to the memory pool
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    #calculate TD-error of each node after new memory added to the memory pool
    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

   #search the memory with the largest TD-error
    def get_leaf(self, v):

        parent_idx = 0

        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        data_idx = leaf_idx - self.capacity + 1

        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_p(self):
        return self.tree[0]



#memory pool
class Memory:

    def __init__(self, action_num, actions ,memory_size, batch_size, epsilon, alpha, beta, beta_increment_rate, err_upp):

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory_pointer = 0
        self.action_num=action_num
        self.actions=actions
        self.rewards = []
        self.full = False
        self.tree = SumTree(memory_size)
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_rate
        self.abs_err_upper = err_upp

    #take a batch of samples form the memory pool
    def sample(self):

        observations = {'history':np.zeros((self.batch_size, self.history_size[0], self.history_size[1], self.history_size[2])),
                        'weights':np.zeros((self.batch_size, self.weight_size[0]))}

        observations_ = {'history':np.zeros((self.batch_size, self.history_size[0], self.history_size[1], self.history_size[2])),
                         'weights':np.zeros((self.batch_size, self.weight_size[0]))}

        actions_idx = []

        rewards = []

        ISWeights = np.empty((self.batch_size, 1))     #initialize the weight of each sample, to obtain the loss function

        b_idx=np.empty((self.batch_size,), dtype=np.int32)

        pri_seg = self.tree.total_p() / self.batch_size

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p()

        for i in range(self.batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p()
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i] = idx
            observations['history'][i, :, :, :] = data[0]['history']
            observations['weights'][i, :] = data[0]['weights']
            observations_['history'][i, :, :, :] = data[3]['history']
            observations_['weights'][i, :] = data[3]['weights']
            actions_idx.append(data[1])
            rewards.append(data[2])

        return observations, actions_idx, rewards, observations_, b_idx, ISWeights

    #add a memory to the memory pool
    def store(self, observation, action, reward, observation_):

        if len(self.tree.data)>=self.memory_size:
            self.full = True
            self.history_size = np.shape(self.tree.data[0][0]['history'])
            self.weight_size = np.shape(self.tree.data[0][0]['weights'])

        max_p = np.max(self.tree.tree[-self.tree.capacity:])

        if max_p == 0:
            max_p = self.abs_err_upper

        transition = (observation, action, reward, observation_)

        self.tree.add(max_p, transition)

        self.rewards.append(reward)

    #update the node of sumtree
    def batch_update(self, tree_idx, abs_errors):

        abs_errors += self.epsilon

        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)

        ps = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    #when the memory pool is full, start experience replay process
    def start_replay(self):
        return self.full