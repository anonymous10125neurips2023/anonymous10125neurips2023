import numpy as np 

def sample_offline_actions(opt_actions, num_contexts, num_actions, behavior_pi='eps-greedy', behavior_epsilon=0.1, partial_space_rate = 0.5): 
    """Sample offline actions 
    Args:
        opt_actions: (num_contexts,)
        num_contexts: int 
        num_actions: int
    """
    if behavior_pi == 'partial-space-unif': # uniform over a partial action space 
        return np.random.randint(low=0, high=int(partial_space_rate * num_actions) , size=(num_contexts,)) 
    elif behavior_pi == 'eps-greedy':
        subopt_actions = []
        for i in range(num_contexts):
            subopt_actions.append(np.random.choice( [a for a in range(num_actions) if a != opt_actions[i] ])  )
        subopt_actions = np.array(subopt_actions)
        delta = np.random.uniform(size=(num_contexts,))
        selector = np.array(delta <= behavior_epsilon).astype('float32') 
        actions = selector.ravel() * subopt_actions + (1 - selector.ravel()) * opt_actions 
        actions = actions.astype('int')
        return actions
    elif behavior_pi == 'unif':
        return np.random.randint(low=0, high=num_actions, size=(num_contexts,)) 
    else: 
        raise NotImplementedError('{} is not implemented.'.format(behavior_pi))