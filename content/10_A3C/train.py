import gym
from shared_adam import SharedAdam
import torch.multiprocessing as mp
from worker import Worker

network_type = "discrete"
if network_type == "continuous":
    from continuous_action_model import Net
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    N_S = env.observation_space.shape[0]
    N_A = env.action_space.shape[0]
elif network_type == "discrete":
    from discrete_action_model import Net
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    N_S = env.observation_space.shape[0]
    N_A = env.action_space.n
elif network_type == "RNN":
    from RNN_model import ACNet as Net
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    N_S = env.observation_space.shape[0]
    N_A = env.action_space.shape[0]
    A_BOUND = [env.action_space.low, env.action_space.high]

else:
    print("wrong network type!")
    exit(1)


if __name__ == "__main__":

    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(N_S, N_A, env_name, gnet, Net, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()