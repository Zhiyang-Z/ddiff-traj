import osmnx as ox
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import shared_memory
import copy

def fetch_neighbors(road_net, node):
    """ NOTICE: returned list include node itself. """
    neighbors = [node]
    for edge in road_net.edges():
        if node in edge: neighbors.append(edge[0] if node != edge[0] else edge[1])
    return neighbors

# class Node:
#     def __init__(self, road_net, road_net_traj, pos, node_num):
#         self.pos, self.node_num = pos, node_num
#         self.children = []
#         if road_net_traj[self.pos] == road_net_traj[self.pos+1]:
#             self.children.append(Node(road_net, road_net_traj, self.pos+1, self.node_num))
#         else:
#             G_neighbors = road_net.subgraph(fetch_neighbors(road_net, road_net_traj[j - 1]))
#             assert len(G_neighbors) > 1
#             for neighbor in G_neighbors:
#
#
#
#
# def gen_road_net_traj(road_net_trajs_np, road_net, trajs, i):
#     traj = trajs[i]
#     road_net_traj = ox.distance.nearest_nodes(road_net, X=traj[:, 0], Y=traj[:, 1])
def cal_road_net_traj(trajs, road_net, dict, i, shared_name, shared_size):
    traj = trajs[i]
    road_net_traj_forward = ox.distance.nearest_nodes(road_net, X=traj[:, 0], Y=traj[:, 1])
    road_net_traj_backward = copy.deepcopy(road_net_traj_forward)
    for j in range(1, road_net_traj_forward.shape[0]):
        if road_net_traj_forward[j] != road_net_traj_forward[j - 1]:
            # if road_net_traj.shape[0]-1 == j: print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            G_neighbors = road_net.subgraph(fetch_neighbors(road_net, road_net_traj_forward[j - 1]))
            assert len(G_neighbors) > 1
            road_net_traj_forward[j] = ox.distance.nearest_nodes(G_neighbors, X=traj[j, 0], Y=traj[j, 1])
    for j in reversed(range(0, road_net_traj_backward.shape[0] - 1)):
        if road_net_traj_backward[j] != road_net_traj_backward[j + 1]:
            G_neighbors = road_net.subgraph(fetch_neighbors(road_net, road_net_traj_backward[j + 1]))
            assert len(G_neighbors) > 1
            road_net_traj_backward[j] = ox.distance.nearest_nodes(G_neighbors, X=traj[j, 0], Y=traj[j, 1])
    road_net_traj = road_net_traj_forward if np.unique(road_net_traj_forward).shape >= np.unique(
        road_net_traj_backward).shape else road_net_traj_backward
    for j in range(road_net_traj.shape[0]):
        road_net_index = np.where(dict == road_net_traj[j])[0]
        assert road_net_index.shape == (1,)
        road_net_traj[j] = road_net_index[0]
    shm = shared_memory.SharedMemory(name=shared_name)
    shared_road_net_trajs_np = np.ndarray(shared_size, dtype=np.int64, buffer=shm.buf)
    shared_road_net_trajs_np[i,:] = road_net_traj
    shm.close()


def GPS_to_Road_Net(src_path: str, src_label_path: str, road_net_path: str, dict_path: str, ex_path: str):
    print('start: ', ex_path)
    trajs, trajs_label, road_net, dict = np.load(src_path), np.load(src_label_path, allow_pickle=True), ox.io.load_graphml(road_net_path), np.load(dict_path)
    assert trajs.shape[0] == trajs_label.shape[0]
    road_net_trajs_np = np.zeros((trajs.shape[0], trajs.shape[1]), dtype=np.int64)
    shm = shared_memory.SharedMemory(create=True, size=road_net_trajs_np.nbytes)
    shared_road_net_trajs_np = np.ndarray(road_net_trajs_np.shape, dtype=road_net_trajs_np.dtype, buffer=shm.buf)
    np.copyto(shared_road_net_trajs_np, road_net_trajs_np)
    process_list = [multiprocessing.Process(target=cal_road_net_traj, args=(trajs, road_net, dict, i, shm.name, road_net_trajs_np.shape))
                    for i in range(trajs.shape[0])]
    P = 32
    batch, remain = int(len(process_list)/P), len(process_list) - P * int(len(process_list)/P)
    for i in tqdm(range(batch)):
        for j in range(P): process_list[i * P + j].start()
        for j in range(P): process_list[i * P + j].join()
    for i in tqdm(range(remain)): process_list[P * int(len(process_list)/P) + i].start()
    for i in tqdm(range(remain)): process_list[P * int(len(process_list) / P) + i].join()
    np.save(ex_path, shared_road_net_trajs_np)
    # np.save(ex_path + '_label', road_net_trajs_label_np)
    print('done: ', ex_path)
    shm.close()
    shm.unlink()

if __name__ == '__main__':
    prefix = '/home/zzhang18/datasets/DIDI/chengdu/extract_from_csv/'
    road_net_path, dict_path = '/home/zzhang18/datasets/DIDI/chengdu/chengdu_rode_net.graphml', '/home/zzhang18/datasets/DIDI/chengdu/chengdu_rode_net_dict.npy'
    threads_list = []
    for i in range(30):
        src_path, src_label_path, ex_path = None, None, None
        day = i+1
        if day < 10:
            src_path, src_label_path, ex_path = prefix + 'gps_2016110' + str(day) + '.npy', prefix + 'gps_2016110' + str(day) + '_label.npy', prefix + 'road_net_trajs/gps_2016110' + str(day)
        else:
            src_path, src_label_path, ex_path = prefix + 'gps_201611' + str(
                day) + '.npy', prefix + 'gps_201611' + str(
                day) + '_label.npy', prefix + 'road_net_trajs/gps_201611' + str(day)
        threads_list.append(multiprocessing.Process(target=GPS_to_Road_Net, args=(src_path, src_label_path, road_net_path, dict_path, ex_path)))
    # start threads
    parallelism = 1
    for i in range(int(30 / parallelism)):
        for j in range(parallelism):
            threads_list[i * parallelism + j].start()
        for j in range(parallelism):
            threads_list[i * parallelism + j].join()

