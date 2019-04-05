import numpy as np
def train_val_split(data, batch_size=100):
    """[summary]
    
    Arguments:
        data {[numpy array]} -- [2 by N]
    
    Keyword Arguments:
        batch_size {int} -- [description] (default: {100})
    
    Returns:
        [type] -- [description]
    """

    if data.shape[0] >= batch_size * 2:
        partSize = int(data.shape[0]/2)
        indices = list(range(data.shape[0]))
        valid_idx = indices[:partSize]
        train_idx = indices[partSize:]
        np.random.shuffle(valid_idx)
        np.random.shuffle(train_idx)
        train_data = data[train_idx]
        valid_data = data[valid_idx]
        return train_data, valid_data
    else:
        raise ValueError



# def sample_batch_backup(data, resp, cond, batch_size=100, sample_mode='joint', randomJointIdx=True):
#     index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
#     batch_joint = data[index]
#     if randomJointIdx == True:
#         joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
#         marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
#         if data.shape[1] == 2:
#             batch_mar = np.concatenate([data[joint_index][:,0].reshape(-1,1),
#                                          data[marginal_index][:,1].reshape(-1,1)],
#                                        axis=1)
#         else:
#             batch_mar = np.concatenate([data[joint_index][:,resp].reshape(-1,1),
#                                          data[marginal_index][:,cond].reshape(-1,len(cond))],
#                                        axis=1)
#     else:
#         marginal_index = np.random.choice(range(batch_joint.shape[0]), size=batch_size, replace=False)
#         if data.shape[1] == 2:
#             batch_mar = np.concatenate([batch_joint[:,0].reshape(-1,1),
#                                          batch_joint[marginal_index][:,1].reshape(-1,1)],
#                                        axis=1)
#         else:
#             batch_mar = np.concatenate([batch_joint[:,resp].reshape(-1,1),
#                                          batch_joint[marginal_index][:,cond].reshape(-1,len(cond))],
#                                        axis=1)
#     if (type(cond)==list):
#         whole = cond.copy()
#         whole.append(resp)
#         batch_joint = batch_joint[:,whole]
#     return batch_joint, batch_mar
