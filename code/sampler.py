import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import Sampler

class IdentityPeriodSampler(Sampler):
    """
    PK Sampler 变体：确保一个 Batch 中挑选出 P 个人的身份，
    然后每个人挑选 K 张照片（K 张照片由于存在同一个用户的多个时期记录，天然有更大概率包含不同年龄段）。
    使得 Contrastive Loss 能稳定计算。
    """
    def __init__(self, dataset, batch_size, num_instances=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        
        self.index_dic = defaultdict(list)
        for index, sample in enumerate(self.dataset.samples):
            pid = sample[1]  # sample[1] is identity
            self.index_dic[pid].append(index)
            
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.dataset.samples)
        
    def __len__(self):
        return self.num_samples // self.batch_size
        
    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = self.index_dic[pid].copy()
            np.random.shuffle(idxs)
            batch_idxs_dict[pid] = idxs
            
        avai_pids = self.pids.copy()
        
        for _ in range(self.__len__()):
            batch = []
            if len(avai_pids) < self.num_pids_per_batch:
                avai_pids = self.pids.copy()
                
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                idxs = batch_idxs_dict[pid]
                if len(idxs) >= self.num_instances:
                    batch.extend(idxs[:self.num_instances])
                    del idxs[:self.num_instances]
                    if len(idxs) < self.num_instances:
                        avai_pids.remove(pid)
                else:
                    # 样本不足时进行放回重采样补齐，避免 batch size 被改变
                    batch.extend(np.random.choice(self.index_dic[pid], self.num_instances, replace=True).tolist())
                    if pid in avai_pids:
                        avai_pids.remove(pid)
            yield batch
