from torch.utils.data import Dataset
from PIL import Image
import random
import pickle

class MyDataset(Dataset):
    def __init__(self, samples, transform=None, preload_image=True):
        self.samples = samples  # [(imgpath, identity, age)]
        self.transform = transform
        self.preload_image = preload_image

        if preload_image:
            total_count = len(samples)
            total_byten = max(1, total_count // 10)
            print(f'image pre-loading... total {total_count}')
            for idx, sample in enumerate(self.samples):
                image_path = sample[0]
                image = self.load_image(image_path)
                self.samples[idx] = (image,sample[1],sample[2])
                
                if (idx+1) % total_byten == 0 or (idx+1) == total_count:
                    print(f'{idx+1}/{total_count} loaded!')
            print('image loading complete!')

        self.id_to_label = {}
        label = 0
        for sample in samples:
            id = sample[1]
            if not id in self.id_to_label:
                self.id_to_label[id] = label
                label += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.preload_image:
            image = sample[0]
        else:
            image_path = sample[0]
            image = self.load_image(image_path)

        identity = sample[1]
        label = self.id_to_label[identity]

        age = sample[2]
        if age == -1:
            period_label = -1
        else:
            # 将age映射到0-9的范围内，确保不超出num_periods=10的范围
            # 可以根据实际数据的年龄范围进行调整
            period_label = min(9, max(0, age))
        return image, label, period_label
    
    def load_image(self, path):
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def get_dataset(path, transform=None, preload_image=True, pre_dataset = False):
    if pre_dataset:
        with open('train_data.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open('test_data.pkl', 'rb') as f:
            test_data = pickle.load(f)
            
    else:
        # 读取CSV文件并分割数据集
        with open(path, 'r') as f:
            lines = f.readlines()
        lines = lines[1:]  # 跳过表头

        train_data = []
        test_data = []

        temp_data = []
        temp_shootdates = []

        def process_current_id():
            nonlocal temp_data, temp_shootdates, train_data, test_data
            if not temp_data:
                return

            # 如果数据只有一条，直接放到训练集
            if len(temp_data) < 2:
                train_data.extend(temp_data)
                temp_data = []
                temp_shootdates = []
                return

            # 提取年份信息并排序
            # 创建一个包含(数据, 年份)的列表
            data_with_years = []
            for tdata, tshootdate in zip(temp_data, temp_shootdates):
                try:
                    year = int(tshootdate[:4])
                except:
                    year = 0  # 无法解析年份时设为0
                data_with_years.append((tdata, year))
            
            # 按年份排序
            data_with_years.sort(key=lambda x: x[1])
            sorted_data = [item[0] for item in data_with_years]
            sorted_years = [item[1] for item in data_with_years]
            
            # 检查年份跨度
            min_year = min(sorted_years)
            max_year = max(sorted_years)
            year_span = max_year - min_year
            
            # 如果年份跨度不足3年，全部放入训练集
            if year_span < 3:
                train_data.extend(sorted_data)
                temp_data = []
                temp_shootdates = []
                return
            
            # 寻找合适的分割点，确保训练集和测试集的年份跨度≥3年
            split_idx = -1
            for i in range(1, len(sorted_years)):
                # 尝试在i处分割，训练集为前i个，测试集为后len-i个
                train_years = sorted_years[:i]
                test_years = sorted_years[i:]
                
                if not train_years or not test_years:
                    continue
                
                train_max_year = max(train_years)
                test_min_year = min(test_years)
                
                if test_min_year - train_max_year >= 3:
                    split_idx = i
                    break
            
            # 如果找到合适的分割点
            if split_idx != -1:
                train_data.extend(sorted_data[:split_idx])
                test_data.extend(sorted_data[split_idx:])
            else:
                # 无法找到合适的分割点，全部放入训练集
                train_data.extend(sorted_data)
            
            temp_data = []
            temp_shootdates = []

        cur_id = None
        for line in lines:
            parts = line.strip().split(',')
            path = parts[0]  # 图像路径
            id = parts[1]
            dob = parts[3]
            shootdate = parts[8]

            image_path = './data/Images/'+id+'/'+path

            if len(shootdate) >= 4 and len(dob) >= 4 and dob[:4].isdigit() and shootdate[:4].isdigit():
                age = int(shootdate[:4]) - int(dob[:4])
            else:
                age = -1 # 无年龄信息

            # 当遇到新的id时，处理之前的id的数据
            if cur_id is not None and id != cur_id:
                process_current_id()

            cur_id = id
            temp_data.append((image_path, id, age))
            temp_shootdates.append(shootdate)

        # 处理最后一个id
        process_current_id()

        with open('train_data.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        with open('test_data.pkl', 'wb') as f:
            pickle.dump(test_data, f)

    train_dataset = MyDataset(train_data, transform=transform, preload_image=preload_image)
    return train_dataset, test_data