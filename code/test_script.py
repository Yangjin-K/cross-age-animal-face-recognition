print("Testing script execution")
print("Hello from test script")

# 测试加载数据集
import os
def load_fgnet_dataset():
    """加载FGNET数据集"""
    data_dir = "../HumanfaceDataset/FGNET/images"
    data_list = []
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.lower().endswith('.jpg'):
                try:
                    # 解析文件名：001A11.JPG或001A11.jpg
                    id_str = filename[:3]
                    age = int(filename[4:6])
                    img_path = os.path.join(data_dir, filename)
                    data_list.append((img_path, id_str, age))
                except:
                    continue
    
    print(f"Loaded {len(data_list)} images from FGNET")
    return data_list

# 测试加载数据集
data_list = load_fgnet_dataset()
print(f"Data list length: {len(data_list)}")
if data_list:
    print(f"First item: {data_list[0]}")
else:
    print("No data found")