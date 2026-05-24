from torch.utils.data import Dataset
from PIL import Image
import os

#Flickr数据读取
class FlickrDataset(Dataset):

    def __init__(self, root_dir, caption_dir): #root:根目录
        self.caption_dir = caption_dir
        self.root_dir = root_dir #data/image
        self.img_path = os.listdir(self.root_dir)

    def __getitem__(self, idx):
        image_name = self.img_path[idx]
        caption = []
        img_item_path = os.path.join(self.root_dir,image_name)
        img = Image.open(img_item_path)

        caption_file = open("../data/captions.txt",encoding="utf-8")
        caption_lines = caption_file.readlines()

        for line in range(5 * idx + 1,5 * idx + 6):
            caption_img,caption_temp = caption_lines[line].split(",") #split将一句话从','分割
            caption_temp = caption_temp.strip() # 去除’\n'
            caption.append(caption_temp) #将句子读取到caption

        return img,caption

    def __len__(self):
        return len(self.img_path)



Root_dir = "../data/Image"
Caption_dir = "../data/captions.txt"
image_dataset = FlickrDataset(Root_dir,Caption_dir)
Img,Caption = image_dataset[0]
print(Img)
print(Caption)

