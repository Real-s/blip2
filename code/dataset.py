from torchvision import transforms
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from PIL import Image
import os

#Flickr数据读取
class FlickrDataset(Dataset):

    def __init__(self, root_dir, caption_dir , trans = False , resize = False): #root:根目录
        self.caption_dir = caption_dir
        self.trans = trans
        self.root_dir = root_dir #data/image
        self.img_path = os.listdir(self.root_dir)
        self.trans_tensor = transforms.ToTensor()
        self.trans_Resize = transforms.Resize((256,256))
        self.resize = resize
        with open(self.caption_dir, encoding="utf-8") as f:
            self.caption_lines = f.readlines()

    def __getitem__(self, idx):
        image_name = self.img_path[idx]
        caption = []
        img_item_path = os.path.join(self.root_dir,image_name)
        img = Image.open(img_item_path)

        if self.trans:
            img = self.trans_tensor(img)
        #统一尺寸可存入Dataloader
        if self.resize:
            img = self.trans_Resize(img)

        for line in range(5 * idx + 1,5 * idx + 6):
            caption_img,caption_temp = self.caption_lines[line].split(",",1) #split将一句话从','分割
            caption_temp = caption_temp.strip() # 去除’\n'
            caption.append(caption_temp) #将句子读取到caption

        return img,caption

    def __len__(self):
        return len(self.img_path)

#
#
# Root_dir = "../data/Image"
# Caption_dir = "../data/captions.txt"
# writer = SummaryWriter("../test/logs")
# image_dataset = FlickrDataset(Root_dir,Caption_dir , True)
# Img,Caption = image_dataset[0]
#
# for i in range(0,10):
#     Img, Caption = image_dataset[i]
#     writer.add_image("Img_ToTensor",Img,i)
#     print(Img)
#     print(Caption)
# writer.close()
# print(Img)
# print(Caption)

