import json
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

from config import Config


class LLaVAPretrainDataset(Dataset):

    def __init__(self, config: Config, path: str, transform: transforms.Compose = None):
        super().__init__()
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(config.img_size, interpolation = transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(config.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            self.transform = transform
        self.text = []
        with open(f'{path}/blip_laion_cc_sbu_558k.json', 'r', encoding = 'utf-8') as f:
            for v in json.load(f):
                self.text.append({
                    'img_path': f'{path}/images/{v["image"]}',
                    v['conversations'][0]['from']: v['conversations'][0]['value'],
                    v['conversations'][1]['from']: v['conversations'][1]['value']
                })

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        text = self.text[i]
        img = Image.open(text['img_path']).convert('RGB')
        return self.transform(img), text['gpt']


if __name__ == "__main__":
    config = Config()
    dataset = LLaVAPretrainDataset(config, "E:/data-hub/LLaVA-Pretrain")
    # print(len(dataset))
    dataset[22]