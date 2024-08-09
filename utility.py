import clip
from torch.utils.data import Dataset, DataLoader
import torch
import os
from PIL import Image
import numpy as np
import json
import random
from pycocotools.coco import COCO


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, transform = clip.load("ViT-B/32", device=device)
COCO_DATA_DIRECTORY = "/home/monika/Documents/PhD-Project/MLN-Caption-Verification-bkup/MLN-Caption-Verification/coco-caption"


class CocoDataset(Dataset):
    def __init__(self, image_ids, coco, image_dir, state):
        self.image_ids = image_ids
        self.coco = coco
        self.transform = transform
        self.image_dir = image_dir
        self.state = state

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(
            self.image_dir,
            "COCO_" + self.state + "2014_" + "{:012d}".format(image_id) + ".jpg",
        )
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        captions = [self.coco.loadAnns(random.choice(ann_ids))[0]["caption"]]

        return image_id, image, captions


class ValTestDataset(Dataset):
    def __init__(self, image_ids, captions, image_dir, state):
        self.image_ids = image_ids
        self.transform = transform
        self.image_dir = image_dir
        self.state = state
        self.captions = captions

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(
            self.image_dir,
            "COCO_" + self.state + "2014_" + "{:012d}".format(int(image_id)) + ".jpg",
        )
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        caption = self.captions[image_id]

        return image_id, image, caption


def collate_fn(batch):
    image_ids, images, captions = zip(*batch)
    images = torch.stack(images)
    return image_ids, images, captions


def clipEmbeddingBatch(image_ids, images, captions):
    image_embeddings = model.encode_image(images.to(device))
    combined_embeddings = []
    for i, caption in enumerate(captions):
        # tmp = []
        # for caption in caption_list:
        text = clip.tokenize([caption[0]]).to(device)
        t_embedding = model.encode_text(text)
        # tmp.append(t_embedding.cpu().detach().numpy())
        text_embedding = t_embedding.cpu().detach().numpy()
        # text_embedding = np.mean(np.array(t_embedding), axis=0)
        image_embedding = image_embeddings[i].cpu().detach().numpy()
        concatenated_embedding = np.concatenate(
            (text_embedding, image_embedding), axis=None
        )
        combined_embeddings.append(concatenated_embedding)
    return np.array(combined_embeddings)


def getCaption(captionFile):
    with open(captionFile, "r") as f1:
        cap_data = f1.read()
        capTest = json.loads(cap_data)
    dict_ = {}
    for i in capTest:
        dict_[i["image_id"]] = i["caption"]

    return dict_


def getCocoObject(captionorInstance, dataType):
    captionFile = os.path.join(
        COCO_DATA_DIRECTORY,
        "annotations",
        "{}_{}.json".format(captionorInstance, dataType),
    )
    coco = COCO(captionFile)
    return coco


def generatedCaptions(imageid, data):
    captions = tuple([data[id] for id in imageid])  # str(id)
    return captions
