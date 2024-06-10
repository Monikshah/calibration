import os
import clip
from PIL import Image
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import pickle
import calibration_model as cm
import torch.nn as nn
import torch.optim as optim


device = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_IMAGE_DIRECTORY = "/home/monika/Downloads/train2014"
COCO_DATA_DIRECTORY = "/home/monika/Documents/PhD-Project/MLN-Caption-Verification-bkup/MLN-Caption-Verification/coco-caption"

model, transform = clip.load("ViT-B/32", device=device)
state = "train"


class CocoDataset(Dataset):
    def __init__(self, image_ids, coco, transform, image_dir):
        self.image_ids = image_ids
        self.coco = coco
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(
            self.image_dir, "COCO_train2014_" + "{:012d}".format(image_id) + ".jpg"
        )
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        captions = [self.coco.loadAnns(ann_id)[0]["caption"] for ann_id in ann_ids]

        return image_id, image, captions


def collate_fn(batch):
    image_ids, images, captions = zip(*batch)
    images = torch.stack(images)
    return image_ids, images, captions


def clipEmbeddingBatch(image_ids, images, captions):
    image_embeddings = model.encode_image(images.to(device))
    combined_embeddings = []
    for i, caption_list in enumerate(captions):
        tmp = []
        for caption in caption_list:
            text = clip.tokenize([caption]).to(device)
            t_embedding = model.encode_text(text)
            tmp.append(t_embedding.cpu().detach().numpy())
        text_embedding = np.mean(np.array(tmp), axis=0)
        image_embedding = image_embeddings[i].cpu().detach().numpy()
        concatenated_embedding = np.concatenate(
            (text_embedding, image_embedding), axis=None
        )
        combined_embeddings.append(concatenated_embedding)

    return np.array(combined_embeddings)


def getCocoObject(captionorInstance, dataType):
    captionFile = os.path.join(
        COCO_DATA_DIRECTORY,
        "annotations",
        "{}_{}.json".format(captionorInstance, dataType),
    )
    coco = COCO(captionFile)
    return coco


def generatedCaptions(imageid, data):
    captions = tuple([data[str(id)] for id in imageid])

    return captions


def main():
    coco = getCocoObject("captions", state + "2014")
    imgIds = coco.getImgIds()

    dataset = CocoDataset(imgIds, coco, transform, TRAIN_IMAGE_DIRECTORY)
    dataloader = DataLoader(
        dataset, batch_size=cm.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    # READ the generated captions
    with open(
        "/home/monika/Documents/PhD-Project/meshed-memory-transformer/output_logs/m2genCaptionTrain_withimageids.pickle",
        "rb",
    ) as handle:
        data = pickle.load(handle)

    labels_0 = torch.zeros(cm.BATCH_SIZE, dtype=torch.long).to(device)
    labels_1 = torch.ones(cm.BATCH_SIZE, dtype=torch.long).to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=cm.learning_rate)
    # model = cm.MLP(cm.input_size, cm.hidden_size, cm.num_classes).to(device)
    model = cm.MLP(cm.input_size, cm.hidden_size, cm.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cm.learning_rate)

    for epoch in range(cm.num_epoch):
        model.train
        for batch_idx, (image_ids, images, captions) in enumerate(dataloader):
            combined_embeddings_GT = clipEmbeddingBatch(image_ids, images, captions)
            captions_GEN = generatedCaptions(image_ids, data)
            combined_embeddings_GEN = clipEmbeddingBatch(
                image_ids, images, captions_GEN
            )
            combined_embeddings_GT = torch.Tensor(combined_embeddings_GT).to(device)
            combined_embeddings_GEN = torch.Tensor(combined_embeddings_GEN).to(device)
            outputs = model(combined_embeddings_GT)
            loss = criterion(outputs, labels_0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{cm.num_epoch}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )

            outputs = model(combined_embeddings_GEN)
            loss = criterion(outputs, labels_1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{cm.num_epoch}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )


if __name__ == "__main__":
    main()
