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
import matplotlib.pyplot as plt
import utility as ut
from sklearn.calibration import calibration_curve, CalibrationDisplay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_IMAGE_DIRECTORY = "/home/monika/Downloads/train2014"
VAL_IMAGE_DIRECTORY = "/home/monika/Downloads/val2014"
COCO_DATA_DIRECTORY = "/home/monika/Documents/PhD-Project/MLN-Caption-Verification-bkup/MLN-Caption-Verification/coco-caption"
model, transform = clip.load("ViT-B/32", device=device)
state = "train"


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


def main():
    coco = getCocoObject("captions", state + "2014")
    imgIds = coco.getImgIds()
    train_dataset = ut.CocoDataset(imgIds, coco, TRAIN_IMAGE_DIRECTORY, "train")

    dataloader = DataLoader(
        train_dataset,
        batch_size=cm.BATCH_SIZE,
        num_workers=4,
        shuffle=True,
        collate_fn=ut.collate_fn,
    )

    val_data_file_GEN = (
        "./../Caption_Verification_HMLN_CVPR/captionGenTest/m2genCaptionVal.json"
    )
    val_data_GEN = ut.getCaption(val_data_file_GEN)
    valimgIds_GEN = [id for id in val_data_GEN][:2500]
    coco = getCocoObject("captions", "val" + "2014")
    # imgIds = coco.getImgIds()[:2500]
    val_dataset = ut.CocoDataset(valimgIds_GEN, coco, VAL_IMAGE_DIRECTORY, "val")

    # val_dataset = ut.ValTestDataset(valimgIds, val_data, VAL_IMAGE_DIRECTORY, "val")
    val_dataloader = DataLoader(
        val_dataset, batch_size=cm.BATCH_SIZE, shuffle=False, collate_fn=ut.collate_fn
    )

    # READ the TRAIN SET generated captions
    with open(
        "/home/monika/Documents/PhD-Project/MESHED-MEMORY-TRANSFORMER/output_logs/m2genCaptionTrain_withimageids.pickle",
        "rb",
    ) as handle:
        traindata = pickle.load(handle)

    criterion = nn.BCELoss()  # nn.CrossEntropyLoss()
    _model = cm.MLP(cm.input_size, cm.hidden_size, cm.num_classes).to(device)
    optimizer = optim.Adam(_model.parameters(), lr=cm.learning_rate)

    train_losses = []
    val_losses = []
    for epoch in range(cm.num_epoch):
        _model.train()
        epoch_loss = 0
        for batch_idx, (image_ids, images, captions) in enumerate(dataloader):
            embeddings_GT = ut.clipEmbeddingBatch(image_ids, images, captions)
            embeddings_GT = torch.tensor(embeddings_GT, dtype=torch.float32).to(device)
            image_ids_ = [str(id) for id in image_ids]
            captions_GEN = generatedCaptions(image_ids_, traindata)
            embeddings_GEN = ut.clipEmbeddingBatch(image_ids, images, captions_GEN)
            embeddings_GEN = torch.tensor(embeddings_GEN, dtype=torch.float32).to(
                device
            )
            combined_embeddings = torch.cat((embeddings_GT, embeddings_GEN), 0)
            batch_size = embeddings_GT.size(0)
            labels_0 = torch.zeros(batch_size, dtype=torch.float32).to(device)
            labels_1 = torch.ones(batch_size, dtype=torch.float32).to(device)
            labels = torch.cat((labels_0, labels_1), 0).view(-1, 1)

            outputs = _model(combined_embeddings)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{cm.num_epoch}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )

        avg_epoch_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_epoch_loss)
        print(
            f"Epoch [{epoch+1}/{cm.num_epoch}], Average Training Loss: {avg_epoch_loss:.4f}"
        )

        _model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            start = 0
            for batch_idx, (image_ids, images, captions) in enumerate(val_dataloader):
                valembeddings_GT = ut.clipEmbeddingBatch(image_ids, images, captions)
                valembeddings_GT = torch.tensor(
                    valembeddings_GT, dtype=torch.float32
                ).to(device)
                # valimgIds_GEN = valimgIds_GEN[start : start + cm.BATCH_SIZE]
                start += cm.BATCH_SIZE
                captions_GEN = generatedCaptions(image_ids, val_data_GEN)
                embeddings_GEN = ut.clipEmbeddingBatch(
                    image_ids,
                    images,
                    captions_GEN,
                )
                embeddings_GEN = torch.tensor(embeddings_GEN, dtype=torch.float32).to(
                    device
                )
                combined_embeddings = torch.cat((valembeddings_GT, embeddings_GEN), 0)
                batch_size = valembeddings_GT.size(0)
                labels_0 = torch.zeros(batch_size, dtype=torch.float32).to(device)
                labels_1 = torch.ones(batch_size, dtype=torch.float32).to(device)
                labels = torch.cat((labels_0, labels_1), 0).view(-1, 1)
                outputs = _model(combined_embeddings)
                loss = criterion(outputs, labels)
                val_epoch_loss += loss.item()

        avg_val_loss = val_epoch_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        print(
            f"Epoch [{epoch+1}/{cm.num_epoch}], Average Validation Loss: {avg_val_loss:.4f}"
        )

    torch.save(_model.state_dict(), "data/model.pth")
    print("Model saved to data/model.pth")

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.show()


def Calibration():
    val_data_file_GEN = (
        "./../Caption_Verification_HMLN_CVPR/captionGenTest/m2genCaptionVal.json"
    )
    val_data_GEN = ut.getCaption(val_data_file_GEN)
    valimgIds_GEN = [id for id in val_data_GEN][:2500]
    coco = getCocoObject("captions", "val" + "2014")
    # imgIds = coco.getImgIds()[:2500]
    val_dataset = ut.CocoDataset(valimgIds_GEN, coco, VAL_IMAGE_DIRECTORY, "val")

    # val_dataset = ut.ValTestDataset(valimgIds, val_data, VAL_IMAGE_DIRECTORY, "val")
    val_dataloader = DataLoader(
        val_dataset, batch_size=cm.BATCH_SIZE, shuffle=False, collate_fn=ut.collate_fn
    )
    _model = cm.MLP(cm.input_size, cm.hidden_size, cm.num_classes).to(device)
    _model.load_state_dict(torch.load("data/model.pth"))
    _model.eval()
    with torch.no_grad():
        start = 0
        for batch_idx, (image_ids, images, captions) in enumerate(val_dataloader):
            valembeddings_GT = ut.clipEmbeddingBatch(image_ids, images, captions)
            valembeddings_GT = torch.tensor(valembeddings_GT, dtype=torch.float32).to(
                device
            )
            # valimgIds_GEN = valimgIds_GEN[start : start + cm.BATCH_SIZE]
            start += cm.BATCH_SIZE
            captions_GEN = generatedCaptions(image_ids, val_data_GEN)
            embeddings_GEN = ut.clipEmbeddingBatch(
                image_ids,
                images,
                captions_GEN,
            )
            embeddings_GEN = torch.tensor(embeddings_GEN, dtype=torch.float32).to(
                device
            )
            combined_embeddings = torch.cat((valembeddings_GT, embeddings_GEN), 0)
            batch_size = valembeddings_GT.size(0)
            labels_0 = torch.zeros(batch_size, dtype=torch.float32).to(device)
            labels_1 = torch.ones(batch_size, dtype=torch.float32).to(device)
            labels = torch.cat((labels_0, labels_1), 0).view(-1, 1)
            outputs = _model(combined_embeddings)

            # prob_true, prob_pred = calibration_curve(
            # labels.cpu(), outputs.cpu(), n_bins=10
            # )
            disp = CalibrationDisplay.from_predictions(labels.cpu(), outputs.cpu())
            plt.show()

            # break


Calibration()

# if __name__ == "__main__":
# main()
