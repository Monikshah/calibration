import torch
import calibration_model as cm

# import train as tr
import clip
import torch.nn.functional as F
import json
import utility as ut
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score

state = "val"
VAL_IMAGE_DIRECTORY = "/home/monika/Downloads/val2014"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = cm.MLP(cm.input_size, cm.hidden_size, cm.num_classes).to(device)
_, transform = clip.load("ViT-B/32", device=device)

# Load the model
model.load_state_dict(torch.load("output/model.pth"))
model.eval()  # Set the model to evaluation mode

coco = ut.getCocoObject("captions", state + "2014")
# imgIds = coco.getImgIds()
data_file = "./../Caption_Verification_HMLN_CVPR/captionGenTest/result_test_m2.json"
# data_file = "./../Caption_Verification_HMLN_CVPR/captionGenTest/captionGT_Test.json"
data = json.load(open(data_file))
data = ut.getCaption(data_file)
imgIds = [id for id in data]

dataset = ut.CocoDataset(imgIds, coco, VAL_IMAGE_DIRECTORY, "val")

# dataset = ut.ValTestDataset(imgIds, data, VAL_IMAGE_DIRECTORY, "val")
dataloader = DataLoader(
    dataset, batch_size=cm.BATCH_SIZE, shuffle=False, collate_fn=ut.collate_fn
)
criterion = nn.BCELoss()

# for inference
with torch.no_grad():
    # Prepare your data for inference
    for batch_idx, (image_ids, images, captions) in enumerate(dataloader):
        # combined_embeddings_GEN = tr.clipEmbeddingBatch(image_ids, images, captions_GEN)
        embeddings_GT = ut.clipEmbeddingBatch(image_ids, images, captions)
        embeddings_GT = torch.tensor(embeddings_GT, dtype=torch.float32).to(device)
        batch_size = embeddings_GT.size(0)

        captions_GEN = ut.generatedCaptions(image_ids, data)
        embeddings_GEN = ut.clipEmbeddingBatch(image_ids, images, captions_GEN)
        embeddings_GEN = torch.tensor(embeddings_GEN, dtype=torch.float32).to(device)
        combined_embeddings = torch.cat((embeddings_GT, embeddings_GEN), 0)

        # combined_embeddings = clipEmbeddingBatch(image_ids, images, captions)
        # combined_embeddings = torch.Tensor(combined_embeddings_GT).to(device)
        labels_0 = torch.zeros(batch_size, dtype=torch.float32).to(device)
        labels_1 = torch.ones(batch_size, dtype=torch.float32).to(device)
        labels = torch.cat((labels_0, labels_1), 0).view(-1, 1)

        outputs = model(combined_embeddings)
        loss = criterion(outputs, labels)
        print(roc_auc_score(labels.cpu(), outputs.cpu()))
        # print(outputs)
        print(loss.item())
        exit()
        # Process the outputs as needed
