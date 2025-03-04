import numpy as np
import pandas as pd
import os
from pathlib import Path
import time
import copy
import matplotlib.pyplot as plt
from PIL import Image
import zipfile
import shutil
from tqdm import tqdm

import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import models as torchvision_models
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights


def main():
    fix_random_seeds()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device count = {torch.cuda.device_count()}')
    print(f'Device: {device}')

    IS_KAGGLE_ENV = False
    DATA_ROOT = 'data/plates/plates'
    OUTPUT_DIR = 'Analysis/dinov2/plates'

    create_dir_structure(DATA_ROOT, OUTPUT_DIR, is_kaggle=IS_KAGGLE_ENV)

    BACKBONE = 'dinov2'  # in ('dinov2', 'resnet50')
    THRESHOLD_FOR_CLEANED = 0.3
    CROP_SIZE = 182
    BATCH_SIZE = 8
    NUM_WORKERS = 8 if IS_KAGGLE_ENV else 0

    VAL_FREQ = 1
    submission_path = f"{OUTPUT_DIR}/submission.csv"

    model_configurator = {
        'dinov2': {
            'backbone_name': 'dinov2',
            'arch': 'large',  # in ("small", "base", "large" or "giant")
            'use_n_blocks': 1,
            'use_avgpool': True,
            'init_lr': 0.0002,
            'epochs': 20,
        },
        'resnet50': {
            'backbone_name': 'resnet50',
            'weights': ResNet50_Weights.IMAGENET1K_V2,
            'feature': 'flatten',  # Specify here the layer from where to extract features
            'init_lr': 0.01,
            'epochs': 30,
        },
    }

    model_config = model_configurator[BACKBONE]

    if BACKBONE == 'dinov2':
        backbone_archs = {
            'small': 'vits14',
            'base': 'vitb14',
            'large': 'vitl14',
            'giant': 'vitg14',
        }
        backbone_arch = backbone_archs[model_config['arch']]
        backbone_name = f"dinov2_{backbone_arch}"
        backbone_model = torch.hub.load(repo_or_dir='facebookresearch/dinov2', model=backbone_name)
        feature_model = ModelWithIntermediateLayers(backbone_model, n_last_blocks=model_config['use_n_blocks'])
        embed_dim = backbone_model.embed_dim * (model_config['use_n_blocks'] + int(model_config['use_avgpool']))

    else:
        backbone_model = torchvision_models.get_model(model_config["backbone_name"], weights=model_config["weights"])
        feature_model = ModelWithIntermediateLayers(backbone_model, model_config["feature"])
        embed_dim = feature_model.embed_dim

    feature_model = feature_model.to(device)

    linear_classifier = LinearClassifier(embed_dim, model_config, num_classes=2)
    linear_classifier = linear_classifier.to(device)

    ## data load
    data_transforms = {
        "train": v2.Compose([
            v2.PILToTensor(),
            v2.CenterCrop(CROP_SIZE),
            v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        ]),
        "val": v2.Compose([
            v2.PILToTensor(),
            v2.CenterCrop(CROP_SIZE),
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        ]),
        "test": v2.Compose([
            v2.PILToTensor(),
            v2.CenterCrop(CROP_SIZE),
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_ROOT, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True,
                                                  num_workers=NUM_WORKERS)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f"Data loaded with {dataset_sizes['train']} train and {dataset_sizes['val']} val images.")

    image_datasets["test"] = CustomDataset(Path(os.path.join(DATA_ROOT, "test")), data_transforms["test"])
    dataloaders["test"] = torch.utils.data.DataLoader(
        image_datasets["test"], batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    dataset_sizes["test"] = len(image_datasets["test"])
    print(f"Data loaded with {dataset_sizes['test']} test images.")

    # Visualization examples
    inputs, classes = next(iter(dataloaders['train']))
    show_batch(inputs, titles=[class_names[x] for x in classes])

    inputs, classes = next(iter(dataloaders['val']))
    show_batch(inputs, titles=[class_names[x] for x in classes])

    # loss funcion, optimizer, scheduler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(linear_classifier.parameters(),
                                lr=model_config['init_lr'],
                                momentum=0.9,
                                weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, model_config['epochs'], eta_min=0
    )

    run_train_loop_for_images(model_config, dataloaders, feature_model, linear_classifier,
                              loss_fn, optimizer, device, scheduler, OUTPUT_DIR, VAL_FREQ)

    visualize_result(OUTPUT_DIR, model_config, embed_dim, feature_model,
                     device=device, dataloaders=dataloaders)

    test_predict(feature_model, embed_dim, model_config, dataloaders, OUTPUT_DIR, THRESHOLD_FOR_CLEANED,
                 submission_path, device)


def main2():
    fix_random_seeds()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device count = {torch.cuda.device_count()}')
    print(f'Device: {device}')

    IS_KAGGLE_ENV = False
    DATA_ROOT = 'data/plates/plates'
    OUTPUT_DIR = 'Analysis/dinov2/plates_g'

    create_dir_structure(DATA_ROOT, OUTPUT_DIR, is_kaggle=IS_KAGGLE_ENV)

    BACKBONE = 'dinov2'  # in ('dinov2', 'resnet50')
    THRESHOLD_FOR_CLEANED = 0.3
    CROP_SIZE = 182
    BATCH_SIZE = 8
    NUM_WORKERS = 8 if IS_KAGGLE_ENV else 0

    VAL_FREQ = 1
    submission_path = f"{OUTPUT_DIR}/submission.csv"

    model_configurator = {
        'dinov2': {
            'backbone_name': 'dinov2',
            'arch': 'giant',  # in ("small", "base", "large" or "giant")
            'use_n_blocks': 1,
            'use_avgpool': True,
            'init_lr': 0.0002,
            'epochs': 20,
        },
        'resnet50': {
            'backbone_name': 'resnet50',
            'weights': ResNet50_Weights.IMAGENET1K_V2,
            'feature': 'flatten',  # Specify here the layer from where to extract features
            'init_lr': 0.01,
            'epochs': 30,
        },
    }

    model_config = model_configurator[BACKBONE]

    if BACKBONE == 'dinov2':
        backbone_archs = {
            'small': 'vits14',
            'base': 'vitb14',
            'large': 'vitl14',
            'giant': 'vitg14',
        }
        backbone_arch = backbone_archs[model_config['arch']]
        backbone_name = f"dinov2_{backbone_arch}"
        backbone_model = torch.hub.load(repo_or_dir='facebookresearch/dinov2', model=backbone_name)
        feature_model = ModelWithIntermediateLayers(backbone_model, n_last_blocks=model_config['use_n_blocks'])
        embed_dim = backbone_model.embed_dim * (model_config['use_n_blocks'] + int(model_config['use_avgpool']))

    else:
        backbone_model = torchvision_models.get_model(model_config["backbone_name"], weights=model_config["weights"])
        feature_model = ModelWithIntermediateLayers(backbone_model, model_config["feature"])
        embed_dim = feature_model.embed_dim

    feature_model = feature_model.to(device)

    linear_classifier = LinearClassifier(embed_dim, model_config, num_classes=2)
    linear_classifier = linear_classifier.to(device)

    ## data load
    data_transforms = {
        "train": v2.Compose([
            v2.PILToTensor(),
            v2.CenterCrop(CROP_SIZE),
            v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        ]),
        "val": v2.Compose([
            v2.PILToTensor(),
            v2.CenterCrop(CROP_SIZE),
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        ]),
        "test": v2.Compose([
            v2.PILToTensor(),
            v2.CenterCrop(CROP_SIZE),
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_ROOT, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True,
                                                  num_workers=NUM_WORKERS)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f"Data loaded with {dataset_sizes['train']} train and {dataset_sizes['val']} val images.")

    image_datasets["test"] = CustomDataset(Path(os.path.join(DATA_ROOT, "test")), data_transforms["test"])
    dataloaders["test"] = torch.utils.data.DataLoader(
        image_datasets["test"], batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    dataset_sizes["test"] = len(image_datasets["test"])
    print(f"Data loaded with {dataset_sizes['test']} test images.")

    # Visualization examples
    inputs, classes = next(iter(dataloaders['train']))
    show_batch(inputs, titles=[class_names[x] for x in classes])

    inputs, classes = next(iter(dataloaders['val']))
    show_batch(inputs, titles=[class_names[x] for x in classes])

    # loss funcion, optimizer, scheduler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(linear_classifier.parameters(),
                                lr=model_config['init_lr'],
                                momentum=0.9,
                                weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, model_config['epochs'], eta_min=0
    )

    run_train_loop_for_images(model_config, dataloaders, feature_model, linear_classifier,
                              loss_fn, optimizer, device, scheduler, OUTPUT_DIR, VAL_FREQ)

    visualize_result(OUTPUT_DIR, model_config, embed_dim, feature_model,
                     device=device, dataloaders=dataloaders)

    test_predict(feature_model, embed_dim, model_config, dataloaders, OUTPUT_DIR, THRESHOLD_FOR_CLEANED,
                 submission_path, device)


def visualize_result(OUTPUT_DIR, model_config, embed_dim, feature_model, device, dataloaders):
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'dino_classifier_ckpt.pth'))
    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    best_loss = checkpoint['best_loss']
    print(f'Best result (validation): epoch:{epoch}, accuracy{best_acc}, cross entropy: {best_loss}')

    model_inf = LinearClassifier(embed_dim, model_config)
    model_inf = model_inf.to(device)
    model_inf.load_state_dict(checkpoint['state_dict'])
    model_inf.eval()

    dl_iter = iter(dataloaders["test"])
    for i, _ in enumerate(range(3)):
        imgs, _ = next(dl_iter)
        output_path = f"{OUTPUT_DIR}/test_{i}.png"
        visualize_model(feature_model, model_inf, imgs, device, output_path=output_path)


def fix_random_seeds(seed=12345):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def create_dir_structure(DATA_ROOT, OUTPUT_DIR, is_kaggle):
    if is_kaggle:
        if not os.path.exists(DATA_ROOT):
            # Input data files are available in the "/kaggle/input/" directory.
            with zipfile.ZipFile('/kaggle/input/platesv2/plates.zip', 'r') as zip_obj:
                # Extract all the contents of zip file in current directory
                zip_obj.extractall()

            indexes = np.random.choice(20, size=4, replace=False)
            file_names = [f'{idx:04}.jpg' for idx in indexes]
            print('Moving files from train to val dir:', file_names)

            val_path = os.path.join(DATA_ROOT, 'val')
            os.makedirs(val_path)
            train_path = os.path.join(DATA_ROOT, 'train')
            class_names = ['cleaned', 'dirty']
            for class_name in class_names:
                dest_val_dir = os.path.join(val_path, class_name)
                src_train_dir = os.path.join(train_path, class_name)
                os.makedirs(dest_val_dir)
                for file_name in file_names:
                    shutil.move(os.path.join(src_train_dir, file_name), dest_val_dir)

    os.makedirs(OUTPUT_DIR, exist_ok=True)


# creating models for feature extraction
# class for dinov2
class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks

    def forward(self, images):
        with torch.inference_mode():
            features = self.feature_model.get_intermediate_layers(
                images, self.n_last_blocks, return_class_token=True
            )
        return features


# class for other models
class ModelWithFeaterExtractor(nn.Module):
    def __init__(self, feature_model, feature_layer):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.feature_extractor = create_feature_extractor(
            self.feature_model, return_nodes={feature_layer: 'out'}
        )
        inp = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            outp = self.feature_extractor(inp)
        self.embed_dim = outp["out"].size(-1)

    def forward(self, images):
        with torch.inference_mode():
            features = self.feature_model(images)
        return features


# creating a linear classifier
def create_linear_input(x_tokeens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokeens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)

    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),
            ),
            dim=-1
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, config, num_classes=2):
        super().__init__()
        self.out_dim = out_dim
        if config["backbone_name"] == 'dinov2':
            self.use_dinov2 = True
            self.use_n_blocks = config["use_n_blocks"]
            self.use_avgpool = config["use_avgpool"]
        else:
            self.use_dinov2 = False
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, input):
        if self.use_dinov2:
            return self.linear(create_linear_input(input, self.use_n_blocks, self.use_avgpool))
        else:
            return self.linear(input)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, tfms: v2.Compose):
        self.path = path
        self.tfms = tfms
        self.filenames = list(path.glob('*.jpg'))

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = read_image(str(filename))
        img = self.tfms(img)
        return img, filename.stem

    def __len__(self):
        return len(self.filenames)


def show_batch(imgs, titles=None, rows=2, cols=4):
    if titles is None:
        titles = ['image ' + str(i + 1) for i in range(imgs.size(0))]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    fig = plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(imgs.size(0)):
        img = imgs[i].cpu().numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.set_title(titles[i])
        ax.axis('off')
    fig.tight_layout()


def train_loop(dataloader, feature_model, linear_classifier, loss_fn, optimizer, device):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    linear_classifier.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    running_loss = 0.0
    running_corrects = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        features = feature_model(X)
        # Compute prediction and loss
        pred = linear_classifier(features)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Statistics
        running_loss += loss.item()
        running_corrects += (pred.argmax(1) == y).type(torch.float).sum().item()

    epoch_loss = running_loss / num_batches
    epoch_acc = running_corrects / size
    return epoch_acc, epoch_loss


def run_train_loop_for_images(model_config, dataloaders, feature_model, linear_classifier,
                              loss_fn, optimizer, device, scheduler, OUTPUT_DIR, VAL_FREQ):
    best_acc = 0.
    best_acc_loss = np.inf
    train_data = []
    for t in range(model_config['epochs']):
        print(f'Epoch {t + 1}/{model_config["epochs"]}')
        print('-' * 10)

        train_acc, train_loss = train_loop(dataloaders['train'], feature_model=feature_model,
                                           linear_classifier=linear_classifier, loss_fn=loss_fn,
                                           optimizer=optimizer, device=device)
        train_data.append({
            "phase": "train",
            "epoch": t,
            "lr": optimizer.param_groups[0]['lr'],
            "accuracy": train_acc,
            "loss": train_loss
        })
        scheduler.step()
        print(f'Train:\n  train_acc = {train_acc}, train_loss = {train_loss}')

        if (t % VAL_FREQ == 0) or (t == model_config['epochs'] - 1):
            val_acc, val_loss = val_loop(dataloaders['val'], feature_model, linear_classifier, loss_fn, device)
            train_data.append({
                "phase": "val",
                "epoch": t,
                "lr": optimizer.param_groups[0]['lr'],
                "accuracy": val_acc,
                "loss": val_loss
            })

            print(f'Val:\n  val_acc = {val_acc}, val_loss = {val_loss}')

            if ((val_acc == best_acc) and (val_loss < best_acc_loss)) or (val_acc > best_acc):
                best_acc, best_acc_loss = val_acc, val_loss
                print(f"Best val_acc = {best_acc}, Best val_loss = {best_acc_loss}")

                save_dict = {
                    "epoch": t + 1,
                    "state_dict": linear_classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "best_loss": best_acc_loss
                }
                torch.save(save_dict, os.path.join(OUTPUT_DIR, "dino_classifier_ckpt.pth"))

        print('\n')
    print('Training completed.')


@torch.inference_mode()
def val_loop(dataloader, feature_model, linear_classifier, loss_fn, device):
    linear_classifier.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, val_acc = 0.0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            features = feature_model(X)
            pred = linear_classifier(features)
            val_loss += loss_fn(pred, y).item()
            val_acc += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    val_acc /= size
    return val_acc, val_loss


def visualize_model(feature_model, linear_classifier, images, device, output_path, rows=2, cols=4):
    was_training = linear_classifier.training
    linear_classifier.eval()

    with torch.no_grad():
        imgs = images.to(device)
        features = feature_model(imgs)
        outputs = linear_classifier(features)
        outputs = nn.functional.softmax(outputs, dim=1)
        # prediction_score, pred_label_idx = torch.topk(outputs, 1)
        _, preds = torch.max(outputs, 1)

        titles = [f"Clean:{outputs[i, 0].squeeze().item():.3f}\nDirty:{outputs[i, 1].squeeze().item():.3f}"
                  for i in range(imgs.size(0))]
        show_batch(imgs, titles=titles, rows=rows, cols=cols)
        plt.savefig(output_path)
        plt.close()

    linear_classifier.train(mode=was_training)


def test_predict(feature_model, embed_dim, model_config, dataloaders, OUTPUT_DIR, THRESHOLD_FOR_CLEANED,
                 submission_path, device):
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'dino_classifier_ckpt.pth'))
    model_inf = LinearClassifier(embed_dim, model_config)
    model_inf = model_inf.to(device)
    model_inf.load_state_dict(checkpoint['state_dict'])
    model_inf.eval()

    test_predictions = []
    test_img_paths = []
    for inputs, paths in tqdm(dataloaders['test']):
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            features = feature_model(inputs)
            preds = model_inf(features)
        test_predictions.append(
            nn.functional.softmax(preds, dim=1)[:, 0].data.cpu().numpy())
        test_img_paths.extend(paths)

    test_predictions = np.concatenate(test_predictions)

    submission_df = pd.DataFrame.from_dict({'id': test_img_paths, 'label': test_predictions})
    submission_df['label'] = submission_df['label'].map(
        lambda pred: 'cleaned' if pred >= THRESHOLD_FOR_CLEANED else 'dirty')
    submission_df['id'] = submission_df['id'].str.replace('/kaggle/working/test/unknown/', '')
    submission_df['id'] = submission_df['id'].str.replace('.jpg', '')
    submission_df.set_index('id', inplace=True)
    submission_df.to_csv(submission_path)


if __name__ == '__main__':
    # main()
    main2()
