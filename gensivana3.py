# GenSivana — Real or Fake — Level 3 : Localization
# Pipeline K-Fold Cross-Validation (classifieur EfficientNet-B0/ResNet18 + SegFormer-B0)
#
!pip install -q transformers accelerate opencv-python-headless scikit-learn

import os
import gc
import cv2
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import SegformerForSemanticSegmentation
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score
from PIL import Image


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/kaggle/input/competitions/gensivana-real-or-fake-level-3-localization"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "images")
VAL_IMG_DIR   = os.path.join(DATA_ROOT, "val", "images")
TEST_IMG_DIR  = os.path.join(DATA_ROOT, "test", "images")

TRAIN_CSV = os.path.join(DATA_ROOT, "train", "segmentation.csv")
VAL_CSV   = os.path.join(DATA_ROOT, "val", "segmentation.csv")

IMG_SIZE = 512
BATCH_SIZE = 8

N_FOLDS = 5
PATIENCE_CLS = 4          # early stopping : epochs sans amélioration avant arrêt (classifieur)
PATIENCE_SEG = 4          # idem pour la segmentation
EPOCHS_CLS_MAX = 20       # borne haute, l'early stopping arrête généralement avant
EPOCHS_SEG_MAX = 30

# Grilles testées lors de l'optimisation des hyperparamètres post-entraînement
THRESHOLDS_TO_TEST = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
MIN_AREA_RATIOS_TO_TEST = [0.0001, 0.0005, 0.001, 0.002]
EPSILON_RATIOS_TO_TEST = [0.005, 0.01, 0.02]

# Deux architectures différentes pour diversifier l'ensemble du classifieur
CLASSIFIER_BACKBONES = ["efficientnet_b0", "resnet18"]

# Test-Time Augmentation : moyenne des prédictions sur l'image + ses flips
USE_TTA_CLASSIFIER = True
USE_TTA_SEGMENTATION = True

# Pondération Dice vs Cross-Entropy dans la perte de segmentation
DICE_WEIGHT = 0.5
CE_WEIGHT = 0.5

# Budget de temps total du pipeline, pour ne pas dépasser le quota GPU Kaggle
TIME_BUDGET_SECONDS = 8 * 60 * 60
SAFETY_MARGIN_SECONDS = 180

MODEL_DIR = "fold_models"     # checkpoints des modèles, par backbone et par fold
OOF_DIR = "oof_seg_probs"     # cartes de probabilité out-of-fold (segmentation)
CLS_OOF_PATH = "cls_oof_probs.npy"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OOF_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Seeds — reproductibilité
# ----------------------------------------------------------------------------

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)
print("Device:", DEVICE)


# ----------------------------------------------------------------------------
# Budget de temps
# ----------------------------------------------------------------------------
# Chronomètre global démarré au chargement du script. Les boucles d'entraînement
# vérifient budget_exceeded() régulièrement pour s'arrêter proprement (sans
# tout perdre) si le temps imparti est écoulé.

_pipeline_start_time = time.time()

def elapsed_seconds() -> float:
    return time.time() - _pipeline_start_time

def budget_exceeded(safety_margin_seconds: float = SAFETY_MARGIN_SECONDS) -> bool:
    return elapsed_seconds() >= (TIME_BUDGET_SECONDS - safety_margin_seconds)

def time_left_str() -> str:
    left = max(TIME_BUDGET_SECONDS - elapsed_seconds(), 0)
    return f"{left/3600:.2f}h restantes"


# ----------------------------------------------------------------------------
# Utilitaires polygon <-> mask
# ----------------------------------------------------------------------------

def safe_imread(path):
    """Lecture d'image avec message d'erreur explicite si le fichier est
    manquant ou corrompu, plutôt qu'un crash silencieux plus loin."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {path}")
    return img


def polygon_str_to_points(poly_str, width, height):
    """Convertit une chaîne de polygone normalisée (0-1) en points pixel.
    Gère les formats 'x1;y1 x2;y2 ...' et 'x1 y1 x2 y2 ...'."""
    poly_str = poly_str.strip()
    if poly_str == "" or poly_str == " ":
        return None

    tokens = poly_str.split()
    coords = []

    if all(";" in t for t in tokens):
        for t in tokens:
            parts = t.split(";")
            if len(parts) != 2:
                continue
            coords.append(float(parts[0]))
            coords.append(float(parts[1]))
    else:
        for t in tokens:
            try:
                coords.append(float(t))
            except ValueError:
                continue

    if len(coords) < 6 or len(coords) % 2 != 0:
        return None

    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= width
    pts[:, 1] *= height
    return pts.astype(np.int32)


def polygon_to_mask(poly_str, width, height):
    """Reconstruit un masque binaire (0/1) à partir d'une chaîne polygone."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = polygon_str_to_points(poly_str, width, height)
    if pts is None:
        return mask
    cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_to_polygon_str(mask, min_area_ratio=0.0005, epsilon_ratio=0.01):
    """Extrait le plus grand contour du masque et le simplifie en polygone
    normalisé. Retourne ' ' si aucun contour significatif n'est trouvé.
    Note : le format de soumission Kaggle n'accepte qu'un seul polygone par
    image, d'où le choix de ne garder que le plus grand contour."""
    h, w = mask.shape[:2]
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return " "

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < min_area_ratio * h * w:
        return " "

    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        return " "

    pts = approx.reshape(-1, 2).astype(np.float32)
    pts[:, 0] /= w
    pts[:, 1] /= h
    pts = np.clip(pts, 0.0, 1.0)

    coords = pts.flatten().round(6)
    return " ".join(f"{c:.6f}" for c in coords)


# ----------------------------------------------------------------------------
# Chargement des données
# ----------------------------------------------------------------------------

def load_segmentation_csv(csv_path, img_dir):
    df = pd.read_csv(csv_path)
    df["polygon"] = df["polygon"].fillna(" ").astype(str)
    df["has_manip"] = df["polygon"].apply(lambda s: 0 if s.strip() == "" else 1)
    df["img_path"] = df["image_id"].apply(lambda fn: os.path.join(img_dir, fn))

    # Résolution native de chaque image (lecture d'en-tête via PIL, rapide),
    # utilisée plus tard pour comparer les prédictions OOF à la bonne échelle.
    widths, heights = [], []
    for p in df["img_path"]:
        with Image.open(p) as im:
            w, h = im.size
        widths.append(w)
        heights.append(h)
    df["width"] = widths
    df["height"] = heights

    return df

train_df_raw = load_segmentation_csv(TRAIN_CSV, TRAIN_IMG_DIR)
val_df_raw   = load_segmentation_csv(VAL_CSV, VAL_IMG_DIR)

print("train:", train_df_raw["has_manip"].value_counts().to_dict())
print("val:  ", val_df_raw["has_manip"].value_counts().to_dict())

# train/ et val/ sont fusionnés pour maximiser les données disponibles avant
# de refaire notre propre découpage en K-Fold.
full_df = pd.concat([train_df_raw, val_df_raw], ignore_index=True)
print("\nfull_df (train+val fusionnés):", full_df.shape)
print(full_df["has_manip"].value_counts().to_dict())

# Folds du classifieur : stratifiés sur has_manip, sur l'ensemble des images.
skf_cls = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
full_df["cls_fold"] = -1
for fold_id, (_, val_idx) in enumerate(skf_cls.split(full_df, full_df["has_manip"])):
    full_df.loc[val_idx, "cls_fold"] = fold_id
assert (full_df["cls_fold"] >= 0).all()

# Folds de segmentation : uniquement sur les images manipulées (Cat2).
seg_df = full_df[full_df["has_manip"] == 1].reset_index(drop=True).copy()
kf_seg = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
seg_df["seg_fold"] = -1
for fold_id, (_, val_idx) in enumerate(kf_seg.split(seg_df)):
    seg_df.loc[val_idx, "seg_fold"] = fold_id
assert (seg_df["seg_fold"] >= 0).all()

print("\nRépartition cls_fold:")
print(full_df.groupby("cls_fold")["has_manip"].agg(["count", "sum"]))
print("\nRépartition seg_fold:")
print(seg_df.groupby("seg_fold").size())


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------

class ClassificationDataset(Dataset):
    """Cat2 (manipulé) vs Cat0/Cat1 (non manipulé)."""
    def __init__(self, df, img_size=IMG_SIZE, train=True):
        self.df = df.reset_index(drop=True)
        self.train = train

        if train:
            # Augmentations UNIQUEMENT en entraînement.
            self.tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # Validation déterministe : pas d'augmentation aléatoire.
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = safe_imread(row["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms.functional.to_pil_image(img)
        img = self.tf(img)
        label = torch.tensor(row["has_manip"], dtype=torch.long)
        return img, label, row["image_id"]


class SegmentationDataset(Dataset):
    """df doit déjà être filtré sur has_manip==1 (voir seg_df)."""
    def __init__(self, df, img_size=IMG_SIZE, train=True):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.train = train  # contrôle uniquement le flip aléatoire ci-dessous

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_bgr = safe_imread(row["img_path"])
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = polygon_to_mask(row["polygon"], w, h)

        img_rgb = cv2.resize(img_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Flip aléatoire uniquement en entraînement, jamais en validation/OOF.
        if self.train and random.random() < 0.5:
            img_rgb = np.fliplr(img_rgb).copy()
            mask = np.fliplr(mask).copy()

        img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        mask_t = torch.from_numpy(mask).long()
        return img_t, mask_t, row["image_id"]


# ----------------------------------------------------------------------------
# Modèles
# ----------------------------------------------------------------------------

def build_classifier(backbone_name="efficientnet_b0"):
    """Construit un classifieur binaire (has_manip 0/1) à partir d'un backbone
    pré-entraîné sur ImageNet. Supporte plusieurs architectures pour permettre
    un ensemble diversifié (voir CLASSIFIER_BACKBONES)."""
    if backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
    elif backbone_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
    else:
        raise ValueError(f"Backbone classifieur inconnu : {backbone_name}")
    return model


def build_segformer(num_labels=2):
    """SegFormer-B0 pré-entraîné sur ADE20K, adapté à 2 classes (fond / zone
    manipulée) — la couche de classification finale est réinitialisée."""
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    return model


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------

class DiceCELoss(nn.Module):
    """Combine Cross-Entropy (précision pixel par pixel) et Dice (recouvrement
    global de la zone) — mieux corrélée à l'IoU que la CE seule, surtout pour
    de petites zones à segmenter."""
    def __init__(self, dice_weight=DICE_WEIGHT, ce_weight=CE_WEIGHT, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets)

        probs = F.softmax(logits, dim=1)[:, 1]  # proba de la classe "manipulé"
        targets_f = (targets == 1).float()

        intersection = (probs * targets_f).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets_f.sum(dim=(1, 2))
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


# ----------------------------------------------------------------------------
# Métrique — Mean IoU exactement comme Kaggle
# ----------------------------------------------------------------------------

def compute_iou(pred_mask, gt_mask):
    """Convention Kaggle : deux masques vides -> IoU = 1.0 (pas de division
    par zéro)."""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    return intersection / union


# ----------------------------------------------------------------------------
# Test-Time Augmentation
# ----------------------------------------------------------------------------

@torch.no_grad()
def tta_classifier_probs(model, img_tensor):
    """Moyenne la probabilité 'manipulé' sur l'image originale + son flip
    horizontal."""
    variants = [img_tensor, torch.flip(img_tensor, dims=[3])]
    if not USE_TTA_CLASSIFIER:
        variants = variants[:1]
    probs = [F.softmax(model(v), dim=1)[0, 1].item() for v in variants]
    return float(np.mean(probs))


@torch.no_grad()
def tta_segmentation_probs(model, img_tensor, out_size):
    """Moyenne la carte de probabilité sur 4 variantes (identité, flip
    horizontal, flip vertical, flip des deux), chaque variante étant
    "dé-flippée" avant d'être moyennée pour revenir au référentiel original."""
    if USE_TTA_SEGMENTATION:
        variants = [
            img_tensor,
            torch.flip(img_tensor, dims=[3]),
            torch.flip(img_tensor, dims=[2]),
            torch.flip(torch.flip(img_tensor, dims=[3]), dims=[2]),
        ]
    else:
        variants = [img_tensor]

    probs_sum = None
    for i, v in enumerate(variants):
        out = model(pixel_values=v)
        logits = F.interpolate(out.logits, size=out_size, mode="bilinear", align_corners=False)
        p = F.softmax(logits, dim=1)[0, 1]

        # annule le flip appliqué à cette variante pour revenir à l'orientation d'origine
        if i == 1:
            p = torch.flip(p, dims=[1])
        elif i == 2:
            p = torch.flip(p, dims=[0])
        elif i == 3:
            p = torch.flip(torch.flip(p, dims=[1]), dims=[0])

        probs_sum = p if probs_sum is None else probs_sum + p

    return probs_sum / len(variants)


# ----------------------------------------------------------------------------
# Entraînement du classifieur (un fold)
# ----------------------------------------------------------------------------

def train_classifier_fold(backbone_name, fold_id, train_part, val_part,
                           epochs=EPOCHS_CLS_MAX, lr=1e-4, patience=PATIENCE_CLS):
    seed_everything(SEED + fold_id)
    model = build_classifier(backbone_name).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # Pondération de la CrossEntropy pour compenser le déséquilibre has_manip 0/1.
    counts = train_part["has_manip"].value_counts().sort_index()
    class_weights = torch.tensor(
        [1.0 / counts.get(0, 1), 1.0 / counts.get(1, 1)], dtype=torch.float32
    )
    class_weights = (class_weights / class_weights.sum()) * 2.0
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # LR scheduler cosine : décroît en douceur jusqu'à la fin de l'entraînement.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    train_ds = ClassificationDataset(train_part, train=True)
    val_ds   = ClassificationDataset(val_part, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    best_acc = 0.0
    best_path = f"{MODEL_DIR}/classifier_{backbone_name}_fold{fold_id}_best.pt"
    epochs_no_improve = 0

    for epoch in range(epochs):
        if budget_exceeded():
            print(f"[fold {fold_id} | {backbone_name}] budget dépassé, arrêt. ({time_left_str()})")
            break

        model.train()
        total_loss = 0.0
        for imgs, labels, _ids in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * imgs.size(0)
        train_loss = total_loss / len(train_ds)
        scheduler.step()

        # Validation : aucune mise à jour de poids ici, uniquement des métriques.
        model.eval()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for imgs, labels, _ids in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(dim=1)
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()

        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        print(f"[fold {fold_id} | {backbone_name}] epoch {epoch+1}/{epochs} "
              f"train_loss={train_loss:.4f} val_acc={acc:.4f} "
              f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[fold {fold_id} | {backbone_name}] early stopping")
                break

    # OOF : on recharge le MEILLEUR checkpoint (pas le dernier epoch) et on
    # calcule la probabilité "manipulé" avec TTA pour chaque image de validation.
    best_model = build_classifier(backbone_name).to(DEVICE)
    best_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    best_model.eval()

    cls_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    oof_probs = {}
    with torch.no_grad():
        for _, row in val_part.iterrows():
            img = safe_imread(row["img_path"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = transforms.functional.to_pil_image(img)
            x = cls_tf(pil_img).unsqueeze(0).to(DEVICE)
            oof_probs[row["image_id"]] = tta_classifier_probs(best_model, x)

    # Libère la mémoire GPU avant de passer au fold suivant.
    del model, opt, scaler, best_model
    torch.cuda.empty_cache()
    gc.collect()

    return {"fold": fold_id, "backbone": backbone_name, "best_acc": best_acc,
            "checkpoint": best_path, "oof_probs": oof_probs}


# ----------------------------------------------------------------------------
# Entraînement de la segmentation (un fold)
# ----------------------------------------------------------------------------

def train_segmentation_fold(fold_id, seg_train_part, seg_val_part,
                             epochs=EPOCHS_SEG_MAX, lr=6e-5, patience=PATIENCE_SEG):
    seed_everything(SEED + fold_id)
    model = build_segformer(num_labels=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    criterion = DiceCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    train_ds = SegmentationDataset(seg_train_part, train=True)
    val_ds   = SegmentationDataset(seg_val_part, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    best_iou = -1.0
    best_path = f"{MODEL_DIR}/segformer_fold{fold_id}_best.pt"
    epochs_no_improve = 0

    for epoch in range(epochs):
        if budget_exceeded():
            print(f"[fold {fold_id} | Segformer] budget dépassé, arrêt. ({time_left_str()})")
            break

        model.train()
        total_loss = 0.0
        for imgs, masks, _ids in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                out = model(pixel_values=imgs)
                logits = F.interpolate(out.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * imgs.size(0)
        train_loss = total_loss / len(train_ds)
        scheduler.step()

        model.eval()
        ious = []
        with torch.no_grad():
            for imgs, masks, _ids in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                out = model(pixel_values=imgs)
                logits = F.interpolate(out.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                preds = logits.argmax(dim=1)
                for p, g in zip(preds, masks):
                    ious.append(compute_iou(p.cpu().numpy(), g.cpu().numpy()))
        mean_iou = float(np.mean(ious)) if ious else 0.0

        print(f"[fold {fold_id} | Segformer] epoch {epoch+1}/{epochs} "
              f"train_loss={train_loss:.4f} val_iou={mean_iou:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[fold {fold_id} | Segformer] early stopping")
                break

    # OOF à résolution NATIVE (pas 512x512) + TTA, avec le meilleur checkpoint.
    best_model = build_segformer(num_labels=2).to(DEVICE)
    best_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    best_model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    with torch.no_grad():
        for _, row in seg_val_part.iterrows():
            img_bgr = safe_imread(row["img_path"])
            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            img_t = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
            img_t = ((img_t - mean) / std).unsqueeze(0).to(DEVICE)

            # prédiction à IMG_SIZE puis TTA, ensuite reprojetée à la taille native (h, w)
            probs_512 = tta_segmentation_probs(best_model, img_t, out_size=(IMG_SIZE, IMG_SIZE))
            probs_native = F.interpolate(
                probs_512.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            )[0, 0].cpu().numpy()

            prob_u8 = (probs_native * 255).astype(np.uint8)
            np.save(os.path.join(OOF_DIR, f"{row['image_id']}.npy"), prob_u8)

    del model, opt, scaler, best_model
    torch.cuda.empty_cache()
    gc.collect()

    return {"fold": fold_id, "best_iou": best_iou, "checkpoint": best_path}


# ----------------------------------------------------------------------------
# Boucles de cross-validation
# ----------------------------------------------------------------------------

def run_classification_cv():
    """Entraîne chaque backbone sur les N_FOLDS, puis moyenne les probabilités
    OOF entre backbones pour obtenir une estimation "ensemble" honnête."""
    all_results = []
    cls_oof_per_backbone = {b: {} for b in CLASSIFIER_BACKBONES}

    for backbone_name in CLASSIFIER_BACKBONES:
        if budget_exceeded():
            print(f"Budget dépassé avant le backbone {backbone_name}, on saute.")
            continue
        for fold_id in range(N_FOLDS):
            if budget_exceeded():
                print(f"Budget dépassé, arrêt des folds restants pour {backbone_name}.")
                break
            print(f"\n===== CLASSIFIEUR [{backbone_name}] — FOLD {fold_id+1}/{N_FOLDS} ===== ({time_left_str()})")
            train_part = full_df[full_df["cls_fold"] != fold_id]
            val_part   = full_df[full_df["cls_fold"] == fold_id]
            res = train_classifier_fold(backbone_name, fold_id, train_part, val_part)
            all_results.append(res)
            cls_oof_per_backbone[backbone_name].update(res["oof_probs"])

    # moyenne des probabilités OOF entre backbones (ensembling)
    all_ids = full_df["image_id"].tolist()
    cls_oof_probs = {}
    for img_id in all_ids:
        vals = [d[img_id] for d in cls_oof_per_backbone.values() if img_id in d]
        if vals:
            cls_oof_probs[img_id] = float(np.mean(vals))
    np.save(CLS_OOF_PATH, cls_oof_probs, allow_pickle=True)

    accs = [r["best_acc"] for r in all_results]
    print("\n=== CLASSIFICATION CROSS VALIDATION ===")
    for r in all_results:
        print(f"[{r['backbone']}] Fold {r['fold']+1} : accuracy = {r['best_acc']:.4f}")
    if accs:
        print(f"Mean accuracy : {np.mean(accs):.4f} | Std : {np.std(accs):.4f}")
    return all_results, cls_oof_probs


def run_segmentation_cv():
    fold_results = []
    for fold_id in range(N_FOLDS):
        if budget_exceeded():
            print("Budget dépassé, arrêt des folds de segmentation restants.")
            break
        print(f"\n===== SEGMENTATION — FOLD {fold_id+1}/{N_FOLDS} ===== ({time_left_str()})")
        seg_train_part = seg_df[seg_df["seg_fold"] != fold_id]
        seg_val_part   = seg_df[seg_df["seg_fold"] == fold_id]
        res = train_segmentation_fold(fold_id, seg_train_part, seg_val_part)
        fold_results.append(res)

    ious = [r["best_iou"] for r in fold_results]
    print("\n=== SEGMENTATION CROSS VALIDATION ===")
    for r in fold_results:
        print(f"Fold {r['fold']+1} : IoU = {r['best_iou']:.4f}")
    if ious:
        print(f"Mean IoU : {np.mean(ious):.4f} | Std : {np.std(ious):.4f}")
    return fold_results


# ----------------------------------------------------------------------------
# Optimisation des hyperparamètres (uniquement sur les prédictions OOF,
# jamais sur test/, pour ne pas biaiser les réglages)
# ----------------------------------------------------------------------------

def optimize_classifier_threshold(cls_oof_probs, thresholds=THRESHOLDS_TO_TEST):
    """Cherche le seuil qui maximise le F1 de la décision has_manip, sur OOF."""
    y_true = full_df.set_index("image_id").loc[list(cls_oof_probs.keys()), "has_manip"].values
    y_probs = np.array(list(cls_oof_probs.values()))

    results = {}
    for thr in thresholds:
        y_pred = (y_probs > thr).astype(int)
        results[thr] = f1_score(y_true, y_pred)
        print(f"Seuil classifieur {thr:.2f} -> F1 {results[thr]:.4f}")

    best_threshold = max(results, key=results.get)
    print(f"Meilleur seuil classifieur = {best_threshold} (F1={results[best_threshold]:.4f})")
    return best_threshold, results


def optimize_threshold(thresholds=THRESHOLDS_TO_TEST):
    """Cherche le seuil de binarisation qui maximise l'IoU moyen sur OOF
    (masques comparés à leur résolution native)."""
    results = {}
    for thr in thresholds:
        ious = []
        for _, row in seg_df.iterrows():
            oof_path = os.path.join(OOF_DIR, f"{row['image_id']}.npy")
            if not os.path.exists(oof_path):
                continue
            prob_u8 = np.load(oof_path)
            pred_mask = (prob_u8.astype(np.float32) / 255.0) > thr
            gt_mask = polygon_to_mask(row["polygon"], row["width"], row["height"]) > 0
            ious.append(compute_iou(pred_mask, gt_mask))
        mean_iou = float(np.mean(ious)) if ious else 0.0
        results[thr] = mean_iou
        print(f"Threshold {thr:.2f} -> IoU {mean_iou:.4f}")

    best_threshold = max(results, key=results.get)
    print(f"Best threshold = {best_threshold} (IoU={results[best_threshold]:.4f})")
    return best_threshold, results


def optimize_polygon_params(best_threshold,
                             min_area_ratios=MIN_AREA_RATIOS_TO_TEST,
                             epsilon_ratios=EPSILON_RATIOS_TO_TEST):
    """Teste le cycle complet masque -> polygone -> masque sur OOF, pour
    trouver les paramètres qui maximisent l'IoU réellement scoré par Kaggle
    (et non l'IoU du masque brut avant conversion)."""
    best_combo, best_iou, all_results = None, -1.0, []

    for min_area_ratio in min_area_ratios:
        for epsilon_ratio in epsilon_ratios:
            ious = []
            for _, row in seg_df.iterrows():
                oof_path = os.path.join(OOF_DIR, f"{row['image_id']}.npy")
                if not os.path.exists(oof_path):
                    continue
                prob_u8 = np.load(oof_path)
                pred_mask = ((prob_u8.astype(np.float32) / 255.0) > best_threshold).astype(np.uint8)

                poly_str = mask_to_polygon_str(pred_mask, min_area_ratio=min_area_ratio,
                                                epsilon_ratio=epsilon_ratio)
                reconstructed_mask = polygon_to_mask(poly_str, row["width"], row["height"]) > 0
                gt_mask = polygon_to_mask(row["polygon"], row["width"], row["height"]) > 0
                ious.append(compute_iou(reconstructed_mask, gt_mask))

            mean_iou = float(np.mean(ious)) if ious else 0.0
            all_results.append((min_area_ratio, epsilon_ratio, mean_iou))
            print(f"min_area_ratio={min_area_ratio} epsilon_ratio={epsilon_ratio} -> IoU {mean_iou:.4f}")

            if mean_iou > best_iou:
                best_iou = mean_iou
                best_combo = (min_area_ratio, epsilon_ratio)

    print(f"Best combo (min_area_ratio, epsilon_ratio) = {best_combo} | IoU = {best_iou:.4f}")
    return best_combo, all_results


def select_best_fold_checkpoints(cls_fold_results, seg_fold_results):
    """Stratégie alternative à l'ensemble complet : ne garder que le meilleur
    fold de chaque architecture/étage (utile pour comparer aux résultats de
    l'ensembling, ou pour une inférence plus légère)."""
    best_by_backbone = {}
    for backbone_name in CLASSIFIER_BACKBONES:
        candidates = [r for r in cls_fold_results if r["backbone"] == backbone_name]
        if candidates:
            best_by_backbone[backbone_name] = max(candidates, key=lambda r: r["best_acc"])
            b = best_by_backbone[backbone_name]
            print(f"Meilleur fold [{backbone_name}] : fold {b['fold']+1} (acc={b['best_acc']:.4f})")

    best_seg = max(seg_fold_results, key=lambda r: r["best_iou"]) if seg_fold_results else None
    if best_seg:
        print(f"Meilleur fold segformer : fold {best_seg['fold']+1} (iou={best_seg['best_iou']:.4f})")

    return best_by_backbone, best_seg


# ----------------------------------------------------------------------------
# Inférence et génération de submission.csv
# ----------------------------------------------------------------------------

@torch.no_grad()
def predict_submission(strategy="ensemble", cls_threshold=0.5, seg_threshold=0.5,
                        min_area_ratio=0.0005, epsilon_ratio=0.01,
                        cls_checkpoints=None, seg_checkpoint=None):
    """
    strategy = "best_fold" -> un seul modèle par backbone/étage
               (cls_checkpoints={backbone: path}, seg_checkpoint=path)
    strategy = "ensemble"  -> charge tous les checkpoints disponibles
               (tous backbones x tous folds pour le classifieur, tous les
               folds pour la segmentation) et moyenne leurs prédictions (TTA).

    Pipeline : image test -> classifieur (Cat2 ?) -> " " si non
                                                    -> Cat2 -> SegFormer -> seuil
                                                    -> masque -> polygone -> submission.csv
    """
    cls_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    classifiers = []  # liste de (backbone_name, model)
    segmenters = []

    if strategy == "best_fold":
        assert cls_checkpoints is not None and seg_checkpoint is not None
        for backbone_name, path in cls_checkpoints.items():
            m = build_classifier(backbone_name).to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            classifiers.append((backbone_name, m))
        m = build_segformer(num_labels=2).to(DEVICE)
        m.load_state_dict(torch.load(seg_checkpoint, map_location=DEVICE))
        segmenters.append(m)

    elif strategy == "ensemble":
        for backbone_name in CLASSIFIER_BACKBONES:
            for fold_id in range(N_FOLDS):
                p = f"{MODEL_DIR}/classifier_{backbone_name}_fold{fold_id}_best.pt"
                if os.path.exists(p):
                    m = build_classifier(backbone_name).to(DEVICE)
                    m.load_state_dict(torch.load(p, map_location=DEVICE))
                    classifiers.append((backbone_name, m))
        for fold_id in range(N_FOLDS):
            p = f"{MODEL_DIR}/segformer_fold{fold_id}_best.pt"
            if os.path.exists(p):
                m = build_segformer(num_labels=2).to(DEVICE)
                m.load_state_dict(torch.load(p, map_location=DEVICE))
                segmenters.append(m)
        print(f"Ensemble : {len(classifiers)} classifieurs, {len(segmenters)} segmenteurs")
    else:
        raise ValueError("strategy doit être 'best_fold' ou 'ensemble'")

    for _, m in classifiers:
        m.eval()
    for m in segmenters:
        m.eval()

    test_files = sorted(os.listdir(TEST_IMG_DIR))
    results = []

    for fname in test_files:
        img_path = os.path.join(TEST_IMG_DIR, fname)
        img_bgr = safe_imread(img_path)
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Étage 1 : classification, TTA + moyenne multi-backbones
        pil_img = transforms.functional.to_pil_image(img_rgb)
        cls_input = cls_tf(pil_img).unsqueeze(0).to(DEVICE)
        manip_probs = [tta_classifier_probs(m, cls_input) for _, m in classifiers]
        is_manip = np.mean(manip_probs) > cls_threshold

        if not is_manip:
            results.append({"image_id": fname, "polygon": " "})
            continue

        # Étage 2 : segmentation, TTA + moyenne multi-folds
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        img_t = ((img_t - mean) / std).unsqueeze(0).to(DEVICE)

        probs_sum = None
        for m in segmenters:
            p = tta_segmentation_probs(m, img_t, out_size=(IMG_SIZE, IMG_SIZE))
            probs_sum = p if probs_sum is None else probs_sum + p
        probs_mean = probs_sum / max(len(segmenters), 1)

        mask_small = (probs_mean.cpu().numpy() > seg_threshold).astype(np.uint8)
        mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

        poly_str = mask_to_polygon_str(mask_full, min_area_ratio=min_area_ratio,
                                        epsilon_ratio=epsilon_ratio)
        results.append({"image_id": fname, "polygon": poly_str})

    sub_df = pd.DataFrame(results)
    sub_df.to_csv("submission.csv", index=False)
    print("submission.csv genere:", sub_df.shape)
    return sub_df


# ----------------------------------------------------------------------------
# Exécution
# ----------------------------------------------------------------------------
# À séparer en plusieurs cellules sur Kaggle pour suivre la progression et
# pouvoir relancer une étape sans tout refaire depuis le début.

cls_fold_results, cls_oof_probs = run_classification_cv()

seg_fold_results = run_segmentation_cv()

best_cls_threshold, cls_thr_results = optimize_classifier_threshold(cls_oof_probs)

best_seg_threshold, seg_thr_results = optimize_threshold()

best_combo, poly_results = optimize_polygon_params(best_seg_threshold)
best_min_area_ratio, best_epsilon_ratio = best_combo

# Stratégie finale recommandée : ensemble complet (tous backbones x tous folds)
sub_df = predict_submission(
    strategy="ensemble",
    cls_threshold=best_cls_threshold,
    seg_threshold=best_seg_threshold,
    min_area_ratio=best_min_area_ratio,
    epsilon_ratio=best_epsilon_ratio,
)

sub_df.head()
