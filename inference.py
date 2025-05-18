# Import necessary libraries
import os
import zipfile
import gdown
import pandas as pd
import argparse
import torch
import clip 
import numpy as np
from PIL import Image
import os
from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel
from tqdm import tqdm
import evaluate
import re
import math
from bert_score import score
from pathlib import Path
from sentence_transformers import SentenceTransformer

#Define constants and configuration

BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2 
SCRIPT_DIR = Path(__file__).resolve().parent # Get current script directory
OUTPUT_DIR = os.path.join(SCRIPT_DIR,"INPUTS/kaggle/working/blip_vqa_lora_finetuned")
LORA_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "lora_adapters/")
MODEL_NAME = "Salesforce/blip-vqa-base"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
FILE_ID = '1HvMBKwKywVFinX_Y-SoLRdTWGTEkl84u'
MAX_LENGTH = 128

# Function to download and unzip a file from Google Drive
def download_and_unzip(file_id: str, output_folder: str = None):
    """
    Download and unzip a file from Google Drive.
    Args:
        file_id (str): The Google Drive file ID.
        output_folder (str, optional): The folder to extract the files to. If None, a folder with the same name as the zip file will be created.
    """

    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    zip_path = "temp_download.zip"

    print(f"Downloading from {url!r} …")
    gdown.download(url, zip_path, quiet=False)

    if output_folder is None:
        base, _ = os.path.splitext(zip_path)
        output_folder = base
    os.makedirs(output_folder, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        print("Contents:", zf.namelist())
        zf.extractall(output_folder)

    print(f"✔ Downloaded and extracted to ./{output_folder}/")

# Metric 1: VTGS (Visual Textual Grounded Score)
def metric1(df, image_dir):
    def load_clip(device):
        # Load CLIP model and preprocessing
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model.eval(), preprocess
     
    # Encode image features using CLIP
    def embed_images(model, preprocess, image_paths, device, batch_size=32):
        embs = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Images"):
            batch = image_paths[i:i+batch_size]
            imgs = [preprocess(Image.open(os.path.join(image_dir,p)).convert("RGB")).unsqueeze(0) for p in batch]
            imgs = torch.cat(imgs, dim=0).to(device)
            with torch.no_grad():
                e = model.encode_image(imgs)
                e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.cpu().numpy())
        return np.vstack(embs)
    
    # Encode text features using CLIP
    def embed_texts(model, texts, device, batch_size=32):
        embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Texts"):
            batch = texts[i:i+batch_size]
            toks = clip.tokenize(batch, truncate=True).to(device)
            with torch.no_grad():
                e = model.encode_text(toks)
                e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.cpu().numpy())
        return np.vstack(embs)
    
    def cosine(a, b):
        return np.einsum('ij,ij->i', a, b)
    
     # Compute VTGS combining CLIP and BERTScore
    def compute_vtgs(df, device):
        clip_model, preprocess = load_clip(device)
        img_emb = embed_images (clip_model, preprocess, df["image_name"].tolist(), device)
        pred_emb= embed_texts (clip_model, df["generated_answer"].tolist(),      device)
        raw_clip = cosine(img_emb, pred_emb)
        clip_norm = (raw_clip + 1.0) / 2.0
        P, R, F1 = score(df["generated_answer"].tolist(),df["answer"].tolist(),lang="en",verbose=False,device=device)
        bert_f1 = F1.cpu().numpy()
        vtgs = np.sqrt(bert_f1 * clip_norm)
        df_out = df.copy()
        df_out["bertscore_f1"] = bert_f1
        df_out["clipscore"]    = clip_norm
        df_out["vtgs"]         = vtgs
        return df_out
    
    # Normalize text
    df['generated_answer']=df['generated_answer'].apply(lambda x:str(x).lower())
    df['answer']=df['answer'].apply(lambda x:str(x).lower())
    result = compute_vtgs(df, DEVICE)
    mean_v = result["vtgs"].mean()
    std_v  = result["vtgs"].std()
    return [mean_v,std_v]

def metric2(
    df,
    image_dir=None,
    model_name: str = "all-MiniLM-L6-v2",
    alpha: float = 5.0,
    epsilon: float = 1e-6,
    txt_batch: int = 256
):
    """
    CSNS — Contextual Semantic-Numeric Score (batched, with NaN guard, skipping NaNs).
    
    Returns:
        [mean_score, std_score]
    """
    def is_number(x: str) -> bool:
        try:
            float(x)
            return True
        except:
            return False

    def numeric_score(gt: str, pred: str) -> float:
        a = float(gt)
        b = float(pred)
        rel_err = abs(b - a) / (abs(a) + epsilon)
        return math.exp(-alpha * rel_err)

    df = df.copy()
    df["answer"]           = df["answer"].astype(str).str.lower()
    df["generated_answer"] = df["generated_answer"].astype(str).str.lower()
    df = df[df["answer"].notna() & df["generated_answer"].notna()]
    df = df[~df["answer"].isin(["nan", "nan "])]  # in case strings 'nan'
    df = df[~df["generated_answer"].isin(["nan", "nan "])]
    df = df.reset_index(drop=True)
    gts   = df["answer"].tolist()
    preds = df["generated_answer"].tolist()
    mask_semantic = [
        not (is_number(g) and is_number(p))
        for g, p in zip(gts, preds)
    ]
    idx_semantic = np.where(mask_semantic)[0]

    if len(idx_semantic) > 0:
        model = SentenceTransformer(model_name)
        g_list = [gts[i] for i in idx_semantic]
        p_list = [preds[i] for i in idx_semantic]

        def batched_encode(texts, desc: str):
            embs = []
            for i in tqdm(range(0, len(texts), txt_batch), desc=desc):
                chunk = texts[i : i + txt_batch]
                embs.append(
                    model.encode(
                        chunk,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        batch_size=txt_batch,
                        show_progress_bar=False
                    )
                )
            return np.vstack(embs)

        emb_g = batched_encode(g_list, desc="CSNS: encoding GT")
        emb_p = batched_encode(p_list, desc="CSNS: encoding Pred")

        cos = (emb_g * emb_p).sum(axis=1)
        cos = np.nan_to_num(cos, nan=0.0, posinf=1.0, neginf=0.0)
    scores = np.zeros(len(df), dtype=float)
    for i in tqdm(range(len(df)), desc="CSNS: numeric scoring"):
        if not mask_semantic[i]:
            scores[i] = numeric_score(gts[i], preds[i])
    if len(idx_semantic) > 0:
        scores[idx_semantic] = np.clip(cos, 0.0, 1.0)
    mean_score = float(np.mean(scores))
    std_score  = float(np.std(scores))
    return [mean_score, std_score]


# Metric 3: BERTScore Precision
def metric3(df, image_dir):
    df['generated_answer']=df['generated_answer'].apply(lambda x:str(x).lower())
    df['answer']=df['answer'].apply(lambda x:str(x).lower())
    P, _, _ = score(df["generated_answer"].tolist(),df["answer"].tolist(),lang="en",verbose=False,device=DEVICE)
    return [P.mean(), P.std()]

# Metric 4: Exact Match Accuracy
def metric4(df, image_dir):
    df['generated_answer']=df['generated_answer'].apply(lambda x:str(x).lower())
    df['answer']=df['answer'].apply(lambda x:str(x).lower())
    res=(df['generated_answer']==df['answer'])
    return [res.mean(), res.std()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()
    
    download_and_unzip(FILE_ID, "INPUTS") # Download required files from Google Drive

    # Load base and fine-tuned models
    base_model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME)
    finetuned_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    finetuned_model = finetuned_model.to(DEVICE) 
    finetuned_model.eval()
    eval_processor = BlipProcessor.from_pretrained(LORA_ADAPTER_DIR)

    # Load evaluation dataset
    eval_df = pd.read_csv(args.csv_path)
    predictions_ft = []
    ground_truths_normalized_ft = []
    original_indices_ft = []

    num_batches_eval = math.ceil(len(eval_df) / EVAL_BATCH_SIZE)

    IMAGE_BASE_DIR = args.image_dir

    print(eval_df.head())

    # Evaluate the model in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(eval_df), EVAL_BATCH_SIZE), total=num_batches_eval, desc="Evaluating Fine-tuned Model"):
            batch_df = eval_df[i:i+EVAL_BATCH_SIZE]
            
            batch_images_pil = []
            batch_questions = []
            current_batch_ground_truths = [] # Ground truths for this specific batch
            current_batch_original_indices = [] # Original indices for this specific batch

            for idx_in_batch, (original_df_idx, row) in enumerate(batch_df.iterrows()):
                question = str(row['question'])
                true_answer = str(row['answer']).lower().strip()
                # Use the 'filename' column
                image_filename = str(row['image_name']) #change the col
                img_path = os.path.join(IMAGE_BASE_DIR, image_filename)

                try:
                    raw_image = Image.open(img_path).convert('RGB')
                    batch_images_pil.append(raw_image)
                    batch_questions.append(question)
                    current_batch_ground_truths.append(true_answer)
                    current_batch_original_indices.append(original_df_idx) #loading file_path, question and answers for each batch
                except FileNotFoundError:
                    print(f"Warning (Eval): Image not found at {img_path}. Skipping.")
                except Exception as e:
                    print(f"Warning (Eval): Error loading image {img_path}: {e}. Skipping.")

            if not batch_images_pil:
                print(f"Warning (Eval): No valid images for batch starting at {i}. Skipping.")
                continue
            
            # Generate predictions using the model (processing each batch)
            inputs = eval_processor(images=batch_images_pil, text=batch_questions, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            outputs = finetuned_model.generate(**inputs, max_new_tokens=10)
            batch_preds_decoded = eval_processor.batch_decode(outputs, skip_special_tokens=True)

            for pred_idx, decoded_pred in enumerate(batch_preds_decoded):
                predicted_answer = decoded_pred.strip().lower()
                predicted_answer = re.sub(r'[^\w\s]', '', predicted_answer)

                true_answer_normalized = current_batch_ground_truths[pred_idx]
                true_answer_normalized = re.sub(r'[^\w\s]', '', true_answer_normalized)

                predictions_ft.append(predicted_answer)
                ground_truths_normalized_ft.append(true_answer_normalized)
                original_indices_ft.append(current_batch_original_indices[pred_idx])

    # Update dataframe with predictions
    eval_df['answer']=ground_truths_normalized_ft
    eval_df['generated_answer']=predictions_ft
    results_ft_df = eval_df
    results_ft_df.to_csv("results.csv", index=False)
    
    # Compute and print all evaluation metrics
    metrics={"VTGS":metric1, "CSNS":metric2, "BERT":metric3, "Accuracy":metric4}
    for name, func in metrics.items():
        mean, std = func(results_ft_df, IMAGE_BASE_DIR)
        print(f'{name} - {mean:.2f} ± {std:.2f}')
        
if __name__ == "__main__":
    main()