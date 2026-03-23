import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import argparse

def run_matching(course_description, csv_file):
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return

    try:
        df = pd.read_csv(csv_file, sep=None, engine='python', on_bad_lines='skip')
        df = df.fillna("-")

        print("⏳ Loading AI model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        desc_col = ""
        for c in df.columns:
            if 'description' in str(c).lower():
                desc_col = c
                break
        if not desc_col:
            desc_col = df.columns[-1]

        db_texts = df[desc_col].astype(str).tolist()

        print("⚙️ Encoding...")
        db_embeddings = model.encode(db_texts, convert_to_tensor=True)
        query_embedding = model.encode(course_description, convert_to_tensor=True)

        scores = util.cos_sim(query_embedding, db_embeddings)[0]
        values, indices = torch.topk(scores, k=min(5, len(df)))

        print("\n📊 Results:\n")

        for i, (val, idx) in enumerate(zip(values, indices), 1):
            score_pct = val.item() * 100
            i_val = idx.item()

            code = df.iloc[i_val, 0]
            name = df.iloc[i_val, 1]

            status = "✅ PASS" if score_pct >= 70 else "❌ FAIL"

            print(f"{i}. {status} ({score_pct:.2f}%)")
            print(f"{code} - {name}")
            print("-"*50)

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc", required=True)
    parser.add_argument("--csv", default="data/SIIT_Course_Final_v2.csv")
    args = parser.parse_args()

    run_matching(args.desc, args.csv)
