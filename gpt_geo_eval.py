import argparse, base64, json, os, time, csv, openai
from tqdm import tqdm

PROMPT = """You are a geolocation expert. 
For the image the user sends, reply **only** in JSON like:
{
  "top1": "Country",
  "top5": ["Country1","Country2","Country3","Country4","Country5"]
}
Return your five most likely countries ranked from most to least likely."""

def img_to_data_uri(path):
    import mimetypes, base64
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    b64  = base64.b64encode(open(path, "rb").read()).decode()
    return f"data:{mime};base64,{b64}"

def ask_gpt(img_path, model="gpt-4o-mini"):
    resp = openai.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=50,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user",
             "content": [{"type": "image_url",
                          "image_url": {"url": img_to_data_uri(img_path),
                                        "detail": "low"}}]},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def main(csv_path: str,
         print_every: int = 100,
         sleep_sec: float = 1.2,
         jsonl_path: str = "gpt_results.jsonl",
         model_name: str = "gpt-4o-mini"):

    # ensure we don't overwrite previous runs accidentally
    if os.path.exists(jsonl_path):
        raise FileExistsError(f"{jsonl_path} already exists; "
                              "rename or move it before running again.")

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    tot = top1 = top5 = 0
    start = time.time()

    with open(jsonl_path, "w") as fout:
        for row in tqdm(rows, total=len(rows)):
            res = ask_gpt(row["image"], model=model_name)

            label = row["label"]
            top1  += int(res["top1"] == label)
            top5  += int(label in res["top5"])
            tot   += 1

            # stream result to disk
            fout.write(json.dumps({
                "image": row["image"],
                "ground_truth": label,
                "gpt_top1": res["top1"],
                "gpt_top5": res["top5"]
            }) + "\n")

            # interim log
            if tot % print_every == 0:
                elapsed = time.time() - start
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"{tot}/{len(rows)}  "
                      f"top-1={top1/tot:.3f}  top-5={top5/tot:.3f}  "
                      f"{elapsed/60:.1f} min elapsed")

            time.sleep(sleep_sec)   # stay under rate-limit

    print("\n════ FINAL RESULTS ════")
    print(f" Samples evaluated : {tot}")
    print(f" Top-1 accuracy    : {top1/tot:.4f}")
    print(f" Top-5 accuracy    : {top5/tot:.4f}")
    print(f" Raw responses saved to → {jsonl_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns image,label")
    ap.add_argument("--print-every", type=int, default=100,
                    help="Status update interval")
    args = ap.parse_args()
    main(args.csv, args.print_every)
