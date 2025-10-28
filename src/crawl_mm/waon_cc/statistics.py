from datasets import load_dataset
import matplotlib.pyplot as plt
from collections import Counter

ds = load_dataset("speed/waon-cc-pair-deduplicated", split="train")

# ds = ds.select_columns("page_title")

# タイトルごとの画像数をカウント
image_counts = Counter(ds["page_url"])
print("Top 10 titles with most images")
for title, count in image_counts.most_common(10):
    print(f"{title}: {count} images")
# タイトルのユニーク数を取得
unique_titles = len(image_counts)
# タイトルごとの画像数をリストに変換
image_counts = list(image_counts.values())
# 統計情報の表示
print(f"Total unique titles: {unique_titles}")
print(f"Total images: {len(ds)}")
# タイトルごとの画像数の統計情報
print(f"Max images per title: {max(image_counts)}")
print(f"Min images per title: {min(image_counts)}")
print(f"Average images per title: {sum(image_counts) / unique_titles:.2f}")


# 可視化
plt.figure(figsize=(10, 6))
plt.hist(image_counts, bins=50, color="blue", alpha=0.7)
plt.title(f"Distribution of Images per Title (Total Titles: {unique_titles})")
plt.xlabel("Number of Images")
plt.ylabel("Frequency")
plt.yscale("log")  # Y軸を対数スケールに設定
plt.grid(axis="y", alpha=0.75)
plt.savefig("waon_cc_images_per_title_distribution.png")
