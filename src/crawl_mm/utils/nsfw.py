import torch.nn.functional as F
import torch
from crawl_mm.clip_nsfw.inference import build_inference_model
from PIL import Image
from typing import List


class NSFWModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.head, self.backbone, self.pre_processing = build_inference_model(
            "src/crawl_mm/clip_nsfw/models/clip_ViT-B-32_openai_binary_nsfw_head.pth",
            "ViT-B-32",
            "openai",
            self.device,
        )

    def predict(self, x: torch.tensor) -> List[float]:
        # 前処理＋テンソル化
        # processed = [
        #     self.pre_processing(remove_transparency(img)).unsqueeze(0)
        #     for img in imgs
        # ]
        # x = torch.cat(processed, dim=0).to(self.device)
        with torch.no_grad():
            p = self.backbone(x)
            p = F.normalize(p)
            c = self.head(p)
            scores = c[:, 0].tolist()

        return scores


if __name__ == "__main__":
    model = NSFWModel()
    # 複数画像テスト
    images = [Image.open(p) for p in ["cat.jpg", "test.jpg"]]
    inputs = torch.stack([model.pre_processing(img) for img in images]).to(model.device)
    print(inputs.shape)  # (batch_size, 3, 224, 224)
    scores = model.predict(inputs)
    for path, score in zip(images, scores):
        print(f"NSFW score: {score}")
