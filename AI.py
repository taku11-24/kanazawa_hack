import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from pathlib import Path

# ===== モデル定義 =====
class RegressionNet(torch.nn.Module):
    def __init__(self, pretrained=True, embed_dim=512):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = backbone.fc.in_features
        modules = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*modules)
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features, embed_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(embed_dim, 2)
        )

    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.head(out)
        return out[:, 0], out[:, 1]


# ===== 推論関数 =====
def predict_angle_distance(image_path, checkpoint="models/best_model.pth", device=None):
    """
    1枚の画像パスを受け取り、学習済みモデルで角度と距離を返す。
    Args:
        image_path (str or Path): 入力画像のパス
        checkpoint (str): 学習済みモデルのパス
        device (str or torch.device): "cpu" or "cuda"
    Returns:
        (angle, distance): 角度(deg), 距離
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # モデルロード
    model = RegressionNet(pretrained=True).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 画像前処理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    # 推論
    with torch.no_grad():
        angle, distance = model(x)
    return float(angle.item()) % 360.0, float(distance.item())


# ===== 例: main から呼ぶ場合 =====
if __name__ == "__main__":
    # 例: 外部から渡される画像パス
    input_image_path = "images/100_image35.png"

    # 予測
    ang, dist = predict_angle_distance(input_image_path)
    print(f"Predicted angle: {ang:.2f} deg, distance: {dist:.2f}")
