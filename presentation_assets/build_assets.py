from pathlib import Path
import csv
import textwrap

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation_assets"


def font(size=28, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def terminal_image(filename, title, lines):
    width, height = 1500, 850
    img = Image.new("RGB", (width, height), "#101418")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((28, 28, width - 28, height - 28), radius=24, fill="#151b22")
    draw.ellipse((60, 62, 82, 84), fill="#ff5f57")
    draw.ellipse((94, 62, 116, 84), fill="#ffbd2e")
    draw.ellipse((128, 62, 150, 84), fill="#28c840")
    draw.text((185, 57), title, fill="#b7c3d0", font=font(24))

    y = 120
    mono = font(27)
    for raw in lines:
        wrapped = textwrap.wrap(raw, width=90, replace_whitespace=False) or [""]
        for line in wrapped:
            fill = "#7ee787" if line.strip().startswith("$") else "#d6dee8"
            draw.text((70, y), line, fill=fill, font=mono)
            y += 40
    img.save(OUT / filename)


def metrics_chart():
    with open(ROOT / "metrics.csv", newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) + 1 for row in metrics]
    accuracy = [float(row["accuracy"]) * 100 for row in metrics]
    val_accuracy = [float(row["val_accuracy"]) * 100 for row in metrics]
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")
    ax.plot(epochs, accuracy, marker="o", lw=3, label="Train accuracy", color="#2563eb")
    ax.plot(epochs, val_accuracy, marker="o", lw=3, label="Validation accuracy", color="#16a34a")
    ax.set_title("Model Accuracy Across 10 Epochs", fontsize=20, weight="bold", pad=18)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_ylim(65, 100)
    ax.grid(True, color="#dbe3ef", linewidth=1)
    ax.legend(loc="lower right", frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "metrics_chart.png", bbox_inches="tight")
    plt.close(fig)


def sample_montage():
    samples = [
        ROOT / "data/train/cats/cat.1.jpg",
        ROOT / "data/train/cats/cat.2.jpg",
        ROOT / "data/train/dogs/dog.1.jpg",
        ROOT / "data/train/dogs/dog.2.jpg",
    ]
    tile_w, tile_h = 420, 300
    canvas = Image.new("RGB", (tile_w * 2, tile_h * 2 + 72), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    labels = ["train/cats", "train/cats", "train/dogs", "train/dogs"]
    for idx, path in enumerate(samples):
        img = Image.open(path).convert("RGB")
        img.thumbnail((tile_w - 28, tile_h - 58))
        x = (idx % 2) * tile_w + (tile_w - img.width) // 2
        y = (idx // 2) * tile_h + 22
        draw.rounded_rectangle(((idx % 2) * tile_w + 12, (idx // 2) * tile_h + 12, (idx % 2) * tile_w + tile_w - 12, (idx // 2) * tile_h + tile_h - 12), radius=14, fill="#ffffff", outline="#d8e1ec")
        canvas.paste(img, (x, y))
        draw.text(((idx % 2) * tile_w + 28, (idx // 2) * tile_h + tile_h - 42), labels[idx], fill="#334155", font=font(22))
    draw.text((24, tile_h * 2 + 18), "Local DVC-tracked dataset sample: cats vs dogs", fill="#0f172a", font=font(28, bold=True))
    canvas.save(OUT / "dataset_samples.png")


terminal_image(
    "dvc_repro_screenshot.png",
    "DVC pipeline run",
    [
        "$ .env/bin/dvc repro",
        "'data.dvc' didn't change, skipping",
        "Stage 'train' didn't change, skipping",
        "Data and pipelines are up to date.",
    ],
)
terminal_image(
    "dvc_status_screenshot.png",
    "DVC project status",
    [
        "$ .env/bin/dvc status",
        "Data and pipelines are up to date.",
        "",
        "$ find data/train data/validation -maxdepth 2 -type f | wc -l",
        "1800",
    ],
)
terminal_image(
    "project_tree_screenshot.png",
    "Project files",
    [
        "$ tree -L 2",
        ".",
        "|-- data/",
        "|   |-- train/        # 1000 images",
        "|   `-- validation/   # 800 images",
        "|-- dvc.yaml          # train stage",
        "|-- params.yaml       # image size, epochs, paths",
        "|-- train.py          # VGG16 feature extraction + classifier",
        "|-- metrics.csv       # tracked run metrics",
        "`-- model.weights.h5  # trained top model weights",
    ],
)
metrics_chart()
sample_montage()
