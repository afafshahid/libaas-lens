# Libaas Lens

Clothing type classification for Pakistani ethnic wear and Western wear.
A MobileNetV2 transfer-learning model that recognises 10 garment types, running fully in the browser with TensorFlow.js.

CS4045 Deep Learning for Perception, FAST-NUCES Karachi, Fall 2026.

## Classes

- Ethnic wear: Shalwar Kameez, Kurta, Saree, Lehenga, Sherwani
- Western wear: T-shirt, Shirt, Jeans / Trousers, Dress, Jacket / Blazer

## Datasets

- [IndoFashion](https://indofashion.github.io/) (Kurta, Saree, Lehenga, Sherwani)
- [Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) (Western classes)
- [Pakistani Clothes](https://universe.roboflow.com/cooking-pot/pakistani-clothes) (Shalwar Kameez, CC BY 4.0)
- Our own photos (clothes only, no people or faces)

## Project layout

```
index.html      the whole website (HTML, CSS and JS in one file)
model/          TensorFlow.js model, added after training
  model.json
  *.bin
```

## Export the trained model

```
pip install tensorflowjs
tensorflowjs_converter --input_format=keras model.h5 model/
```

Keep `model/` next to `index.html`. The page loads `model/model.json` automatically.
Until it exists, the page runs in demo mode with placeholder predictions.

Then, at the top of the script in `index.html`:

1. Check that `CLASSES` is in the same order as the model outputs (alphabetical if you used `image_dataset_from_directory`).
2. Fill in `RESULTS` with your metrics (for example `accuracy: 0.873`).

## Run locally

The model is fetched over HTTP, so serve the folder instead of opening the file directly:

```
python -m http.server 8000
```

Then open http://localhost:8000.

## Deploy

Push to GitHub and enable Pages (Settings, Pages, deploy from the `main` branch root).

## Team

- Afaf Shahid (23K-0678), group coordinator
- Ahmad Noorani (23K-0547)
