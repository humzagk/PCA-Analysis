# PCA Image Transformer

[![tests](https://github.com/humzagk/PCA-Analysis/actions/workflows/tests.yml/badge.svg)](https://github.com/humzagk/PCA-Analysis/actions/workflows/tests.yml)

Principal component analysis on the colour channels of an image, written from scratch in pure Python. NumPy and scikit-learn are not used: the covariance matrix, the eigen-decomposition and the projection are all coded by hand. The project comes with a command-line tool and a small Tkinter GUI.

## What it does

Each pixel is treated as a 3-dimensional point (R, G, B). The tool:

1. computes the mean of each channel and centres the pixels;
2. builds the 3×3 covariance matrix of the channels;
3. finds its eigenvalues and eigenvectors with **power iteration plus deflation**, keeping each new eigenvector orthogonal to the ones already found (Gram-Schmidt);
4. projects every pixel onto the top *k* principal components and reconstructs the image from them.

With `k = 3` the image is reconstructed losslessly. With `k = 1` or `k = 2` the colour information is compressed onto the strongest axes of variation. `k = 1` gives a "principal-axis" tinted greyscale, and `k = 2` keeps most of the colour while dropping the weakest direction.

## Installation

Requires Python 3.11 or newer (CI runs on 3.11 and 3.13).

```bash
git clone https://github.com/humzagk/PCA-Analysis.git
cd PCA-Analysis
pip install -r requirements.txt
```

## Usage

### Command line

```bash
python pca_py.py input.png output.png -k 2
```

| Argument | Description |
|----------|-------------|
| `input`  | Path to the input image (any format Pillow can read) |
| `output` | Path where the reconstructed image is written |
| `-k`     | Number of principal components to keep: 1, 2 or 3 (default 3) |

### GUI

```bash
python gui_py.py
```

Open an image, choose the number of components with the slider, click **Run PCA** to see the result next to the original, and use **Save As** to keep it. On Windows a pre-built `gui_py.exe` is also included.

### As a library

```python
from pca_py import eigen_decompose, process_image

process_image("photo.jpg", "photo_k1.jpg", k=1)
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests check the eigen-decomposition against matrices with known eigenvalues, including a repeated eigenvalue. They also check that the eigenvectors are orthonormal, that `k = 3` reproduces the input exactly (for both colour and greyscale images), and that invalid values of `k` are rejected.

## Project structure

```
pca_py.py        # linear-algebra helpers, eigen-decomposition and the PCA pipeline + CLI
gui_py.py        # Tkinter front end
tests/           # unit tests
```

## Limitations

- Pure Python: images are processed pixel by pixel, so very large images (tens of megapixels) are slow. A 300×300 image takes well under a second.
- Only 3-channel RGB is supported. Alpha channels are dropped when the image is loaded.

## License

[MIT](LICENSE)
