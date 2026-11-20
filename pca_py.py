
import math
from PIL import Image

# ---------------- Linear Algebra Helpers ----------------

def dot(u, v):
    """Dot product of two vectors."""
    return sum(ui * vi for ui, vi in zip(u, v))


def mat_vec_mul(M, v):
    """Matrix-vector multiplication."""
    return [dot(row, v) for row in M]


def outer(u, v):
    """Outer product of two vectors."""
    return [[ui * vi for vi in v] for ui in u]


def subtract_matrix(A, B):
    """Element-wise subtraction of two matrices."""
    return [[a - b for a, b in zip(rowA, rowB)] for rowA, rowB in zip(A, B)]


def normalize(v):
    """Scale a vector to unit length (returns None for a zero vector)."""
    norm = math.sqrt(dot(v, v))
    if norm < 1e-12:
        return None
    return [x / norm for x in v]


def orthogonalize(v, basis):
    """Remove from v its components along each (unit) vector in basis (Gram-Schmidt)."""
    v = v[:]
    for u in basis:
        proj = dot(v, u)
        v = [vi - proj * ui for vi, ui in zip(v, u)]
    return v


def start_vector(n, found):
    """
    Pick a unit start vector with as much weight as possible outside the
    eigenvectors already found. A fixed start such as [1, 1, 1] can be
    orthogonal to the eigenvector we are looking for, in which case power
    iteration never finds it.
    """
    candidates = [[1.0] * n] + [[1.0 if i == j else 0.0 for i in range(n)] for j in range(n)]
    best = max((orthogonalize(c, found) for c in candidates), key=lambda v: dot(v, v))
    return normalize(best)


def power_iteration(A, num_iters=1000, tol=1e-6, found=()):
    """
    Power iteration to compute the dominant eigenvalue and eigenvector of a
    symmetric matrix A, restricted to the space orthogonal to the eigenvectors
    in `found`. Returns (eigenvalue, eigenvector).
    """
    n = len(A)
    b_k = start_vector(n, found)
    for _ in range(num_iters):
        b_k1 = normalize(orthogonalize(mat_vec_mul(A, b_k), found))
        if b_k1 is None:
            # A is zero on the remaining space: b_k is an eigenvector with eigenvalue 0
            break
        if max(abs(b_k1[i] - b_k[i]) for i in range(n)) < tol:
            b_k = b_k1
            break
        b_k = b_k1
    eigenvalue = dot(b_k, mat_vec_mul(A, b_k))
    return eigenvalue, b_k


def eigen_decompose(C):
    """
    Deflation-based eigen-decomposition for a symmetric matrix C (3×3).
    Returns eigenvalues (desc) and corresponding eigenvectors.
    """
    A = [row[:] for row in C]
    n = len(A)
    pairs = []
    for _ in range(n):
        val, vec = power_iteration(A, found=[p[1] for p in pairs])
        pairs.append((val, vec))
        # Deflation: subtract val * (vec * vec^T)
        deflation = [[val * v_i * v_j for v_j in vec] for v_i in vec]
        A = subtract_matrix(A, deflation)
    pairs.sort(key=lambda p: p[0], reverse=True)
    vals = [p[0] for p in pairs]
    vecs = [p[1] for p in pairs]
    return vals, vecs

# ---------------- PCA Pipeline ----------------

def load_image(path):
    """Load image and split into R, G, B channels as flat lists."""
    img = Image.open(path).convert('RGB')
    w, h = img.size
    pixels = list(img.getdata())
    r = [p[0] for p in pixels]
    g = [p[1] for p in pixels]
    b = [p[2] for p in pixels]
    return [r, g, b], (w, h)


def save_image(channels, size, path):
    """Save R, G, B channel lists as an image."""
    r, g, b = channels
    # clamp values and round to the nearest integer
    pixels = [(round(min(max(r[i],0),255)),
               round(min(max(g[i],0),255)),
               round(min(max(b[i],0),255)))
              for i in range(len(r))]
    img = Image.new('RGB', size)
    img.putdata(pixels)
    img.save(path)


def compute_means(channels):
    """Compute mean of each channel list."""
    return [sum(ch)/len(ch) for ch in channels]


def center_channels(channels, means):
    """Subtract mean from each channel value."""
    return [[channels[i][j] - means[i] for j in range(len(channels[0]))]
            for i in range(3)]


def compute_covariance_matrix(centered):
    """Compute 3×3 covariance matrix from centered channel data."""
    n = len(centered[0])
    C = [[0.0]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            C[i][j] = sum(centered[i][k] * centered[j][k] for k in range(n)) / (n - 1)
    return C


def reconstruct(centered, means, eigvecs, k):
    """Project onto top-k eigvecs and reconstruct channels."""
    n = len(centered[0])
    recon = [[0.0]*n for _ in range(3)]
    for idx in range(n):
        x = [centered[i][idx] for i in range(3)]
        # projection coefficients
        y = [dot(x, eigvecs[j]) for j in range(k)]
        # reconstruct x_hat
        x_hat = [0.0, 0.0, 0.0]
        for j in range(k):
            for i in range(3):
                x_hat[i] += y[j] * eigvecs[j][i]
        # add mean back
        for i in range(3):
            recon[i][idx] = x_hat[i] + means[i]
    return recon


def process_image(input_path, output_path, k=3):
    """End-to-end PCA transform on an image, keeping k components."""
    if not 1 <= k <= 3:
        raise ValueError(f"k must be 1, 2 or 3 for an RGB image, got {k}")
    channels, size = load_image(input_path)
    means = compute_means(channels)
    centered = center_channels(channels, means)
    C = compute_covariance_matrix(centered)
    _, eigvecs = eigen_decompose(C)
    recon = reconstruct(centered, means, eigvecs, k)
    save_image(recon, size, output_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PCA on 24-bit images')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output image path')
    parser.add_argument('-k', type=int, default=3, choices=[1, 2, 3],
                        help='Number of principal components to keep (1-3)')
    args = parser.parse_args()
    process_image(args.input, args.output, args.k)
