import matplotlib.pyplot as plt
import numpy as np
import os

def _ensure_length(values, target_len):
    """
    Turn a scalar or length-1 iterable into an array of length target_len
    by simple repetition. If it's already the right length, just return it.
    Otherwise, error out.
    """
    arr = np.array(values, dtype=float)
    # scalar → full array
    if arr.ndim == 0:
        return np.full(target_len, arr.item(), dtype=float)
    # length-1 vector → full array
    if arr.size == 1 and target_len > 1:
        return np.full(target_len, arr.item(), dtype=float)
    # already correct length
    if arr.size == target_len:
        return arr
    raise ValueError(f"Cannot broadcast array of size {arr.size} → {target_len}")

def plot_metrics(qualities, metrics_dict, metric_name, output_dir='outputs'):
    qualities = np.array(qualities, dtype=float)
    plt.figure(figsize=(8,6))

    for method, values in metrics_dict.items():
        y = _ensure_length(values, len(qualities))
        plt.plot(qualities, y, marker='o', label=method)

    plt.xlabel('JPEG Quality Factor')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs JPEG Quality')
    plt.legend()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{metric_name}.png'))
    plt.close()

def save_metrics_table(qualities, metrics_dict, metric_name, output_dir='outputs'):
    qualities = np.array(qualities, dtype=float)
    os.makedirs(output_dir, exist_ok=True)
    fn = os.path.join(output_dir, f'{metric_name}_table.csv')

    # pre-broadcast all methods
    broadcasted = {
        method: _ensure_length(vals, len(qualities))
        for method, vals in metrics_dict.items()
    }

    with open(fn, 'w') as f:
        header = 'JPEG Quality,' + ','.join(broadcasted.keys()) + '\n'
        f.write(header)
        for i, q in enumerate(qualities):
            row = [f'{broadcasted[m][i]:.4f}' for m in broadcasted]
            f.write(f'{q},{",".join(row)}\n')
