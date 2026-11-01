# Three-Input XNOR Neural Network From Scratch

A small feed-forward neural network built with only NumPy. It learns a
three-input XNOR function, where the output is `1` when all three inputs are
equal (`000` and `111`) and `0` otherwise.

## Truth table

| A | B | C | Output |
|---|---|:-:|:------:|
| 0 | 0 | 0 | 1 |
| 0 | 0 | 1 | 0 |
| 0 | 1 | 0 | 0 |
| 0 | 1 | 1 | 0 |
| 1 | 0 | 0 | 0 |
| 1 | 0 | 1 | 0 |
| 1 | 1 | 0 | 0 |
| 1 | 1 | 1 | 1 |

This is the **all-inputs-equal** interpretation of three-input XNOR. It is not
the even-parity function (which would also output `1` for `011`, `101`, and
`110`).

## Network architecture

```text
3 inputs -> 4 hidden neurons -> 1 output
```

- All neurons use the sigmoid activation.
- Both layers are fully connected.
- There are no bias terms.
- Weights are initialized from a standard normal distribution
  (`np.random.randn`, std 1), so every run starts from different weights.

### Sigmoid activation

$$
\sigma(s) = \frac{1}{1 + e^{-s}}
$$

In the implementation the derivative is evaluated from the activation itself:

$$
\sigma'(s) = s(1-s)
$$

### Forward pass

$$
\begin{aligned}
z   &= XW_1 \\
z_2 &= \sigma(z) \\
z_3 &= z_2W_2 \\
o   &= \sigma(z_3)
\end{aligned}
$$

### Backward pass

$$
\begin{aligned}
\delta_o    &= (y - o) \odot o \odot (1-o) \\
\delta_{z_2} &= \left(\delta_o W_2^\top\right) \odot z_2 \odot (1-z_2) \\
W_1         &\leftarrow W_1 + X^\top \delta_{z_2} \\
W_2         &\leftarrow W_2 + z_2^\top \delta_o
\end{aligned}
$$

Here $\odot$ denotes element-wise multiplication. The `+=` updates descend the
squared-error loss because the output error is defined as $y-o$.

## Training

- **Algorithm:** full-batch gradient descent
- **Epochs:** 1,000
- **Learning rate:** none — the raw summed gradient is applied directly, so the
  effective learning rate is `1.0`
- **Batch averaging:** none — gradients are summed across all 8 samples
- **Loss logged:** sum squared error (SSE)

$$
\mathcal{L}_{\mathrm{SSE}} = \sum_{i=1}^{8} \left(y_i - o_i\right)^2
$$

The reported loss is diagnostic only; it does not scale the weight updates. It
is computed **before** the update for that epoch, so the final CSV row is the
loss before the last update, while the final predictions come from the weights
after it.

## Files

| File | Purpose |
|---|---|
| `Neural_Network.py` | Training script and network implementation |
| `README.md` | Project documentation |
| `requirements.txt` | Python dependencies |
| `Sumsquaredlosslist.csv` | Per-epoch loss log (generated) |
| `weightsLayer1.txt` | Saved layer 1 weights (generated) |
| `weightsLayer2.txt` | Saved layer 2 weights (generated) |
| `.gitignore` | Ignored files and generated outputs |

`Sumsquaredlosslist.csv` is created at the start of each run with the header

```csv
Epoch,SumSquaredLoss
```

and one row per epoch.

## Setup

The project uses a local virtual environment and NumPy (pinned in
`requirements.txt`).

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe Neural_Network.py
```

The script will:

1. Print the inputs, targets, predictions, and SSE for every epoch.
2. Write `Sumsquaredlosslist.csv`.
3. Save both weight matrices.
4. Print a prediction for the sample `[0, 0, 1]`.
5. Print the trained network's predictions for all eight inputs.

## Expected result

Training typically pushes predictions close to their targets. Because the
weights are not seeded, the exact numbers change from run to run. A
representative final result:

| Input | Target | Predicted | Rounded |
|-------|:------:|----------:|:-------:|
| `[0, 0, 0]` | 1 | 0.9363 | 1 |
| `[0, 0, 1]` | 0 | 0.0399 | 0 |
| `[0, 1, 0]` | 0 | 0.0211 | 0 |
| `[0, 1, 1]` | 0 | 0.0453 | 0 |
| `[1, 0, 0]` | 0 | 0.0396 | 0 |
| `[1, 0, 1]` | 0 | 0.0141 | 0 |
| `[1, 1, 0]` | 0 | 0.0453 | 0 |
| `[1, 1, 1]` | 1 | 0.9284 | 1 |

Sample final metrics:

- Sum squared error: `0.017090`
- Mean squared error: `0.002136`
- Rounded accuracy: `8/8`

## Notes and limitations

- The console messages still say "XOR" in a few places; the learned function is
  the three-input XNOR truth table above.
- There is no explicit learning rate, momentum, weight decay, gradient
  clipping, or early stopping.
- There is no bias term, train/test split, or random seed.
- Large updates can saturate the sigmoids and stall learning; the current
  unnormalized gradients make the network sensitive to initialization.
- `xPredicted` is `[0, 0, 1]`, whose correct target is `0`.

## References

- John C. Shovic and Alan Simpson. *Python All-in-One For Dummies*.
  3rd ed., Wiley, 2024. ISBN 978-1-394-23615-2.
