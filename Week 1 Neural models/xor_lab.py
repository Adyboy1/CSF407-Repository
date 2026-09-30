"""CS F407 Week 1 lab: neural models (XOR, symmetry, activations, 3-class extension).

Run:  python xor_lab.py
Prints all results to the console.
"""
import sys

import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):  # absent in some notebook/IDE consoles
    sys.stdout.reconfigure(encoding="utf-8")
torch.set_default_dtype(torch.float64)

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y_BIN = torch.tensor([[0.], [1.], [1.], [0.]])
Y_CLS = torch.tensor([0, 1, 1, 2])  # 0: none, 1: disagree, 2: both

SEED, STEPS, LR = 0, 3000, 0.05
ACTS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}
results = {}


def make_net(act="tanh", out=1, hidden=2, seed=SEED):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, hidden), ACTS[act](), nn.Linear(hidden, out))


def train(net, loss_fn, y, steps=STEPS, lr=LR, track=None):
    """Full-batch Adam. Returns loss history; `track(step, net)` is called every step."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hist = []
    for t in range(steps):
        if track:
            track(t, net)
        opt.zero_grad()
        loss = loss_fn(net(X), y)          # forward pass + scalar loss
        loss.backward()                    # reverse-mode AD
        opt.step()                         # parameter update
        hist.append(loss.item())
    hist.append(loss_fn(net(X), y).item())
    return hist


def first_layer_grad(net, loss_fn, y):
    net.zero_grad()
    loss_fn(net(X), y).backward()
    return net[0].weight.grad.clone()


bce = nn.BCEWithLogitsLoss()
ce = nn.CrossEntropyLoss()


def evaluate_bin(net):
    with torch.no_grad():
        p = torch.sigmoid(net(X)).squeeze(1)
    return p, (p > 0.5).double()


# ---------------------------------------------------------------- Task 1 (linear model)
# Test: a single affine layer + sigmoid cannot fit XOR, so its loss should stay at ln 2 with outputs near 0.5.
print("=== Task 1: single affine + sigmoid (no hidden layer) ===")
torch.manual_seed(SEED)
lin = nn.Linear(2, 1)
h = train(lin, bce, Y_BIN, steps=5000)
p, lab = evaluate_bin(lin)
print("final loss %.4f (ln2 = %.4f)" % (h[-1], torch.log(torch.tensor(2.)).item()))
print("probs", [round(v, 3) for v in p.tolist()], "labels", lab.tolist())
print("accuracy: %d/4" % int((lab == Y_BIN.squeeze(1)).sum()))
results["linear"] = dict(final_loss=h[-1], probs=p.tolist(), correct=int((lab == Y_BIN.squeeze(1)).sum()))

# ---------------------------------------------------------------- Task 4A
# Test: the 2-2-1 tanh network learns XOR if the loss falls far below ln 2 and all four thresholded labels are correct.
print("\n=== Task 4A: 2-2-1 tanh network, BCEWithLogits, Adam lr=%g, %d steps ===" % (LR, STEPS))
net = make_net("tanh")
init_loss = bce(net(X), Y_BIN).item()
hist_tanh = train(net, bce, Y_BIN)
p, lab = evaluate_bin(net)
correct = bool((lab.unsqueeze(1) == Y_BIN).all())
print("initial loss %.4f  final loss %.6f" % (init_loss, hist_tanh[-1]))
print("probabilities", [round(v, 4) for v in p.tolist()])
print("labels", lab.tolist(), "-> all correct:", correct)
results["task4a"] = dict(initial_loss=init_loss, final_loss=hist_tanh[-1], probs=p.tolist(),
                         labels=lab.tolist(), all_correct=correct)

# ---------------------------------------------------------------- Task 4B
# Test: the first-layer gradient from backward() is dL/dW1, checked against the mean of per-example gradients
# and against central finite differences.
print("\n=== Task 4B: gradient checks (tanh net at initialisation, seed 0) ===")
net = make_net("tanh")  # untrained: the gradient is large enough for a meaningful comparison
g = first_layer_grad(net, bce, Y_BIN)
print("dL/dW1 from backward():\n", g)
# Test: with a mean loss, the batch gradient must equal the average of the four example-wise gradients.
# average of per-example gradients
per_ex = []
for i in range(4):
    net.zero_grad()
    bce(net(X[i:i + 1]), Y_BIN[i:i + 1]).backward()
    per_ex.append(net[0].weight.grad.clone())
avg = torch.stack(per_ex).mean(0)
print("max |grad - mean(per-example grads)| = %.2e" % (g - avg).abs().max().item())
# Test: autograd must agree with (L(w+eps) - L(w-eps)) / (2 eps) for every first-layer weight.
# finite differences
eps = 1e-6
fd = torch.zeros_like(g)
W = net[0].weight
for i in range(2):
    for j in range(2):
        with torch.no_grad():
            W[i, j] += eps
            lp = bce(net(X), Y_BIN).item()
            W[i, j] -= 2 * eps
            lm = bce(net(X), Y_BIN).item()
            W[i, j] += eps
        fd[i, j] = (lp - lm) / (2 * eps)
print("finite-difference dL/dW1:\n", fd)
fd_err = (g - fd).abs().max().item()
print("max |autograd - finite diff| = %.2e  (relative to max |grad|: %.2e)" % (fd_err, fd_err / g.abs().max().item()))
results["task4b"] = dict(grad=g.tolist(), per_example_mean_err=(g - avg).abs().max().item(),
                         fd_err=fd_err, fd_rel_err=fd_err / g.abs().max().item(), grad_norm=g.norm().item())

# ---------------------------------------------------------------- Task 4C
# Test: with identical initial parameters the two hidden units should stay identical at every step (symmetry),
# both for all-zero parameters and for a non-zero constant that gives non-zero gradients.
print("\n=== Task 4C: symmetry ===")
sym = {}
for name, val in [("zeros", 0.0), ("constant0.5", 0.5)]:
    n = make_net("tanh")
    with torch.no_grad():
        for prm in n.parameters():
            prm.fill_(val)
    rows = []

    def track(t, m):
        if t in (0, 1, 10, 100, 1000, 2999):
            d = (m[0].weight[0] - m[0].weight[1]).abs().max().item()
            db = (m[0].bias[0] - m[0].bias[1]).abs().item()
            rows.append((t, d, db))
    hh = train(n, bce, Y_BIN, track=track)
    p, lab = evaluate_bin(n)
    print("init =", name)
    for t, d, db in rows:
        print("  step %4d  max|W1[0]-W1[1]| = %.3e  |b1[0]-b1[1]| = %.3e" % (t, d, db))
    print("  final W1 =", n[0].weight.tolist(), " loss %.4f probs %s" % (hh[-1], [round(v, 3) for v in p.tolist()]))
    sym[name] = dict(rows=rows, final_loss=hh[-1], probs=p.tolist(), W1=n[0].weight.tolist())
results["symmetry"] = sym

# ---------------------------------------------------------------- Task 4D
# Test: changing only the hidden activation (same initial weights) shows how it affects the final loss, the 4/4
# check and the early first-layer gradient norm.
print("\n=== Task 4D: activation experiment (same seed / same init weights) ===")
act_res, hists = {}, {}
for a in ACTS:
    n = make_net(a)
    g0 = first_layer_grad(n, bce, Y_BIN).norm().item()  # step 0 = early step
    hh = train(n, bce, Y_BIN)
    p, lab = evaluate_bin(n)
    ok = bool((lab.unsqueeze(1) == Y_BIN).all())
    act_res[a] = dict(final_loss=hh[-1], all_correct=ok, grad_norm_step0=g0, probs=p.tolist())
    hists[a] = hh
    print("%-8s loss %.6f  4/4 correct: %-5s  ||grad W1||2 at step 0 = %.4f  probs %s"
          % (a, hh[-1], ok, g0, [round(v, 3) for v in p.tolist()]))
results["activations_seed0"] = act_res

# Test: repeating over 20 seeds shows how much of the seed-0 ranking is initialisation luck.
# robustness across seeds
NSEEDS = 20
multi = {}
for a in ACTS:
    succ, gn, fl = 0, [], []
    for s in range(NSEEDS):
        n = make_net(a, seed=s)
        gn.append(first_layer_grad(n, bce, Y_BIN).norm().item())
        hh = train(n, bce, Y_BIN)
        _, lab = evaluate_bin(n)
        succ += int((lab.unsqueeze(1) == Y_BIN).all())
        fl.append(hh[-1])
    multi[a] = dict(success=succ, n=NSEEDS, mean_grad0=sum(gn) / NSEEDS, mean_final_loss=sum(fl) / NSEEDS)
    print("%-8s seeds 0-%d: %d/%d solved XOR, mean step-0 grad norm %.4f, mean final loss %.4f"
          % (a, NSEEDS - 1, succ, NSEEDS, multi[a]["mean_grad0"], multi[a]["mean_final_loss"]))
results["activations_multiseed"] = multi

# Test: pre-activations and local derivatives separate a saturated sigmoid (large |z|) from an inactive ReLU (z < 0).
# saturation vs dead-ReLU diagnostic on the seed-0 nets
print("\nDiagnostic: hidden pre-activations / activations at initialisation (seed 0)")
diag = {}
for a in ("sigmoid", "relu"):
    n = make_net(a)
    with torch.no_grad():
        z = n[0](X)
        hid = n[1](z)
    if a == "sigmoid":
        deriv = hid * (1 - hid)
    else:
        deriv = (z > 0).double()
    print(a, "pre-activations:\n", z, "\n  local derivative f'(z):\n", deriv)
    diag[a] = dict(z=z.tolist(), deriv=deriv.tolist())
results["diagnostic"] = diag

# Test: after training, check whether the failed ReLU run has dead units or units that are active on every input.
# post-training diagnosis of the failed ReLU run: dead units vs saturation
print("\nDiagnostic after training (seed 0): pre-activations z and fraction of active/saturated units")
post = {}
for a in ACTS:
    n = make_net(a)
    train(n, bce, Y_BIN)
    with torch.no_grad():
        z = n[0](X)
    if a == "relu":
        frac = (z > 0).double().mean(0).tolist()
        print("relu  z =", z.tolist(), " fraction of inputs with z>0 per unit:", frac)
    elif a == "sigmoid":
        frac = ((torch.sigmoid(z) * (1 - torch.sigmoid(z))) < 0.05).double().mean().item()
        print("sigmoid  min f'(z)=%.4f  max=%.4f" % ((torch.sigmoid(z)*(1-torch.sigmoid(z))).min(), (torch.sigmoid(z)*(1-torch.sigmoid(z))).max()))
    post[a] = z.tolist()
results["post_training_z"] = post

# ---------------------------------------------------------------- Task 5
# Test: with three logits and cross-entropy the same hidden layer should classify all four inputs into 0/1/2.
print("\n=== Task 5: 3-class extension (2-2-3 tanh, CrossEntropyLoss) ===")
net3 = make_net("tanh", out=3)  # same 2 hidden units; only the output layer and loss change
print("final weight matrix shape:", tuple(net3[2].weight.shape), " logits per example:", net3(X).shape[1])
h3 = train(net3, ce, Y_CLS)
with torch.no_grad():
    logits = net3(X)
    P = torch.softmax(logits, dim=1)
pred = P.argmax(1)
print("initial->final loss: %.4f -> %.6f" % (h3[0], h3[-1]))
for i in range(4):
    print("  x=%s  p=%s  pred=%d  target=%d" % (X[i].tolist(), [round(v, 4) for v in P[i].tolist()],
                                              pred[i].item(), Y_CLS[i].item()))
print("all correct:", bool((pred == Y_CLS).all()))
# Test: the softmax probabilities of one example must sum to 1.
i = 1
print("softmax vector for x=%s: %s, sum = %.12f" % (X[i].tolist(), P[i].tolist(), P[i].sum().item()))

# Test: the gradient of the mean cross-entropy with respect to the logits must equal (p - y) / N.
# logit gradient == p - y (per example, and mean-loss gives (p-y)/N)
lg = net3(X).detach().clone().requires_grad_(True)
ce(lg, Y_CLS).backward()
onehot = torch.nn.functional.one_hot(Y_CLS, 3).double()
err = (lg.grad - (torch.softmax(lg.detach(), 1) - onehot) / 4).abs().max().item()
print("max |dL/dlogits - (p-y)/N| = %.2e   (1/N from the mean over N=4 examples)" % err)

# Test: adding a constant to all logits must leave softmax unchanged, and naive exp overflows on large logits
# while subtracting the maximum first does not.
# shift invariance / stability
z = logits[i]
naive = torch.exp(z + 100) / torch.exp(z + 100).sum()
stable = torch.softmax(z + 100, 0)
print("logits + 100 -> softmax:", stable.tolist(), " max diff vs original: %.2e" % (stable - P[i]).abs().max().item())
big = torch.tensor([1000., 1001., 1002.])
print("naive exp on [1000,1001,1002] :", (torch.exp(big) / torch.exp(big).sum()).tolist())
print("subtract-max softmax          :", torch.softmax(big, 0).tolist())
results["task5"] = dict(final_loss=h3[-1], probs=P.tolist(), pred=pred.tolist(), sum_example=P[i].sum().item(),
                        grad_err=err, W_shape=list(net3[2].weight.shape),
                        shift_diff=(stable - P[i]).abs().max().item(),
                        naive_big=(torch.exp(big) / torch.exp(big).sum()).tolist())
