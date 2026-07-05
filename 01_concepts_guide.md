# One-Day Deep Learning Crash Course

Your concept outline is solid — this guide keeps your 4-block structure, tightens it into a schedule, interleaves the coding exercises, and adds the verbal questions you're most likely to be asked. Remember the framing from the email: **this is a level check, not an exam.** "I haven't used X yet, but my understanding is..." is a perfectly good answer.

## Day Schedule

| Time | Block | Activity |
|---|---|---|
| 09:00–10:30 | Setup + Block 1 | Run `00_setup_mac.md`, then Architecture & Activations |
| 10:30–11:30 | Exercise 1 | Tensors & autograd (`exercises/ex1_tensors_autograd.py`) |
| 11:30–13:00 | Block 2 | Training mechanics & optimization |
| 13:00–13:45 | Lunch | Step away from the screen |
| 13:45–15:30 | Exercise 2 | MLP on MNIST — **the core exercise, do this closed-book if possible** |
| 15:30–16:30 | Block 3 | Regularization & overfitting |
| 16:30–17:30 | Exercise 3 | CNN (`exercises/ex3_cnn.py`) |
| 17:30–18:30 | Block 4 | Debugging scenarios + verbal Q&A rehearsal (below) |
| 18:30–19:00 | Redo Ex 2 | From a blank file, no notes. This is your dress rehearsal. |

---

## Block 1: Architecture & Activation Functions

Your notes cover this well. Key additions:

**Activations — one-line answers to have ready:**

- *Why nonlinearity?* Stacked linear layers collapse into one linear map; activations are what let depth mean anything.
- *Why ReLU over sigmoid?* Cheap, non-saturating for x>0 → no vanishing gradient there. Weakness: dying ReLU (fix: LeakyReLU).
- *Where does softmax go?* Last layer of multi-class classification — **but in PyTorch you usually don't add it yourself**, because `nn.CrossEntropyLoss` expects raw logits and applies log-softmax internally. Knowing this is a strong signal you've actually used PyTorch.

**CNN vs Transformer:** your "inductive bias vs. data hunger" framing is exactly right. Add one sentence on attention if asked: *each token computes query/key/value vectors; attention weights are softmax(QKᵀ/√d), used to take a weighted sum of values — so every position can attend to every other position in one step.*

**Also be ready for:** "What does a convolution actually compute?" — a filter (e.g. 3×3×in_channels) slides over the input, dot-product at each location, producing one feature map per filter. Parameters are shared across positions → far fewer weights than a dense layer, and translation equivariance for free.

---

## Block 2: Training Mechanics & Optimization

Your notes are good. The must-know PyTorch mapping:

| Concept | PyTorch |
|---|---|
| MSE (regression) | `nn.MSELoss()` |
| Binary cross-entropy | `nn.BCEWithLogitsLoss()` (takes logits — numerically stable) |
| Multi-class cross-entropy | `nn.CrossEntropyLoss()` (takes logits + integer class labels, NOT one-hot) |
| SGD / Adam | `torch.optim.SGD(params, lr=...)` / `torch.optim.Adam(params, lr=1e-3)` |

**The 5-line training loop — memorize this cold.** If you internalize one thing today, make it this:

```python
for x, y in loader:
    optimizer.zero_grad()      # 1. clear old gradients
    out = model(x)             # 2. forward pass
    loss = criterion(out, y)   # 3. compute loss
    loss.backward()            # 4. backprop (chain rule)
    optimizer.step()           # 5. update weights
```

Classic verbal question: *"What happens if you forget `zero_grad()`?"* → gradients **accumulate** across batches, so updates use stale summed gradients and training misbehaves.

**Backprop in one sentence:** the chain rule applied efficiently from loss back to every weight, reusing intermediate results (that's why the forward pass stores activations).

**Vanishing/exploding:** your notes are complete. Anchor the fixes: vanishing → ReLU, residual connections, BatchNorm, better init; exploding → gradient clipping (`torch.nn.utils.clip_grad_norm_`), lower LR, He/Xavier init.

---

## Block 3: Regularization

Your table is good. Two additions interviewers like:

- **Train vs eval mode:** `model.train()` enables dropout and per-batch BatchNorm statistics; `model.eval()` disables dropout and uses running statistics. Forgetting `model.eval()` at test time is a classic bug — mentioning it unprompted looks great.
- **Weight decay in code:** just `torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)`.
- **Early stopping** is regularization too: watch validation loss, keep the best checkpoint, stop when it stops improving.

---

## Block 4: Debugging Scenarios

Your three scenarios + action plans are exactly the right preparation. Add one more:

**Scenario 4: "Training accuracy is stuck at 10% on MNIST (10 classes)."**
Diagnosis: the model is at chance level — something structural is broken, not just slow learning. Check: labels misaligned with inputs (shuffling bug), loss/label format mismatch (e.g. one-hot into `CrossEntropyLoss`), learning rate absurdly high (diverged immediately), or the optimizer wasn't given `model.parameters()`.

---

## Verbal Q&A Rehearsal (say answers OUT LOUD)

Background questions to expect first — prepare 30–60 second answers:

1. Tell me about your background — math, programming, any ML exposure.
2. What deep learning have you done? (courses, videos, projects — be honest and specific)
3. Why NYCU / what do you want to work on with Prof. Rini?
4. How comfortable are you with Python? With linear algebra and calculus?

Technical spot-checks:

5. Why do we need activation functions?
6. Difference between MSE and cross-entropy — when do you use each?
7. What does `loss.backward()` do? What does `optimizer.step()` do?
8. What is overfitting and how do you detect it? Name three fixes.
9. What's a convolution and why use it for images instead of a dense layer?
10. What's the difference between a training set, validation set, and test set?

(#10 answer: train = fit weights; validation = tune hyperparameters / early stopping; test = touched once, at the end, for an unbiased estimate.)

---

## During the live exercise

- **Think out loud.** They're assessing your starting point, not perfection. Narrating "now I need a loss function; it's classification so cross-entropy" earns more than silent typing.
- If stuck on syntax, say the concept and write your best guess. Concept > syntax.
- It's fine to check official PyTorch docs if allowed — ask "may I look at the docs?" rather than freezing.
- Run your code early and often; don't write 40 lines before the first run.
