"""Week 8 - Bayesian networks and autoregressive language models.

First-order (bigram) and second-order (trigram) models built from counts using
only ordinary Python data structures.  The code was drafted by an LLM (Claude)
from a behavioural specification; the review notes and modifications are in the
report.  Running this file prints everything the report uses (run_output.txt).
"""
import random
from collections import defaultdict

START, END = "<START>", "<END>"
MAX_LEN = 20  # safety cap on generated tokens (greedy decoding can cycle)

DATA = """the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug"""


def tokenise(text):
    return [line.lower().split() for line in text.strip().splitlines()]


class NGramLM:
    """order=1: P(X_t | X_{t-1});  order=2: P(X_t | X_{t-2}, X_{t-1}).

    Sentences are padded with `order` START tokens on the left and one END on
    the right, so P(X1) is just P(X1 | START) (order 1) or P(X1 | START,START)
    (order 2).
    """

    def __init__(self, order):
        self.order = order
        self.counts = defaultdict(lambda: defaultdict(int))  # context -> token -> C
        self.probs = {}
        self.vocab = set()

    def fit(self, sentences):
        for s in sentences:
            toks = [START] * self.order + s + [END]
            self.vocab.update(s)
            for i in range(self.order, len(toks)):
                self.counts[tuple(toks[i - self.order:i])][toks[i]] += 1
        # P(w | ctx) = C(ctx, w) / sum_k C(ctx, k)
        for ctx, nxt in self.counts.items():
            total = sum(nxt.values())
            self.probs[ctx] = {w: c / total for w, c in nxt.items()}
        return self

    def ctx(self, *words):
        return tuple(words)

    def dist(self, ctx):
        return self.probs.get(ctx, {})

    def predict(self, ctx):
        """arg max_w P(w | ctx); ties broken alphabetically. None if unseen."""
        d = self.dist(ctx)
        if not d:
            return None
        return max(sorted(d), key=lambda w: d[w])

    def sample(self, ctx, rng):
        d = self.dist(ctx)
        if not d:
            return None
        words = sorted(d)
        return rng.choices(words, weights=[d[w] for w in words])[0]

    def generate(self, mode, rng=None, prefix=()):
        """Return (tokens, status); status in {'END','UNSEEN','MAXLEN'}.

        `prefix` is an optional list of tokens to continue from (default: start of sentence).
        """
        out = list(prefix)
        ctx = tuple(([START] * self.order + out)[-self.order:])
        while len(out) < MAX_LEN:
            w = self.predict(ctx) if mode == "greedy" else self.sample(ctx, rng)
            if w is None:
                return out, "UNSEEN"
            if w == END:
                return out, "END"
            out.append(w)
            ctx = ctx[1:] + (w,)
        return out, "MAXLEN"

    # ---- statistics used in the comparison ------------------------------
    def n_params(self):
        """Distinct non-zero conditional probabilities (context, next) stored."""
        return sum(len(d) for d in self.probs.values())

    def n_contexts(self):
        return len(self.probs)

    def full_table_size(self):
        """Rows in the full CPT over the vocabulary (incl. START as a context)."""
        return (len(self.vocab) + 1) ** self.order

    def zero_entries_in_observed_rows(self):
        nexts = len(self.vocab) + 1  # every word plus END
        return sum(nexts - len(d) for d in self.probs.values())


def check_normalisation(model, tol=1e-9):
    bad = []
    for ctx, d in sorted(model.probs.items()):
        total = sum(d.values())
        if abs(total - 1.0) > tol:
            bad.append((ctx, total))
    return bad


def fmt(d):
    return ", ".join(f"{w}: {p:.3f}" for w, p in sorted(d.items(), key=lambda x: (-x[1], x[0])))


def show_cpt(model, ctx):
    d = model.dist(ctx)
    name = " ".join(ctx)
    if not d:
        print(f"  P(next | {name}) = <no observed transitions>")
    else:
        print(f"  P(next | {name}) = {{{fmt(d)}}}")


def distinct(sents):
    return len({" ".join(s) for s in sents})


def main():
    data = tokenise(DATA)
    m1 = NGramLM(1).fit(data)
    m2 = NGramLM(2).fit(data)
    print("Vocabulary:", ", ".join(sorted(m1.vocab)), f"({len(m1.vocab)} words)")

    print("\n=== First-order CPTs (Q3) ===")
    for w in [START, "the", "cat", "dog", "sat", "ran", "on", "to", "mat", "rug", "park"]:
        show_cpt(m1, (w,))
    print("\nZero-probability transitions for the requested words (next in vocab + END):")
    nexts = sorted(m1.vocab | {END})
    for w in ["the", "cat", "dog", "sat", "ran"]:
        z = [v for v in nexts if v not in m1.dist((w,))]
        print(f"  after '{w}': {', '.join(z)}")

    print("\n=== Second-order CPTs (selected contexts) ===")
    for c in [(START, START), (START, "the"), ("the", "cat"), ("the", "dog"),
              ("cat", "sat"), ("dog", "ran"), ("on", "the"), ("to", "the")]:
        show_cpt(m2, c)
    print("  Unseen context example:")
    show_cpt(m2, ("cat", "park"))

    print("\n=== Normalisation tests: sum_v P(v | ctx) ===")
    for name, m in (("first-order", m1), ("second-order", m2)):
        print(f"{name}:")
        for ctx, d in sorted(m.probs.items()):
            print(f"  {' '.join(ctx):<18} {sum(d.values()):.6f}")
        bad = check_normalisation(m)
        print(f"  -> {len(m.probs)} contexts checked, failures: {len(bad)}")
    # sanity check that the test can fail: corrupt one row
    m1.probs[("the",)]["cat"] -= 0.13
    print("Negative control (subtract 0.13 from P(cat|the)):",
          [(' '.join(c), round(t, 3)) for c, t in check_normalisation(m1)])
    m1.probs[("the",)]["cat"] += 0.13

    print("\n=== Next-word prediction, first-order (Section 11) ===")
    for w in ["the", "cat", "dog", "sat", "ran", "on", "to"]:
        show_cpt(m1, (w,))
        print(f"    argmax -> {m1.predict((w,))}")

    print("\n=== Next-word prediction, second-order ===")
    for c in [("the", "cat"), ("the", "dog"), ("sat", "on"), ("on", "the"), ("ran", "to"), ("to", "the")]:
        show_cpt(m2, c)
        print(f"    argmax -> {m2.predict(c)}")

    print("\n=== 20 sampled sentences, first-order (seed 407) ===")
    rng = random.Random(407)
    s1 = []
    for i in range(20):
        t, st = m1.generate("sample", rng)
        s1.append(t)
        print(f"  {i + 1:2d}. {' '.join(t)}  [{st}]")
    print(f"  distinct: {distinct(s1)}/20")

    print("\n=== 20 sampled sentences, second-order (seed 407) ===")
    rng = random.Random(407)
    s2 = []
    for i in range(20):
        t, st = m2.generate("sample", rng)
        s2.append(t)
        print(f"  {i + 1:2d}. {' '.join(t)}  [{st}]")
    print(f"  distinct: {distinct(s2)}/20")

    print("\n=== Greedy vs sampling (Part X) ===")
    for name, m in (("first-order", m1), ("second-order", m2)):
        outs = []
        for i in range(5):
            t, st = m.generate("greedy")
            outs.append(t)
            print(f"{name} Mode A greedy {i + 1}: {' '.join(t)}  [{st}]")
        print(f"  distinct in 5 greedy runs: {distinct(outs)}")
        rng = random.Random(1)
        outs = []
        for i in range(5):
            t, st = m.generate("sample", rng)
            outs.append(t)
            print(f"{name} Mode B sample {i + 1}: {' '.join(t)}  [{st}]")
        print(f"  distinct in 5 samples: {distinct(outs)}")

    print("\n=== Model comparison (Part XIII) ===")
    train = {" ".join(s) for s in data}
    print(f"{'':44}{'first-order':>12}{'second-order':>14}")
    rows = [
        ("Observed contexts (CPT rows with data)", m1.n_contexts(), m2.n_contexts()),
        ("Non-zero parameters (context,next) pairs", m1.n_params(), m2.n_params()),
        ("Rows in full CPT over vocabulary", m1.full_table_size(), m2.full_table_size()),
        ("Unobserved contexts (rows with no data)", m1.full_table_size() - m1.n_contexts(),
         m2.full_table_size() - m2.n_contexts()),
        ("Zero entries inside observed rows", m1.zero_entries_in_observed_rows(),
         m2.zero_entries_in_observed_rows()),
        ("Distinct sentences in 20 samples", distinct(s1), distinct(s2)),
        ("Samples identical to a training sentence", sum(" ".join(s) in train for s in s1),
         sum(" ".join(s) in train for s in s2)),
    ]
    for label, a, b in rows:
        print(f"{label:44}{a:>12}{b:>14}")
    novel1 = sorted({" ".join(s) for s in s1} - train)
    novel2 = sorted({" ".join(s) for s in s2} - train)
    print("Novel (not in training data) first-order:", novel1)
    print("Novel (not in training data) second-order:", novel2)
    print("Note: the full-CPT row counts include <START> as a context word; the number of "
          "rows that can actually be reached is smaller.")

    print("\n=== Unseen-context behaviour (Q7) ===")
    print("m1.predict(('park',)) ->", m1.predict(("park",)), "(park is only followed by END)")
    print("m1.predict(('banana',)) ->", m1.predict(("banana",)))
    for name, m, prefix in (("first-order", m1, ["banana"]), ("second-order", m2, ["cat", "park"])):
        for mode in ("greedy", "sample"):
            t, st = m.generate(mode, random.Random(407), prefix=prefix)
            print(f"{name} generate({mode!r}) continuing from '{' '.join(prefix)}' -> tokens {t}, status {st}")
            assert st == "UNSEEN" and t == prefix


if __name__ == "__main__":
    main()
