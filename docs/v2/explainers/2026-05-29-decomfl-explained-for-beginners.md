# DeComFL, Explained for Beginners

**Date:** 2026-05-29
**Audience:** A first-year computer-science student who has never heard of federated learning. Curiosity assumed, prior knowledge not.
**Companion spec (the technical source of truth):** [`docs/v2/specs/2026-05-29-decomfl-correctness-design.md`](../specs/2026-05-29-decomfl-correctness-design.md)

This document explains, in plain language, what our platform does, the clever trick at its heart, three real bugs we found in it, and how we know our fixes actually worked. There is a glossary at the very end that defines every term — if a word looks unfamiliar, it is probably down there.

---

## 1. What is Federated Learning?

Imagine you want to build a really good autocorrect keyboard. To make it smart, you need to train it on how thousands of real people type. The obvious way is to collect everyone's text messages onto one big server and learn from them there. But that is a privacy nightmare — nobody wants to ship their private messages to a company's data center.

**FL (Federated Learning)** is the idea that you can train a shared model *without ever collecting everyone's raw data in one place*. The data stays on each person's phone. Only the *lessons learned* travel.

Here is the everyday analogy. Think of a cooking school where the head chef wants to perfect one shared recipe, but the students are spread across many cities and none of them will mail their groceries to the school:

```
   Head Chef (the SERVER)
   "Here is today's recipe. Try it at home, tell me how it went."
            |
   sends recipe  v
   +--------------+--------------+--------------+
   |              |              |              |
 Student A     Student B     Student C     ... (the CLIENTS)
 cooks at      cooks at      cooks at
 home with     home with     home with
 own kitchen   own kitchen   own kitchen
   |              |              |
   "too salty"   "too bland"   "perfect"   <- feedback only, NOT the groceries
   +--------------+--------------+--------------+
            ^
   collects feedback |
   Head Chef adjusts the recipe, sends the new version out again.
   Repeat for many "rounds" until the recipe is great.
```

Each student keeps their own ingredients private (your data never leaves your device). They only send back *feedback* about the recipe. The chef merges all the feedback into one improved recipe and sends it back out. One full cycle of "send recipe out, get feedback in, improve" is called a **round**. After enough rounds, everyone shares one excellent recipe — and no private data ever moved.

In real terms:
- The **server** holds the shared **model** (the recipe).
- The **clients** are phones/laptops, each with their own private data (the home kitchen).
- Each round, the server sends the model out, clients improve it locally, and send back only their improvements.

That is FL in one picture.

---

## 2. The Problem: Big Models Are Heavy to Mail

The cooking analogy hides a nasty practical issue. In classic FL (the standard method is called **FedAvg**, short for Federated Averaging), the "feedback" each client sends back is *a whole copy of the improved model*.

For a small model that is fine. For a big modern model it is a disaster. A model is just a giant list of numbers called **parameters**, and big language models have a *lot* of them. (Sizes below are in **MB (Megabytes)** and **GB (Gigabytes)** — a GB is roughly 1,000 MB.)

| Model | Parameters | Size to send, once | Cost per client, per round |
|---|---|---|---|
| Small image model | ~1.2 million | ~4.8 MB | basically free |
| GPT-2 Small | ~117 million | ~468 MB | 468 MB up + 468 MB down |
| LLaMA-7B | ~7 billion | ~28 GB | 28 GB up + 28 GB down |

*(Source: in-repo wiki [`docs/wikis/framework/06_decomfl.md:26-35`](../../wikis/framework/06_decomfl.md).)*

Now remember FL runs for *many rounds*. Asking a phone on mobile data to upload 28 GB every single round, dozens of times, is simply not going to happen. The model's size — its **dimension**, written `d`, meaning "how many numbers are in it" — is the bottleneck. This is exactly the wall that **DeComFL** was invented to climb over.

---

## 3. The Big Idea: Shared Dice

**DeComFL (Decomposed Federated Learning)** is the part of our platform built specifically to solve the "the model is too big to mail" problem. It is the platform's paper-backed differentiator, developed at RIT (Rochester Institute of Technology) by the group of Professor Haibo Yang. For large language models it can use on the order of **a million times less bandwidth** than FedAvg *(source: in-repo wiki [`docs/wikis/framework/06_decomfl.md`](../../wikis/framework/06_decomfl.md))*.

How can sending feedback possibly get a million times smaller? The trick is one beautiful insight about randomness.

### Shared seed = shared dice

Computers can't make *truly* random numbers; they make *pseudo-random* numbers using a **RNG (Random Number Generator)**. A RNG takes one starting number — called a **seed** — and from it produces a whole stream of "random-looking" numbers. The crucial property: **the same seed always produces the exact same stream of numbers.**

Think of two friends with two identical, magic dice. If they both whisper the same secret word ("seed") to their dice before rolling, the dice are guaranteed to land on the same sequence of numbers — every time. So instead of one friend rolling and then *describing every roll* to the other (expensive), they just agree on the secret word and both roll privately, knowing they got identical results (nearly free).

DeComFL uses this. The server and every client share the same seed. So they can each *independently* generate the exact same giant random vector — a list of `d` random numbers, one per parameter. We call that vector `z` (just a name for "a random nudge direction in the model"). Nobody has to *send* `z`; everyone *regenerates* it from the shared seed.

### What actually gets sent: just a score

So if the big random nudge `z` is free to regenerate, what does a client actually mail back? Just one tiny number per nudge — a score saying "nudging the model in direction `z` made it a little better / a little worse, by this much." That score is called `g`.

Here is the conceptual move, with no calculus required:

```
1. Server picks a seed and shares it.
2. Server and client BOTH regenerate the same random nudge z from that seed.
3. Client tries the nudge: it tweaks the model a tiny bit in direction z,
   and checks "did my loss (my error) go down or up, and by how much?"
       loss = how wrong the model is. Lower is better.
   That single result is the score g.
4. Client mails back ONLY g  (one small number), not z, not the model.
5. Server already knows z (same seed!), so it reconstructs the full update
   as  g times z  and improves the model.
```

This is called **ZO (Zeroth-Order) gradient estimation**. "Gradient" is just the math word for "which way should I nudge the model to reduce error." Normally you compute it with calculus (taking a derivative — the "first-order" way). Zeroth-order skips all the calculus: you literally *try* a direction and *measure* whether things got better. "Zeroth-order" simply means "we never took a derivative; we just poked it and looked."

Because each client only sends a handful of tiny scores `g` instead of millions of parameters, the amount of data on the wire no longer depends on how big the model is. *That* is where the enormous bandwidth savings come from.

### A few symbols you'll see (each explained once)

| Symbol | Plain meaning |
|---|---|
| `d` | how many numbers are in the model (its size/dimension) |
| `z` | a random "nudge direction" — a list of `d` random numbers, regenerated from a seed |
| `g` | the score: a single number saying how much that nudge helped or hurt |
| `P` | how many different nudges we try per step (default in our code: 10) |
| `K` | how many small training steps a client does locally per round |
| `eta`, written η | the **learning rate** — how *big* a step we take when we improve the model |
| `N` | how many clients sent feedback this round |

The only formula worth remembering is the gentle one: **new model = old model − (step size) × (combined nudge)**. We move the model a little bit, in the averaged-out direction the clients' scores pointed.

---

## 4. The Three Bugs (each as a short story)

DeComFL is powerful but subtle: the server and the clients have to stay *perfectly in sync*, because they are both relying on regenerating the *same* random numbers and taking the *same* size steps. Three things were quietly breaking that sync. Here is each one as a story.

### Bug 1 — The server stepped ten times too far

**The story.** Picture walking toward a target. The clients carefully figured out: "take a step of *this* size." But the server, when it applied that step, multiplied it by ten — so it lunged ten times too far past where it meant to land. Every round, it overshot.

**What was really happening.** In the code the server first correctly shrinks the step by dividing by `P` (the number of nudges, 10):

> [`framework/src/fedlearn/server/decomfl_strategy.py:197`](../../../framework/src/fedlearn/server/decomfl_strategy.py) — `delta = delta / (num_clients * self.P)`

...and then, on the very next line, multiplies it right back out again:

> [`framework/src/fedlearn/server/decomfl_strategy.py:200`](../../../framework/src/fedlearn/server/decomfl_strategy.py) — `x_current = x_current - self.eta * delta * self.P`

That trailing `* self.P` cancels the `/ self.P` from the line above, so the server's real step is `P` times (10 times) too big. The *clients* do this correctly — they keep the divide-by-`P`:

> [`framework/src/fedlearn/client/decomfl_client.py:208`](../../../framework/src/fedlearn/client/decomfl_client.py) — `step_update = (eta / P) * delta`

So the server and the clients literally walk along different paths. That matters enormously, because DeComFL's whole correctness guarantee is that a client who *missed* some rounds can perfectly *replay* the server's steps to catch up. If the server steps differently than everyone expects, that replay no longer lands in the right place.

### Bug 2 — The dice didn't match because they came from different dice sets

**The story.** Remember the magic shared dice from Section 3? The whole trick falls apart if the two friends are secretly using *different* dice. Same secret word, but different dice, gives different rolls. That is exactly what happened: the server was rolling on a **GPU (Graphics Processing Unit)** while a phone rolled on its **CPU (Central Processing Unit)** — and it turns out the "same seed" produces a *different* stream of random numbers on a GPU than on a CPU. Same secret word, different dice set, different rolls.

**What was really happening.** Both the server and the client generate their random nudge `z` directly on whatever hardware they happen to have:

> Server: [`framework/src/fedlearn/server/decomfl_strategy.py:77`](../../../framework/src/fedlearn/server/decomfl_strategy.py) picks `cuda` (GPU) if available else `cpu`, and [`:210-219`](../../../framework/src/fedlearn/server/decomfl_strategy.py) generates `z` on that device.
> Client: [`framework/src/fedlearn/estimators/zeroth_order.py:45-48`](../../../framework/src/fedlearn/estimators/zeroth_order.py) does the same on *its* device.

Seeded random number generation is **not** guaranteed identical across CPU, CUDA (NVIDIA GPU), and MPS (Apple GPU). So a GPU server and a CPU phone, *using the same seed*, reconstruct *different* `z`. Then `g × z` is computed against the wrong `z`, and the combined update is garbage — silently. Nothing crashes; the model just quietly fails to learn on any mixed-hardware fleet.

### Bug 3 — We packed the box one way and unpacked it expecting another

**The story.** Imagine shipping a model in a box. The packing crew puts the contents in loose, with no label. The unpacking crew, at the other end, reaches in and asks "where's the item labeled `parameters`? where's the `num_examples` note?" — finds no labels, and the whole process crashes. The two crews never agreed on a layout.

**What was really happening.** When a model is too big to send in one piece, it gets split into **chunks**. The packing function saves the model as a *bare* bag of numbers:

> [`framework/src/fedlearn/communication/serializer.py:97`](../../../framework/src/fedlearn/communication/serializer.py) — `torch.save(params, buffer)`

But the unpacking function tries to pull *labeled* items out of it:

> [`framework/src/fedlearn/communication/serializer.py:155`](../../../framework/src/fedlearn/communication/serializer.py) — `return model_data['parameters'], model_data['num_examples']`

Since the saved box has no `parameters` or `num_examples` labels, the unpack throws a **KeyError** (Python's "that label doesn't exist" error). The catch: this *only* happens to models big enough to be chunked — which is *every* large language model, i.e. exactly the models DeComFL exists to serve. The reassembly happens at [`framework/src/fedlearn/server/grpc_servicer.py:213`](../../../framework/src/fedlearn/server/grpc_servicer.py). Three tests in the suite were already failing on this.

---

## 5. The Fixes — and How We *Know* They Work

We follow **TDD (Test-Driven Development)**: we write the automatic checks *first*, watch them fail, then fix the code until they pass. A **test** here just means a small program that pokes our code with known inputs and shouts if the answer is wrong. Because the test runs by itself, it keeps protecting us forever — if anyone re-introduces a bug later, the test goes red again. All tests run with one command: `cd framework && pytest`.

### The fixes in plain terms

| Bug | The fix, in one sentence |
|---|---|
| **Bug 1** (stepped 10× too far) | Delete the stray `* self.P` on line 200 so the server takes the same correctly-sized step the clients do. |
| **Bug 2** (mismatched dice) | Create one shared helper, `canonical_perturbation`, that *always* rolls the dice on the CPU, then copies the result to the GPU if needed — so everyone, on every device, gets identical numbers from a seed. Both server and client now call this one helper, which also deletes the duplicated code that let them drift apart. |
| **Bug 3** (mismatched box) | Pack the box *with the same labels the unpacker expects*: save `{'parameters': params, 'num_examples': num_examples}` so unpacking finds exactly what it asks for. |

We also folded in two small "while the patient is open" improvements the team chose:
- **C-1 (faster, same answer):** the server was regenerating the big random `z` once *per client*, even though `z` doesn't depend on the client at all. We generate it once and reuse it — far fewer operations, mathematically identical result.
- **C-2 (don't grow forever):** the server's memory of past rounds (its history) grew without limit. We now forget rounds that no client could still need, with a configurable cap, so memory stays bounded.

And two tiny "blocking" fixes that just unstuck the tests: correcting an old test that treated the round-keyed history as if it were a simple list (B-1), and stopping the code from scribbling on the *whole program's* shared randomness (B-2) by giving DeComFL its own private random generators.

### The five checks, and what each one proves

These five automatic tests are the *contract* for "done." Here is what each one guarantees, in one friendly sentence:

| Check | In plain English, what it proves |
|---|---|
| **T1** | A client who trained every round and a client who *skipped* every round and replayed the history end up at the *same* model — proving the server and clients now walk the same path (catches Bug 1). This is our most important alarm; it fails today and must turn green. |
| **T2** | The shared dice helper produces the exact expected numbers from a fixed list of known answers, *and* the server's `z` matches the client's `z` for the same seed even on different hardware (catches Bug 2). |
| **T3** | A model packed into chunks and then unpacked comes back *identical* — for a small model, a multi-chunk big model, and a transformer-shaped model — flipping the three currently-failing tests to passing (catches Bug 3). |
| **T4** | The new faster aggregation produces a model identical to the slow, obvious version (with Bug 1 already fixed), proving the speed-up changed nothing about the answer (covers cleanup C-1). |
| **T5** | A client that missed several rounds reconnects and rebuilds correctly, while the server's history stays bounded over many rounds (covers cleanup C-2). |

When all five are green, we have evidence — not just hope — that DeComFL is correct and stays correct.

---

## 6. Glossary

Every term and abbreviation used above, defined plainly.

- **Aggregation** — the server's step of combining all clients' feedback into one updated model.
- **Chunk** — a small piece of a large model, split up so it can be sent over the network in parts.
- **Client** — a participant device (phone, laptop) that holds private data and trains locally. The "students" in the cooking analogy.
- **CPU (Central Processing Unit)** — the general-purpose processor in every computer.
- **CUDA** — NVIDIA's technology for running computations on their GPUs.
- **`d` (dimension)** — how many numbers make up a model; its size.
- **DeComFL (Decomposed Federated Learning)** — the bandwidth-saving FL method at the heart of our platform; sends tiny scores instead of whole models.
- **DP (Differential Privacy)** — a formal technique for adding noise so individual data can't be reverse-engineered. *(Mentioned for completeness; not part of these fixes — it is explicitly out of scope in the spec.)*
- **`eta` / η (learning rate)** — how big a step we take each time we improve the model.
- **FedAvg (Federated Averaging)** — the classic FL method; clients send back whole model copies, which the server averages.
- **FL (Federated Learning)** — training a shared model without collecting everyone's raw data in one place.
- **`g` (gradient scalar / score)** — the single number a client sends back, saying how much a nudge helped or hurt.
- **Gradient** — the direction that reduces the model's error; "which way to nudge."
- **GPU (Graphics Processing Unit)** — a processor with many cores, great at the heavy math of training models.
- **gRPC** — the network protocol our server and clients use to talk to each other.
- **`K`** — how many small local training steps a client does per round.
- **KeyError** — Python's error for "you asked for a label/key that doesn't exist."
- **Loss** — a number measuring how wrong the model is right now; lower is better.
- **MB (Megabyte) / GB (Gigabyte)** — units of data size; one GB is roughly 1,000 MB.
- **Model** — the thing being trained; under the hood, a giant list of numbers (parameters). The "recipe."
- **MPS** — Apple's technology for running computations on the GPU in Apple Silicon chips.
- **`N`** — the number of clients that sent feedback in a given round.
- **`num_examples`** — how many data samples a client trained on; used to weight its feedback.
- **`P`** — how many different random nudges are tried per step (our default: 10).
- **Parameter** — one of the many numbers that make up a model.
- **Perturbation** — the technical word for the random nudge `z`.
- **RIT (Rochester Institute of Technology)** — the university where DeComFL was developed.
- **RNG (Random Number Generator)** — code that turns a seed into a stream of random-looking numbers.
- **Round** — one full cycle of: server sends model out, clients improve it, server collects feedback and updates.
- **Seed** — the starting number for a RNG; the same seed always yields the same number stream. The "secret word" for the magic dice.
- **Serializer** — code that packs a model into bytes for sending (and unpacks it on arrival).
- **Server** — the central coordinator holding the shared model. The "head chef."
- **TDD (Test-Driven Development)** — writing the automatic checks first, then writing code until they pass.
- **Test** — a small self-running program that verifies code behaves correctly and shouts if it doesn't.
- **`z` (perturbation)** — the random "nudge direction," a list of `d` random numbers regenerated from a shared seed.
- **ZO (Zeroth-Order) gradient estimation** — figuring out which way to nudge a model by *trying* a direction and measuring the change, instead of using calculus.
