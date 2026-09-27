# Stage 4 so far, in plain words

*Last updated 27 September 2026, after M2 and before the squeeze-and-excitation test (SE1). Every
number here was measured on one Windows laptop; details and exact rules are in
[hypotheses_stage4.md](hypotheses_stage4.md).*

## The starting point

Brokkr shrinks image-recognition models so they fit on small devices. The usual way is to store
every number in the model with 8 bits instead of 32 (we call the small version "INT8"). In Stage 3 we
studied one model, MobileNetV3-Large, and found something odd: shrunk, it still worked in normal
light but fell apart in the dark (60% right instead of 74% for the full-size model on very dark
photos), even though darkness barely bothered the full-size model.

Stage 4 asks three things: does this happen to other models too, why does it happen, and can you
tell in advance which models it will happen to?

## What we tested and what we found

**1. Why might darkness hurt the shrunk model? (test M1)** One idea: a dark photo has only a narrow
range of brightness, so after shrinking it gets squeezed into very few of the 256 available steps.
We counted the steps actually used. Dark photos did use fewer (about 12% fewer), but less than the
20% we had said beforehand would count as support. Verdict: inconclusive.

**2. Ten models, thirteen kinds of damage (the breadth study).** We ran ten well-known models on
10,000 test photos, clean and with fog, darkness, blur, noise and low contrast. All ten full-size
models scored within one point of their published accuracy, so the setup is sound. One model's
shrunk version (MobileNetV3-Small) broke completely when shrunk, so it was studied full-size only.

- *Does a model that is more accurate on clean photos also stay more accurate when shrunk and
  damaged?* Only clearly in 3 of the 12 kinds of damage; we had predicted at least 8. **Not
  confirmed.** With only nine models, even a fairly strong link is hard to prove.
- *Does clean accuracy tell you how much shrinking will hurt under damage?* We predicted no strong
  link, and found none in 8 of 12 cases. **Confirmed**, just barely (8 were needed).
- *If shrinking hurts a model a little on clean photos, does it also hurt it more under damage?* Yes,
  in 9 of 12 kinds of damage (6 needed). **Confirmed.** But a later check showed this rests mostly on
  two models: without them, it holds in only 1.
- *Is the darkness collapse common?* No. Only 2 of 9 models lose much more to darkness when shrunk:
  EfficientNet-B0 (39 points more) and MobileNetV3-Large (12 points). The other seven lose at most
  about 2 points. Low contrast looks similar: the same two models stand out (38 and 13 points more),
  ConvNeXt-Tiny loses 4 points more, and the rest at most 2. **Not confirmed** (we had predicted at
  least 5 of 9 for each).

**3. Was it a set-up mistake?** Before trusting these numbers we checked that each shrunk model was
tuned on photos prepared exactly like the photos it was tested on. They were identical, to the last
bit, for every model.

**4. Where inside the models does the extra error appear? (test M2)** We measured, layer by layer,
how much extra rounding error darkness and fog cause, in eight models. We had predicted it would pile
up in the first layers. It does not: for the recommended shrinking method, 6 of 8 models showed no
such early pile-up. **Rejected.** A second question, whether the very first step of the shrunk model
throws away detail that a dark photo still has, came out clearly **no** for all eight models: the
photo itself loses the detail, not the shrinking.

**5. Where the extra error does sit (an exploratory look, not a test).** In the models that have
"squeeze-and-excitation" blocks (small side branches that turn each channel up or down), several of
the largest extra errors sit inside those blocks. Both collapsing models, and the model that broke
completely, have such blocks. But so does RegNetY-400MF, which does not collapse.

## What is still open

- **The squeeze-and-excitation test (SE1), designed but not yet run:** shrink EfficientNet-B0 and
  MobileNetV3-Large again but keep those side branches at full size, and see whether at least half
  of the darkness collapse goes away. RegNetY-400MF is a check (a weak one: it barely collapses
  anyway), and MobileNetV3-Small is measured alongside. It is only judged if the collapse also shows
  up on the photos used for this test.
- A summary table for the breadth study; turning results into "nutrition labels"; model
  recommendations; and speed tests on a Raspberry Pi 5, which has not arrived yet.

## The main lesson so far

Shrinking usually costs little, even on damaged photos, but a few models collapse under darkness
and low contrast. A model's accuracy on clean photos does not predict which ones. How much a model
loses when shrunk, measured on clean photos, does go together with how much it loses under damage,
but mostly because of the two collapsing models: without them the link almost disappears. So it is
not an established warning sign, and each model has to be tested under the conditions it will meet.
