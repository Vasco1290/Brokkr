# Stage 4 so far, in plain words

*Last updated 3 October 2026 (findings wording approved by H; numbers generated from the 4.1 records and
the labels). Every number here was measured on one Windows laptop; details and exact rules are in
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
- *Is the collapse common?* Not in our pre-registered test. (9 models, 10,000 test images.) The two
  predictions about collapse were judged at two conditions only, darkness (Brokkr) s5 (H21) and contrast
  (ImageNet-C) s3 (H20). In each, 2 of 9 models lost much more to shrinking under the damage than on
  clean images (MobileNetV3-Large and EfficientNet-B0); we had predicted at least 5, so both are FAIL.
  Clean accuracy and how much shrinking cost under damage showed no strong relation (H18b: PASS, 8 of 11
  judged conditions, 8 needed; a weak test with 9 models). For example, MobileNetV3-Large's shrunk build
  scores 73.60% on clean images, 2.0 points below its full-precision build, but 13.6 points below it
  under darkness (Brokkr) s5.

  *Looking wider (exploratory, not pre-registered):* Across all 12 damaged conditions on the labels, 6
  of the 9 usable shrunk builds have at least one condition with a large shrinking cost (the whole
  interval more than 5.0 points below the full-precision build). Besides MobileNetV3-Large and
  EfficientNet-B0, this includes ConvNeXt-Tiny, MobileNetV2, RegNetY-400MF and ResNet-50. The condition
  hitting the most models is contrast (ImageNet-C) s5 (6 models). Exploratory, not pre-registered: large
  shrinking costs appeared in 19 of 51 usable build-condition pairs under darkness, fog and low
  contrast, against 3 of 43 under noise and blur (worst extra gap 38.9 vs 8.8 points; near-floor cells
  excluded). The median build's extra loss was under 5.0 points in every condition except contrast
  (ImageNet-C) s5 (14.0 points). Our conditions did not include impulse noise, which Xiao et al. found
  hit quantized models most.

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

**6. Are those side branches the cause? (test SE1)** We shrank the two collapsing models again, this
time keeping the side branches at full size, and tested them on 4,936 photos. We had said beforehand
that at least half of the darkness collapse must go away. It did not: EfficientNet-B0 still lost 38
points more in the dark than on clean photos (recovered: 0%), MobileNetV3-Large 12 points
(recovered: 5%). The broken MobileNetV3-Small stayed broken. **Not confirmed.** So what causes the
collapse is still unknown.

## What is still open

- **Why the two models collapse** is unknown: not from extra error in the first layers (M2), and
  probably not from the side branches (SE1). This question is parked: Brokkr now pauses new research
  until the first version of the website is out.
- **Next is the product:** a "nutrition label" for each model, built only from checked result files;
  then the first website; then model recommendations. Speed tests on a Raspberry Pi 5 wait for the
  board to arrive.

## The main lesson so far

Consistent with prior work on quantized models (Xiao et al., 2023, arXiv:2304.03968; Yaghoubi Araghi et
al., 2026, 4-bit, arXiv:2607.18540), we found that shrunk models can pass a normal accuracy check and
still collapse in dark or low-contrast images, and in our small pre-registered test, clean accuracy
didn't predict which. In wider exploratory checks, which models were hit varied with the condition, and
fog hit some models too. How much a model loses when shrunk, measured on clean photos, does go together
with how much it loses under damage, but mostly because of the two collapsing models: without them the
link almost disappears. So it is not an established warning sign, and each model has to be tested under
the conditions it will meet.
