# Related work: verified list

*Checked 3 October 2026; moved from `scratch/` into `docs/` on 6 October 2026 (H), unchanged apart from this
paragraph and the `Site:` lines. Each entry was checked on its arXiv page (abstract page, and the arXiv HTML
version where noted); the one non-arXiv source on its own page. Title, authors and year are as those pages
give them. "Found" is one line from the abstract unless marked. Quotes were read through a web-fetch tool,
so check them against the PDF before quoting them anywhere else. The one-line notes these sources were
suggested with were not trusted: where a note claims more than the page shows, that is said.*

*The website's related-work list is generated from this file (`web/site_sources.py`): each entry's title,
authors and year from its first line, and its key, place on the site, venue and link from its `Site:` line
("cited" = cited in the text of "Why labels?"; "also read" = listed under it; "not shown" = not on the
site). The site shows bibliographic facts only, never the notes.*

## Quantization and robustness under damage

1. **Benchmarking the Robustness of Quantized Models.** Yisong Xiao, Tianyuan Zhang, Shunchang Liu,
   Haotong Qin. 2023 (arXiv:2304.03968; CVPR 2023 workshop).
   *Found:* on ImageNet, lower-bit quantization is more resilient to adversarial attacks but more
   susceptible to natural corruptions and systematic noises (impulse noise affects quantized models
   most). *Note checks out.* Full text checked on 3 October 2026 (v1, the only version, 4 pages): Section
   4.3 says impulse noise "shows the highest impact" (about 50% average decrease, ResNet-18 models,
   2-8 bit, DoReFa/PACT/LSQ). Glass blur is not named as most harmful anywhere in the text.
   Site: key xiao · cited · where arXiv:2304.03968; CVPR 2023 workshop · link https://arxiv.org/abs/2304.03968

2. **Quantization Robustness to Input Degradations for Object Detection.** Toghrul Karimov, Hassan
   Imani, Allan Kazakov. 2025 (arXiv:2508.19600).
   *Found:* for YOLO detectors on COCO under seven degradations, a degradation-aware Static INT8
   calibration (clean plus degraded calibration images) "did not yield consistent, broad improvements in
   robustness over standard clean-data calibration", with an exception for larger models under some
   noise. *Note checks out.* (Object detection, YOLO models: cite only. Its models and code are not
   used; Ultralytics YOLO is AGPL, hard rule 3.)
   Site: key karimov · also read · where arXiv:2508.19600 · link https://arxiv.org/abs/2508.19600

3. **Recti-Q: Feature-Space Rectification for Out-of-Distribution-Robust Quantized Perception in Edge
   Robotics.** Hamidreza Yaghoubi Araghi, Parastoo Pilevar, Ming C. Lin. 2026 (arXiv:2607.18540,
   submitted 20 July 2026; accepted at IROS 2026 per its comments).
   *Found:* post-training quantization "often preserves clean in-distribution accuracy" but "can
   substantially degrade reliability under deployment-relevant distribution shifts"; 4-bit models on
   ImageNet-C and PACS show "pronounced robustness degradation despite negligible ID accuracy loss".
   *Note checks out* (its quantization is 4-bit, not the INT8 Brokkr uses).
   Site: key recti · cited · where arXiv:2607.18540 · link https://arxiv.org/abs/2607.18540

## Conformal prediction under distribution shift

4. **Empirically Validating Conformal Prediction on Modern Vision Architectures Under Distribution
   Shift and Long-tailed Data.** Kevin Kasa, Graham W. Taylor. 2023 (arXiv:2307.01088).
   *Found:* across conformal methods and network families, "performance greatly degrades under
   distribution shifts violating safety guarantees". *Note partly checks out:* the abstract does not say
   "all shifts"; the HTML version's Section 3.1 says the target coverage "is consistently violated
   across all models" and "even on small distribution shifts, such as ImageNet-V2" (shifts tested:
   ImageNet-V2, -C, -A, -R, and -W in an appendix). Say "under every shift they tested" only after
   checking that section in the PDF.
   Site: key kasa · cited · where arXiv:2307.01088 · link https://arxiv.org/abs/2307.01088

5. **Adapting Prediction Sets to Distribution Shifts Without Labels.** Kevin Kasa, Zhiyu Zhang, Heng
   Yang, Graham W. Taylor. 2024 (arXiv:2406.01416, v1 3 June 2024; UAI 2025, PMLR 286).
   *Found (abstract):* prediction sets' effectiveness "is frequently impaired by distribution shifts";
   the paper proposes ECP and EACP, which adjust conformal prediction using unlabelled test data. *Note
   not in the abstract:* "set sizes stay similar while coverage drops" is not stated there. The HTML
   version's Figure 4 caption says the standard split-conformal baseline "generates prediction sets
   with little variance in the set sizes, regardless of the achieved coverage rates". Cite that
   caption, not the abstract, and check it in the PDF.
   Site: key kasa2024 · also read · where arXiv:2406.01416; UAI 2025 · link https://arxiv.org/abs/2406.01416

## Benchmarks and reporting

6. **Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.** Dan Hendrycks,
   Thomas Dietterich. 2019 (arXiv:1903.12261; ICLR 2019).
   *Found:* introduces ImageNet-C (and ImageNet-P); finds "negligible changes in relative corruption
   robustness from AlexNet classifiers to ResNet classifiers". *Note overstates:* the abstract does not
   say "clean accuracy doesn't predict corruption performance"; it says relative robustness barely
   changed across architectures. Cite it for ImageNet-C, and for that finding in its own words.
   Site: key hendrycks · cited · where arXiv:1903.12261; ICLR 2019 · link https://arxiv.org/abs/1903.12261

7. **Model Cards for Model Reporting.** Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes,
   Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, Timnit Gebru. 2019
   (arXiv:1810.03993, submitted 2018; FAT* 2019).
   *Found:* proposes model cards, short documents released with a model that give benchmarked
   evaluation across conditions and groups and state intended use.
   Site: key mitchell · also read · where arXiv:1810.03993; FAT* 2019 · link https://arxiv.org/abs/1810.03993

8. **Higher accuracy on vision models with EfficientNet-Lite.** Renjie Liu (TensorFlow Blog). 16 March
   2020. **Not on arXiv** (checked on the blog page itself).
   *Found:* EfficientNet-Lite removes squeeze-and-excitation ("not well supported") and replaces swish
   with ReLU6, which "significantly improved the quality of post-training quantization"; the blog says
   EfficientNet's accuracy fell sharply after post-training quantization before that change. (Cited as
   background only; under the research freeze it is not used to explain Brokkr's results.)
   Site: not shown (a blog post, background only)

9. **Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming.** Claudio
   Michaelis, Benjamin Mitzkus, Robert Geirhos, Evgenia Rusak, Oliver Bringmann, Alexander S. Ecker,
   Matthias Bethge, Wieland Brendel. 2019 (arXiv:1907.07484). Added 3 October 2026.
   *Why cited:* it is the paper the `imagecorruptions` package asks users to cite (its GitHub README,
   github.com/bethgelab/imagecorruptions); the package describes itself as an extension of Hendrycks'
   ImageNet-C corruption functions to any image size. Brokkr's ImageNet-C conditions use that package,
   v1.1.2 (installed version read with `pip show`), vendored with the one-line NumPy 2 fix for fog.
   *Found (abstract):* provides "an easy-to-use benchmark to assess how object detection models perform
   when image quality degrades".
   Site: key michaelis · cited · where arXiv:1907.07484 · link https://arxiv.org/abs/1907.07484

## Dropped

None: all eight suggested sources exist and match their titles. Claims were narrowed where the notes
said more than the pages (4, 5, 6). Also narrowed on 3 October 2026: Xiao et al.'s abstract names
impulse noise (natural corruptions) and nearest-neighbour interpolation (systematic noises) as hurting
quantized models most; it does not name blur, so "noise and blur hurt quantized models most" is not
attributed to them.
