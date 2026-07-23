# Cancer Detection with Attention-Based Multiple Instance Learning

Lucas Guzylak, Computer Science, San Diego State University
Python, PyTorch, torchvision, MedMNIST, torchmil

A write-up of how I got from a clothing classifier to a model that finds cancer in pathology slides and locates it without being told where to look.

## Why I built it in stages

I wanted to work on cancer detection, but starting there would have meant learning five hard things at once and having no idea which one was broken when something failed. So I built a ladder instead. Seven stages, each adding one new technique on top of something that already worked.

That structure paid off in a way I didn't expect. When my first Multiple Instance Learning model refused to learn, I knew the training loop was fine because I'd already validated it on easier data. That let me look at the data instead of the code, which turned out to be where the problem was.

## Stage 1 and 2: fundamentals on FashionMNIST

Started with a fully connected network classifying grayscale clothing images. This was where the mechanics stopped being abstract: tensors, batching with a DataLoader, the forward pass, cross-entropy loss, backpropagation, and weight updates through Adam.

Then I replaced it with a convolutional network. A fully connected model flattens the image first, which destroys any sense of where pixels sit relative to each other. A CNN slides small filters across the image instead, so it keeps that spatial structure. Test accuracy went from the high 80s to 91.4%, and more importantly I understood why: parameter sharing and translation invariance, which matter for every image model that came after.

## Stage 3 and 4: CIFAR-10 and transfer learning

Full-color CIFAR-10 dropped a from-scratch CNN to 69.9%. That was the point of moving to it. The fix was transfer learning: load a ResNet18 pretrained on ImageNet, swap the final layer, and fine-tune. Accuracy jumped to 89.4%.

This stage also taught me something about infrastructure. Training ResNet on a MacBook Air M2 was slow enough to be unworkable, which pushed me onto GPU compute in Colab. Figuring out why it was slow (CPU instead of GPU, 224x224 images, a much deeper model) was as useful as the accuracy improvement.

## Stage 5: real medical images

Applied the same transfer learning approach to PathMNIST, which is real colorectal cancer tissue across nine tissue types. Reached 86.4% test accuracy. First time the work involved actual clinical data rather than benchmark images of animals and clothing.

## Stage 6: building an MIL pipeline, and a lesson about data

I wrote a complete Multiple Instance Learning pipeline from scratch: a custom Dataset class that groups patches into bags, a model that runs a ResNet feature extractor over each patch and max-pools the scores, and a binary bag-level classifier.

It never learned. Ten epochs, stuck at 50%, which is chance for a balanced binary problem.

The pipeline was correct. The problem was the data. I had taken TissueMNIST, which is 28x28 grayscale thumbnails, upscaled them to 224x224, and arbitrarily declared one tissue class to be "cancer." Blowing up a 28x28 image to 224x224 doesn't add detail, it just adds blur, and the class I picked wasn't visually distinct from the others in any meaningful way. There was no signal to find.

I could have spent a week tuning the learning rate and gotten nowhere. Recognizing that a model failing to train is often a data problem rather than a bug is the most useful thing I took from this project.

## Stage 7: real whole-slide images

CAMELYON16 is 399 whole-slide images of sentinel lymph nodes from breast cancer patients. The task is detecting whether cancer has spread to those nodes. Each slide is roughly 100,000 pixels on a side, so slides are cut into thousands of 512x512 patches, and only the slide gets a label.

I used a version of the dataset where patches have already been converted to 2048-dimensional feature vectors, and trained an attention-based MIL model on top. Because the feature archive is 26GB and I was working on free-tier compute with sessions that wipe, I used a balanced 60-slide subset (30 tumor, 30 normal) with a fixed-seed 75/25 split.

Training loss fell from about 32 to 8.5 over 20 epochs, and the model reached 86.7% on 15 held-out slides. That gradual curve was itself informative after Stage 6, where the loss had gone nowhere.

### What the attention weights showed

The model only ever saw slide-level labels. It was never told which patches contained tumor. But attention-based MIL assigns every patch a weight for how much it contributed to the final prediction, and those weights can be mapped back onto the slide.

On a representative tumor slide with 7,893 patches, 43 of which were genuinely tumor:

- Mean attention on tumor patches: 0.62
- Mean attention on healthy patches: 0.11
- Of the 50 highest-attention patches, 19 were real tumor

That last number needs context to read correctly. Tumor made up 0.5% of patches, so picking 50 at random would be expected to turn up about 0.3 tumor patches. Getting 19 is roughly 60 times better than chance, and it means the model found nearly half of all the tumor in the slide within its top 50 picks out of almost 8,000.

The model learned to localize the tumor as a side effect of learning to classify slides. That is what makes attention-based MIL useful in pathology: a prediction a pathologist can check against a region rather than a number they have to take on faith.

## What I learned

The techniques, in rough order of difficulty: CNNs and why convolution suits images, transfer learning and fine-tuning pretrained backbones, Multiple Instance Learning and weak supervision, attention mechanisms and using them for interpretability, and reading a train/test gap to spot overfitting.

The engineering was a bigger part of this than I expected. Writing custom Dataset classes and training loops in PyTorch, diagnosing CPU versus GPU bottlenecks, and working within real disk and compute limits, which meant targeted downloads, selective extraction, and building a balanced subset rather than taking the first N files. The first time I extracted a subset I took the first 60 files alphabetically and got 60 tumor slides and zero normal ones, which taught me to check class balance before training rather than after.

And the judgment: separating data problems from code problems, designing an honest train/test split with a fixed seed, and reporting results with the scale attached instead of quoting a number that sounds better without it.

## Limits of this result

The CAMELYON16 run uses 60 of roughly 400 slides. Full-dataset attention MIL reaches around 90% AUC, so this is a smaller demonstration of the same method, not a benchmark.

The test set is 15 slides, so each one moves the accuracy figure by about 6.7 points. It's a genuine held-out result but a coarse one.

The extracted subset all came from CAMELYON16's train partition, so I made my own 75/25 split rather than using the official test set.

The attention localization statistics are strongest on positive slides, and the attention map does light up some tissue that isn't tumor. The claim the numbers support is that attention concentrates on the tumor region, not that it isolates it cleanly.

## What I would do next

Run the full dataset on institutional GPU infrastructure, where persistent storage removes the subset constraint entirely. Add precision, recall, and AUC, since accuracy weights a missed tumor and a false alarm equally and in this setting they are not equally bad. Compare the attention approach against transformer-based MIL. And overlay the attention maps directly on slide thumbnails instead of scatter plots, which would make the qualitative check easier to read.

## Files

```
train.py                 Stages 1-4: FashionMNIST, CIFAR-10, transfer learning
medical.py               Stage 5: PathMNIST
cancer.py                Stage 6: MIL pipeline built from scratch (synthetic bags)
camelyon.py              Stage 7: attention-based MIL on real CAMELYON16
visualize_attention.py   attention analysis and the localization figure
README.md                short version
```