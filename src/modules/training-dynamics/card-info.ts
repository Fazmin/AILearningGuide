import type { ModuleCardInfo } from "@app/module-sdk";

const cardInfo: ModuleCardInfo = {
  "Training curves": {
    title: "Two losses from one real run",
    summary:
      "A polynomial model trained by gradient descent on 16 noisy points and scored on 16 held-out points after every checkpoint. The two curves, the gap, the lowest-validation marker, and the status are all measured from that run; none is a drawn shape.",
    whatYouSee: [
      "The horizontal axis is the epoch: 0 at the axis, then 1 to 10,000 on a log scale, so each labelled tick is ten times the last. The vertical axis is mean squared error on a log scale from 0.003 to 30.",
      "The solid blue line is training loss and the dashed orange line is validation loss. The shaded band between them is the gap. The dotted line is the noise variance σ² = 0.0625, the expected loss of the true curve itself.",
      "A green ring marks the lowest validation loss in the run, and the dotted vertical marker with two dots is the current `Epoch`. A red note appears if the run diverges.",
      "Metrics: training loss at the current epoch, `Validation`, `Gap (val − train)`, and `Lowest validation` with its epoch. The badge is the status: fitting, underfitting, overfitting, or diverged.",
      "At the defaults (degree 5, η 0.1, full batch, λ 0, seed 1) and epoch 100 the metrics read 0.017, 0.111, 0.093, and 0.100 at epoch 11, and the status is fitting.",
    ],
    howItWorks: [
      "The model is `ŷ = Σ θₖ φₖ(x)` with φₖ the Legendre polynomial of degree k scaled by √(2k+1), starting from θ = 0. Each epoch shuffles the 16 training points and takes 16 / batch steps of `θ ← θ − η(∇L + 2λθ)`, with the constant term left undecayed.",
      "Both curves are plain mean squared error. The training curve leaves out the weight-decay penalty, so the two lines measure the same thing on different points.",
      "Losses are recorded at 83 checkpoints: epoch 0 and about 24 per decade up to 10,000. Between checkpoints the chart draws straight segments.",
      "Status is a rule on measured values: overfitting once validation is 1.5 times the best it reached earlier in this run; underfitting while training loss is above 2σ² = 0.125; diverged once training loss passes 10⁶; fitting otherwise. The thresholds are the lab's choice.",
    ],
    controls: [
      "This card has no controls of its own. Everything is on `Training dynamics controls`.",
      "Comparison worth running: set `Polynomial degree` to 11 and drag `Epoch` to 10,000. Validation falls to 0.098 at epoch 13, turns up, reads overfitting from epoch 215, and ends at 8.63 while training loss keeps falling to 0.007.",
      "Second comparison: at degree 11 move `Learning rate` from 0.1 to 0.01. The lowest validation is almost the same, 0.097, but it arrives at epoch 147 instead of 13. On a log axis the whole run slides one decade to the right.",
    ],
    notice: [
      "The signal is the gap widening, not the gap existing. With 16 points some gap is normal from the first epochs.",
      "Training loss below the dotted noise line means the model is fitting the noise in its 16 points: the true curve itself only scores 0.081 on them.",
      "Early stopping is a reading of this chart: the ring marks the epoch you would keep.",
    ],
    limits: [
      "In this lab: a 1-D polynomial on 32 seeded samples of a known curve, so the noise level and the true curve are known, which real data never offers. Half the samples are held out so the validation curve is not decided by a few points; real projects usually hold out 10 to 20 percent.",
      "In this lab: the overfitting here is slow and grows over thousands of epochs because the wiggly directions of a polynomial are weakly curved. Other models overfit on other schedules.",
      "In general: validation curves are noisy, so practitioners smooth them, wait several evaluations before stopping, and average over seeds. Loss and accuracy can also disagree: validation loss can rise from overconfidence while accuracy still improves.",
      "In general: large-model pretraining often sees most data about once, so training and held-out loss stay close for the whole run. Overfitting returns in fine-tuning, in data repeated many times, and when a model is optimized against a learned reward.",
    ],
  },

  "Training dynamics controls": {
    title: "The knobs that change the run",
    summary:
      "The data split, which moment of training to inspect, and the four settings that change the run itself: model capacity, weight decay, learning rate, and batch size, plus which noise draw to use. Changing any run setting retrains all 13 degrees from scratch in the browser.",
    whatYouSee: [
      "A split strip: all 32 samples along x, filled dots above the line for training and hollow dots below for validation.",
      "`Epoch` (0 to 10,000 in 83 checkpoints) and `Stop at lowest validation`, which jumps to the ring on the curve chart.",
      "`Polynomial degree` (1 to 13, printed with its weight count), `Weight decay` (λ 0 to 0.1), `Learning rate` (η 0.003 to 0.6), `Batch size` (4, 8, or 16 = full), and `Noise draw` (seed 1 to 4).",
      "Two gradient readouts at the current epoch: `Full-batch |∇L|` and `Minibatch noise`, the root-mean-square distance between each minibatch's gradient and the full one.",
    ],
    howItWorks: [
      "The generator draws 32 points on a jittered grid over [−1, 1] with y = 0.8 sin(4x) + 0.4x plus Gaussian noise of standard deviation 0.25. Alternate points go to training and validation, so both splits cover the whole range.",
      "`Epoch` only chooses which checkpoint to read. The four run settings and `Noise draw` retrain every degree; the sweep takes a few tens of milliseconds.",
      "Weight decay adds `λ Σ θₖ²` for k ≥ 1 to the training objective, so each step also pulls every weight toward zero by `2ηλθₖ`.",
      "A smaller batch means more steps per epoch: 4 steps at batch 4, 1 at the full 16. Each of those steps follows a noisier gradient.",
    ],
    controls: [
      "Comparison worth running: at degree 11 set `Weight decay` to 0.01, 0.03, and 0.1 and read validation at epoch 10,000: 0.310, 0.135, and 0.102, against 8.63 with no decay.",
      "Second comparison: at degree 5 set `Learning rate` to 0.3 and `Batch size` to 4. The run diverges at epoch 22. Batch 8 at the same rate trains normally.",
      "Third comparison: at degree 11, η 0.6 diverges at epoch 562 while degree 5 at η 0.6 is stable. A bigger model can need a smaller learning rate.",
    ],
    notice: [
      "Two different fixes reach almost the same validation loss at degree 11: stopping at epoch 13 gives 0.098, and training to epoch 10,000 with λ 0.1 gives 0.102.",
      "`Minibatch noise` is several times `Full-batch |∇L|` once the full gradient is small: at degree 11, batch 4, epoch 100 they read 0.199 and 0.051. Late in training, minibatch steps are mostly noise.",
      "Changing `Noise draw` changes how badly each degree overfits. One seed is one sample of a noisy process.",
    ],
    limits: [
      "In this lab: weight decay is the only explicit regularizer, and early stopping is something you do with `Epoch`. There is no dropout, augmentation, or learning-rate schedule, and no test split is drawn.",
      "In general: choosing capacity, learning rate, or stopping point from the validation set spends its independence, so serious pipelines keep a third, test split touched once at the end. Distribution shift and data leakage can make a healthy-looking validation curve meaningless.",
    ],
  },

  "Fitted function": {
    title: "The model the curves are scoring",
    summary:
      "The fitted curve at the current epoch drawn over the data it was trained on and the data it was scored on, with the true curve the data came from. It shows what overfitting looks like in the function itself, not just in the loss.",
    whatYouSee: [
      "x from −1 to 1 and y from −2 to 2. Filled blue dots are the 16 training points; hollow orange dots are the 16 validation points.",
      "The dashed green line is the true curve, 0.8 sin(4x) + 0.4x, which the model never sees. The purple line is the model's ŷ at the current epoch.",
      "A red note gives the peak |ŷ| when the curve leaves the plot. Metrics: `Weights`, the degree plus one, and `Weight size ‖θ‖`, the length of the weight vector.",
    ],
    howItWorks: [
      "The purple line evaluates `Σ θₖ φₖ(x)` at 161 points using the weights stored at the current checkpoint, the same weights that produced the losses on the curve chart.",
      "Validation loss is the mean squared vertical distance from the purple line to the hollow dots; training loss uses the filled dots.",
    ],
    controls: [
      "Driven by `Epoch`, `Polynomial degree`, and the other run settings on `Training dynamics controls`.",
      "Comparison worth running: at degree 11 move `Epoch` from 13 to 10,000. The curve stops following the dashed line and bends toward individual blue dots, and near x = 1, where the last validation point sits beyond the last training point, it swings to a peak |ŷ| of about 24.",
    ],
    notice: [
      "Overfitting shows first at the edges and between training points, where nothing pins the curve down.",
      "Weight decay keeps `Weight size ‖θ‖` small: at degree 11 and epoch 10,000 it reads 2.48 with no decay and 0.74 with λ 0.03.",
    ],
    limits: [
      "In this lab: one input, so the whole function can be drawn. Wild swings between samples are typical of high-degree polynomials in particular.",
      "In general: a network's function has thousands of input dimensions and cannot be drawn. Overfitting is inferred from held-out loss, not seen.",
    ],
  },

  "Capacity sweep": {
    title: "Capacity matters differently at different times",
    summary:
      "Training and validation loss for every degree from 1 to 13 at the current epoch, all trained with the same settings. Moving `Epoch` shows the classic U-shape of validation loss against capacity forming as training goes on.",
    whatYouSee: [
      "The horizontal axis is polynomial degree 1 to 13; the current degree is shaded and its tick is bold. The vertical axis is mean squared error on the same log scale as the curve chart.",
      "Blue dots and line: training loss at the current epoch. Orange dots and dashed line: validation loss. The green ring marks the degree with the lowest validation loss, and a red × at the top marks a degree whose run has diverged.",
      "Metrics: `Best degree now` with its validation loss, and `Diverged runs`.",
    ],
    howItWorks: [
      "Each point reads the checkpoint at the current epoch from a full training run for that degree. There are 13 separate runs; none is interpolated.",
      "Each degree's model family contains the one below it, so at convergence training loss cannot rise with degree. Validation loss has no such guarantee, and at an early epoch neither does training loss, because descent has not finished.",
    ],
    controls: [
      "Driven by `Epoch` and the run settings on `Training dynamics controls`. `Polynomial degree` only moves the shaded band.",
      "Comparison worth running: at epoch 13 validation sits between 0.096 and 0.117 for every degree from 5 to 13. At epoch 10,000 it climbs from 0.100 at degree 7 to 0.246 at degree 9, 8.63 at degree 11, and 10.6 at degree 12.",
    ],
    notice: [
      "Degrees 1 and 2 stay high at every epoch: a line or a parabola cannot follow a sine. That is underfitting.",
      "Early in training, extra capacity costs little because the wiggly directions have not been fitted yet. That is why early stopping works as a regularizer.",
    ],
    limits: [
      "In this lab: degree stops at 13, below the 16 training points, so the model never has enough weights to pass through every training point exactly.",
      "In general: very large models can show double descent, where held-out error rises as capacity approaches the point of fitting the training set exactly and falls again well beyond it. This sweep does not reach that regime.",
    ],
  },
};

export default cardInfo;
