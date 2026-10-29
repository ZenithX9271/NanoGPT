[cs1230342@vsky009 ~/zenithX9271/NanoGPT]
$ python -u gpt.py
Using device: cuda
GPU: Tesla V100-PCIE-32GB
Dataset: 1,115,394 characters

==============================================================
  ENVIRONMENT
==============================================================
  GPU:       Tesla V100-PCIE-32GB
  CUDA:      11.0
  cuDNN:     None
  PyTorch:   1.10.0a0+git36449ea
  Python:    3.7.9
  Platform:  Linux-3.10.0-1160.el7.x86_64-x86_64-with-centos-7.9.2009-Core
  Model:     211,777 params (209,024 decay | 2,753 no-decay)
  Config:    batch=32 block=64 n_embd=64 n_head=4 n_layer=4 dropout=0.0
  Data:      input.txt 1,115,394 chars | vocab 65
  Seed:      1337
==============================================================

BENCHMARK:  50 warmup + 5 windows × 100 steps = 550 total steps

  Warmup done (50 steps, 7.3s from script start).

  Window 1/5:      65,272 tok/s  |  step med 31.2ms  p95 32.8ms
  Window 2/5:      64,285 tok/s  |  step med 31.1ms  p95 35.5ms
  Window 3/5:      65,505 tok/s  |  step med 31.0ms  p95 32.7ms
  Window 4/5:      65,278 tok/s  |  step med 31.1ms  p95 33.4ms
  Window 5/5:      65,124 tok/s  |  step med 31.2ms  p95 33.3ms

==============================================================
  BENCHMARK RESULTS
==============================================================
  Windows measured:   5 x 100 steps
  Throughput (window clock, sync at boundaries only):
    mean      65,093 tok/s
    min       64,285 tok/s
    max       65,505 tok/s
    var          1.9%  (max-min)/mean
  Step latency (CUDA-event, synced per step — pipeline serialized):
    median  31.10 ms
    p95     33.82 ms
  GPU memory (peak over benchmark):
    allocated  48.0 MiB
    reserved   62.0 MiB
  Host RAM peak:      1904.0 MB
  Eval (200b x2):      3.40s  train 2.3191  val 2.3265
  Checkpoint save:    0.16s  1.1 MB
  Startup to warmup:  7.3s
  Tokens processed:   1,126,400
  Tokens/sec (est.)   65,093

  NOTE: Earlier 54K-62K tok/s figures were unvalidated (no CUDA
  sync, single-step wall-clock). This report supersedes them.
==============================================================

TRAINING:  steps 550 → 4999

          grad_norm 3.078 | approx 65248 tok/s (single-step, no sync)
  step  1000: train 2.1125 | val 2.1456 | lr 2.78e-04 | tokens 2,048,000 <- best
          grad_norm 3.680 | approx 67016 tok/s (single-step, no sync)
  step  1500: train 1.9117 | val 1.9982 | lr 2.49e-04 | tokens 3,072,000 <- best
          grad_norm 4.603 | approx 65243 tok/s (single-step, no sync)
  step  2000: train 1.7917 | val 1.9102 | lr 2.12e-04 | tokens 4,096,000 <- best
          grad_norm 4.770 | approx 66428 tok/s (single-step, no sync)
  step  2500: train 1.7165 | val 1.8500 | lr 1.69e-04 | tokens 5,120,000 <- best
          grad_norm 5.165 | approx 65300 tok/s (single-step, no sync)
  step  3000: train 1.6701 | val 1.8270 | lr 1.27e-04 | tokens 6,144,000 <- best
          grad_norm 4.730 | approx 66868 tok/s (single-step, no sync)
  step  3500: train 1.6255 | val 1.7894 | lr 8.78e-05 | tokens 7,168,000 <- best
          grad_norm 4.548 | approx 66582 tok/s (single-step, no sync)
  step  4000: train 1.6041 | val 1.7684 | lr 5.68e-05 | tokens 8,192,000 <- best
          grad_norm 5.593 | approx 66489 tok/s (single-step, no sync)
  step  4500: train 1.5838 | val 1.7603 | lr 3.69e-05 | tokens 9,216,000 <- best
  step  4999: train 1.5695 | val 1.7423 | lr 3.00e-05 | tokens 10,237,952 <- best
          grad_norm 5.262 | approx 65534 tok/s (single-step, no sync)

Loading best checkpoint (val loss: 1.7423)...

Total runtime: 195.0s (3.2 min)

--------------------------------------------------------------
  Generated text (NanoGPT)
--------------------------------------------------------------


Clown:
Stridl'd and is not the acter bure up:
What them me?
I'at such higher's him vet?

MEXENIO:
I away, my fears, to zorn heaow,
From find you vail; I long egrivel:
In cintlatiHer drove to her in preace.

WARWICK:
Nand tear.
How my no my speak; and thew you lord.
I came at is I will
the the call leong-not what evils, when, bout wasches,
We stoong of his but than nonrupted so;
Ange the flown yould of while Prive my of.

HENRY BRIARD:
Nigat adval the Earth, honours knear tey Ruch stoop, after;

--------------------------------------------------------------
[cs1230342@vsky009 ~/zenithX9271/NanoGPT]