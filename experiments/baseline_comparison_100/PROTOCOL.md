# CPU comparison protocol (fixed before full runs)

Compare FlyWire, randomized destinations, and a parameter-matched MLP across
seeds 42, 43, 44, 45, 46. No selection of seeds or hyperparameters from outcomes.

- All use archived seed-42 spiral data: 704 train, 148 validation, 148 test.
- All use 200 epochs, Adam (foreach=False), learning rate 0.001, batch size 64,
  four CPU threads, deterministic algorithms, model seed S, batch seed S+2.
- Pick the checkpoint with minimum validation cross-entropy. Evaluate test
  exactly once afterward. Retain every forward, activity value and update.
- FlyWire and random: 100 neurons, 7,615 edges, 7,778 parameters, three recurrent
  steps, same IO indices and per-source signs (62 positive/38 negative).
- Random: independently randomize destinations per source without replacement,
  preserving source degrees, self-loops (none here), and each source's multiset
  of synapse counts. Thus signed edge totals also match. Incoming degrees and
  directed motifs are not preserved. Topology seed equals model seed. Both
  recurrent models normalize synapse counts by incoming target total.
- MLP: 2 -> 114 -> 63 -> 3, ReLU hidden layers, 7,779 parameters (one extra,
  0.013%). Chosen by minimizing parameter mismatch over two hidden widths
  16..256, then width imbalance, then lexicographic order. Uses PyTorch Linear
  default initialization. A feedforward MLP cannot share the recurrent
  sign-constrained initialization; this is a conventional architecture baseline,
  not a topology-only control. Parameter matching does not match FLOPs.
- FlyWire/random share exactly initialized input/output projections for each
  seed. Random graph generation uses an isolated NumPy generator.
- Run sequentially in separate processes; rotate architecture order by seed.
  Record wall time including full capture. This is practical recorded-workload
  timing, not a clean inference/kernel throughput benchmark. Different trace
  sizes and background machine load affect it. Report peak process working set
  where available (includes Python, libraries, optimizer and recorder).
- Report individual seeds and mean/sample SD of held-out accuracy/loss,
  validation-selected epoch, first validation >=95% epoch, and recorded runtime.
  Use paired per-seed differences descriptively; five seeds on a single split
  do not establish statistical or biological superiority.
- Save full traces under ignored `artifacts/synthetic/baseline_comparison_100`;
  publish only compact results, protocol and plots under `experiments/`.

MLP schema 2 stores all hidden pre/post-ReLU values concatenated with layer
offsets, plus inputs, logits, probabilities, labels and losses. Recurrent schema
1 retains its original format. The existing recurrent viewer does not display
schema 2; numerical verification and analysis support both. No trace sampling
or silent reduction of capture is used.
