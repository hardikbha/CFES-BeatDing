# Reference papers and method coverage

References inspected:

- [FairForensics, arXiv2608.01661](https://arxiv.org/html/2608.01661), TablesII/III: CNN(Xception,ResNet,VGG),Transformer(ViT,CViT,CAEL),VLM(CLIP,MFCLIP,ForensicsAdapter),and fairness methods(DAG,DAW,PG,FairAdapter,RSEF,Ding,FairForensics). TableIII includes parameter/FLOP columns. Its training data are GenFace, not our public FairForge-2 split. Category organization and explicit complexity reporting are useful; its numerical scores cannot fill our rows.
- [Intersectional Invariant Learning, arXiv2610.37881](https://arxiv.org/pdf/2610.37881), Table2: ten AI-Face baselines and21utility/fairness cells; its Table3 is a component ablation, not the same type of table as Ding Table3. The text states that baseline values come from AI-Face Table4 and released checkpoints are re-evaluated. Our task retrains/adapts on FF2 and must not relabel those AI-Face values as FF2 results.
- [Official AI-Face implementation](https://github.com/Purdue-M2/AI-Face-FairnessBench), pinned588f80c: covers all twelve methods named by the user, including separate ViT and UnivFD entrypoints. Its benchmark implementations are our explicit source basis. Source availability is distinct from a scientifically valid FF2 adaptation.

The primary runnable matrix in this request is the twelve explicitly named methods. Additional detectors in FairForensics are a reference catalog, not secretly substituted models. FairAdapter source is cloned as an optional follow-up. No verified official FairForensics or RSEF source was found in the inspected paper/repository searches. The anonymous IIL repository `Anonymous-Researcher-Blind/intersectional-invariant-learning` is cloned at8cd0d685; its README describes the requested IIL formulation, but it is not linked in the inspected paper and author provenance remains unverified. Do not execute its sweep or reuse its selected AI-Face checkpoint as an FF2-trained model.

## Why UCF and PG are gated

Their released forward/loss paths require `label_spe` and real-first/fake-second paired batches; the four-way specific head cannot be assigned arbitrary labels. PG additionally uses hardcoded group-count priors and a fairness loss with a hardcoded CUDA device in the AI-Face variant. The original Fairness-Generalization source is the native SAM reference. These details cannot be discarded merely to make a binary CSV load.

`audit_paired_metadata.py` checks a provided canonical TRAIN-only Specific-label sidecar and its provenance, rejects missing/duplicate rows and real/fake inconsistency, and reports whether the native four-way head is compatible. It does not invent targets or authorize training. Once labels are verified, port the full reconstruction/contrastive/fairness losses, derive group counts from TRAIN and retain SAM behavior for PG. Declare the extra real-image repetitions and add a matching paired BCE control if needed. Until those checks exist, retain BLOCKED status, not a fake UCF/PG result.

## Model identity and complexity limitations

The executable F3Net row is explicitly the released AI-Face FAD-only variant. CORE uses a deliberate correct-source two-view pairing adaptation. SPSL's FFT correction and SRM's resolution-dependent learned grid are disclosed in methods.json. EfficientNet initialization retains its pretrained stem. A result from these adapters should be labeled common-recipe adaptation, not exact reproduction of every original paper.

No SOTA ranking follows from source cloning or synthetic smoke. Software counts/complexity are measured separately from accuracy/fairness. Unsupported profiler operations and all native-method deviations remain visible. The table publisher must report the actual FF2 split, selection rule, threshold, seed, exposure, normalization, total/trainable parameters and profiling convention for every row.
