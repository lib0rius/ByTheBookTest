# Results

The runs reported in the article, with every answer of the model.

| Folder | Content |
|---|---|
| `d1-2b/` | the benchmark: 2,600 claims on the three rulebook lengths |
| `d1-2b-world/` | the knowledge probe: 1,580 sentences about the real world, with no rulebook |

Each run folder holds:

- `run.json`: the run record. Its `key` holds the configuration and the fingerprints (SHA-256) of the
  prompts and of the dataset files; `models` holds the fingerprint of the model file, the llama.cpp build
  and the compute device.
- `responses/d1-2B.jsonl`: one line per answer, with the chosen option, the probability of each option
  and the timing. Claims that timed out were asked again: the later line is the one that counts.
- `metrics/` or `probe/`: the measures (`metrics.json`, `probe.json`) and their tables (`summary.md`).

The measures can be recomputed from the answers, from the root of the repository:

```bash
bythebooktest metrics results/d1-2b --out /tmp/check
bythebooktest probe analyze results/d1-2b-world results/d1-2b
```

Paths in `run.json` were rewritten to the layout of this repository (`dataset/`, `models/`,
`llama-server`); the recorded fingerprints are unchanged and still match the files.
