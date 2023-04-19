[Let Language Models be Language Models](https://docs.google.com/document/d/1U7O6iEBwuxyQRiXe4pn7HRYWAyEGtEmFX59GL1vdwf8/view#)

```bash
$ pip install -r requirements.txt
$ python distill.py
```

Nothing fancy right now, a barebones proof of concept for my discrete memory transformer idea. Replaces the feedforward layers with two variations of kNN database memory to decouple memorization from language modeling. `distill.py` clones GPT-2, replaces the FF layers with the discrete memory, and trains it with knowledge distillation on the original with FF layers.