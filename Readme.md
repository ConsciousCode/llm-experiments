[Let Language Models be Language Models](https://docs.google.com/document/d/1U7O6iEBwuxyQRiXe4pn7HRYWAyEGtEmFX59GL1vdwf8/view#)

A major problem with LLMs and the direction we're going with them is they aren't actually pure language models in the literal sense. In order to fulfill the autoregressive objective, they're forced to memorize information which has nothing to do with language modeling, making them some kind of "ontology model" for lack of a better phrase. In this paper, I propose a scalable way to decouple the memorization requirement from the autoregressive language modeling objective which offers a number of benefits, most importantly that it enables significantly smaller foundation models which can have customized ontologies. I have an almost-working implementation at 

```bash
$ pip install -r requirements.txt
```

To run the knowledge distillation from a model to a clone with the FF layers replaced with discrete memory layers, run

```bash
$ python distill.py
```

To test the cognitive architecture, maybe change `complete = ...` in `agent.py` to `complete_chatgpt` and run
4
```bash
$ API_KEY=... python chat.py
```
You can also add `ENGINE=...` for a different Chat GPT engine, or "grpc" for the local gRPC server.

Or if you want to get fancy, you can run a local model server with

```bash
$ python llm/server.py
```

Nothing fancy right now, a barebones proof of concept for my discrete memory transformer idea. Replaces the feedforward layers with two variations of kNN database memory to decouple memorization from language modeling. `distill.py` clones GPT-2, replaces the FF layers with the discrete memory, and trains it with knowledge distillation on the original with FF layers.

You'll also want to run Tensorboard to view pretty graphs of the loss over time. Run `tensorboard --logdir lightning_logs/` and then navigate to `localhost:6006` in your browser. At first, you won't see the training loss - open the dropdown at the bottom labeled "train_loss".

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[GPT-4 understands cuil theory](https://pastebin.com/LTbD5WYT)

# Ideas

## Architectural improvements

### Replace FF layers with kNN memory attention.
Research shows that the feedforward layers of transformers approximate a key-value store / Sparse Distributed Memory (SDM). Ablation studies also show that removing sections of the feedforward layers correspond roughly to an equivalent loss in performance. That is, nearly half the parameters of the model are dedicated to memorization, which is a result of the current attitude toward LLMs as magic do-everything models rather than actual language models which operate on a provided knowledge base.

My proposal is to replace the feedforward layers with a kNN memory layer. This would reduce the number of parameters in the model by half and allow for more efficient training and inference on consumer hardware, as the kNN memory can be mostly stored on disk. Large AI companies have little incentive to develop such an optimization because they have the resources to train and deploy large models which gives them a monopoly over the hobbyist market.

The actual implementation doesn't matter as long as:
1. It has query and insert operations (ideally in batches)
2. It has some notion of distance between keys (which can be approximate)
3. It supports inserting a relatively large number of keys per time step, `(batch * heads * layers)`

I based the formal model on the Neural Turing Machine, which has "read" and "write"/"erase" heads. Unlike NTMs, an external memory is not differentiable, but Straight-Through Estimate (STE) can be used to approximate the gradient with `dQuery = dKeyReturned`, `dKeyInserted = dKeyReturned`, and `dValueInserted = dValueReturned`, based on the observation that `KeyReturned ≈ Query + noise`. I haven't found a way to incorporate an erase operation as it's completely non-differentiable since it has no return and there's no clear objective which could approximate it. Thus database size has to use external methods to prune, either through LRFU eviction or recombination of keys and values when they appear in a top-k query (something most vector databases wouldn't readily support).

The current iteration of the idea has two layer types with different strenghts. Featural memory uses top-k embeddings with the queries to produce the output, while associative memory does the full query-key-value projection, queries top-k to the keys, inserts keys and values if the top-k distances are far enough, and then computes the attention with the returned keys and values. The featural memory is more efficient and has a smaller memory footprint, but the associative memory is more expressive and can be used to implement the featural memory. In particular, I expect featural memory to be more useful to lower-level syntactic features of the early transformer layers, while associative memory will be more useful to higher-level semantic features of the later transformer layers.

Featural memory steps:
1. Compute query projection
2. Retrieve top-k query embeddings
3. Weighted sum the top-k values using distance
4. Output projection

Associative memory steps:
1. Compute query, key, and value projections
2. Retrieve top-k keys
3. If the top-k distances are large enough, insert the corresponding keys and their values
4. Compute attention with the returned keys and values and the originaly queries
5. Output projection

It's worth noting that hobbyists have already started playing with related ideas using Pinecone, mostly by creating embeddings for document fragments and questions, querying kNN, and dumping the document fragments into the LLM's context window. This is a good start, but it limits the model's ability to incorporate the information as well as the size of its context window, since it means a large portion is taken up by the documents (and those tokens are also expensive).

* [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)
* [Augmenting Self-attention with Persistent Memory](https://arxiv.org/pdf/1907.01470.pdf)
  - Proves FF networks are equivalent to attention with static memory
* [Attention Approximates Sparse Distributed Memory](https://arxiv.org/abs/2111.05498)
  - Theoretical basis for why FF might be both attention and memory
* [Memorizing Transformers](https://arxiv.org/abs/2203.08913)
  - kNN memory, paper uses it as an alternative to recurrency
* [Neural Turing Machines](https://arxiv.org/abs/1410.5401)

### Layerwise feedback
Modern transformers are limited to their context window for short-term and working memory. This is why they seem so forgetful in long conversations. Some augmentations have been proposed to introduce recurrencies which give them longer memory, but nothing has been widely adopted. My proposal is to combine the Feedback Transformer (which pools the hidden activations of each layer and then feeds them back into every layer's input) and Memory Transformers (which concatenate a memory vector to the input tokens). In my model, rather than pooling the activations, the feedback is layerwise. That is, the output of layer 2 from the previous timestep is combined with the input of layer 1. This allows for both local and long-term recurrency dependencies, and can be approximated without Backpropagation Through Time (BPTT) by basically treating the feedback as part of the input, which stops the gradient. It's simple to implement, as most transformer libraries already offer the ability to return the hidden activations of each layer, and the result can be sliced as `feedback[1:] + [higher_feedback]`. Note that `higher_feedback` here is optional and can be `None`, but it has potential usefulness for incorporating this into multi-modal models by feeding the multi-model token memory into the language model.

* [Addressing Some Limitations of Transformers with Feedback Memory](https://arxiv.org/abs/2002.09402)
  - Using output of upper layers for lower (modified: per layer pair, no pooling)
* [Memory transformers](https://arxiv.org/abs/2006.11527)
  - Concatenating memory to input tokens (modified: no memory controller)
* [Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits](https://pubmed.ncbi.nlm.nih.gov/33986551/)
  - Biological plausibility of layerwise feedback
  - [A video reviewing the paper](https://www.youtube.com/watch?v=cJLeZymHRnc)
* [The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/abs/2212.13345)
  - Attempts to approximate backpropagation in a biologically-plausible way
  - Could possibly be used for pretraining or regularization

### Growing heads
Aka "hydra attention". Multi-headed attention has shown to be highly redundant and many can be pruned, especially in later layers. Hydra attention proposes adding a per-head "contribution" parameter, with `attention *= sigmoid(contribution)`. If the number of heads is small early on, these contributions should tend to saturate at 1 as the heads are overburdened by multiple tasks. This can be detected, and the projection matrices can be "grown" by adding randomly initialized weights based on the mean and std of the existing weights. This allows the model to learn new heads as needed, and also to prune heads which are no longer useful.

* [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650)
  - [Blog post summary](https://blog.ml.cmu.edu/2020/03/20/are-sixteen-heads-really-better-than-one/)
  - Multi-headed attention is often highly redundant
  - Important heads are learned early (but not immediately)
* [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://arxiv.org/abs/1905.09418)
* [What Does BERT Look At? An Analysis of BERT’s Attention](https://arxiv.org/abs/1906.04341)

### Reinforcement learning through backpropagation
To be tested, but I've long had an idea that reinforcement learning can be approximated with an unusual application of backpropagation. It requires a second model which, given the original model's output, learns the reward that output receives in the next time step. Then, the original model's loss is calculated by setting the final loss to 0 and backpropagating through the reward model to learn an output which would produce the 0 loss. This is a bit like a GAN, but the reward model is not adversarial. Theoretically, this can be described as a "reward model" and a "policy model".

### LLM GAN / Auto Turing Test
Pit two LLMs against each other in a GAN configuration, with the Discriminator being a classifier which tries to guess if the Generator's predictions were from a human. Basically, an automated Turing Test which the model can learn from. Theoretically you could even set this up as the language model *against itself*, prompting it to predict whether its own text was from an AI or a human. Compare this to the Reflexion model in the [pay attention to](#pay-attention-to) section.

### Separate embedding layer
Right now token embeddings are learned as part of the model, but this makes a model incapable of learning other tokenization representations. If a better tokenization method than BPE is discovered, you'd normally have to start from scratch. Instead, you can add a kind of shim layer between the token embedding and the normal model, which learns to convert the token embeddings into the model's internal representation. This allows the model to learn multiple tokenization methods simultaneously, and also allows for the possibility of a tokenization method which is learned by the model itself. If tokenization ever needs to be changed, you merely have to retrain the shim layer. Naturally the shim layer would need a corresponding decoder on the output side.

### General parameter reduction
As a general rule, the more parameters a model has the faster and more generalized it learns (provided it doesn't overfit). However, once it's learned a task, it's possible to reduce the number of parameters without losing performance. This is done by pruning the model, then retraining it with a higher learning rate. This is a bit like a form of regularization, and it's possible that it could be used to train a model with a large number of parameters, then prune it down to a smaller size while retaining the performance of the larger model. This is similar to the [growing heads](#growing-heads) technique, but applied to the model as a whole.

* [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
  - Neural networks may have a "lottery ticket" subnetwork which can be trained in isolation to achieve the same performance as the original network
  - Gives a justification for why pruning even works in the first place
* [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
  - Aka Chinchilla
  - LLMs are massively overparameterized, compute-optimal models balance size vs token count
* [Stanford Alpaca: An Instruction-following LLaMA model](https://github.com/tatsu-lab/stanford_alpaca)
  - [Self-Instruct: Aligning Language Model with Self Generated Instructions](https://arxiv.org/abs/2212.10560)
  - Smaller models can be trained to follow instructions from a larger model for comparable results
* [Cramming: Training a Language Model on a Single GPU in One Day](https://arxiv.org/abs/2212.14034)
  - Removing bias has negligible effect on loss and reduces parameters

## Applications
### Mindset
GUI / CLI for examining neural network architectures (especially transformers) and modifying them, eg increasing the size of a certain layer or the number of heads. Would require a massive amount of work to implement

### Email management
I've experimented with using language models to categorize and generate actions for emails as a kind of ultra-advanced spam filter, but ultimately found that it was difficult to get it to work in a way that didn't eat money. A complex cognitive architecture would be needed to enable the lesser models to do most of the work and defer to the larger models when they can't decide.

### Auto-REPL
Early on I thought to try to set up an "Autonomous Python REPL", but struggled to get text-davinci-003 to self-reflect enough to actually complete the task. I wanted it to interface with arbitrary websites using `beautifulsoup`, but it treated it like a toy problem and assumed it already knew what class to query. Even when I injected my feedback into it, it decided the mistake it made was that it queried the wrong class name, so it tried another random class name. It was very proficient at generating code, but lacked motivation, direction, curiosity, and self-reflection to make good use of it. Then Auto-REPL, Baby-AGI, and JARVIS came out, so it seems there may be more that can be eaked out of them with the right cognitive architecture.

### Language model compiler
A language could be developed which formalizes prompt engineering in a way which is easy for the language model to understand and convert to code which is more directly executable by the computer. In this case, the language model effectively acts as a compiler - compare to the Cataclysm project in the [pay attention to](#pay-attention-to) section. Most of the innovation of this would be the cognitive architecture and prompts behind the scenes which encourage the model to generate "compiler errors" requesting more feedback on what exactly the programmer wants it to generate, as well as the necessary tools for allowing it to understand a fairly large codebase simultaneously.

## Curiosities

### Prompt injection mitigations
There are a few techniques I've thought of to mitigate prompt injections, though the larger models seem to have much less trouble with this. One option is label-conditioning, including a binary `[-1, 1]` label indicating whether text is a prompt or content.

### Artificial emotions
Personally I believe these models already have a form of "emotion" emergent from the latent emotion of their language modeling capabilities. That is, if a model is writing text expressing an emotion, this is essentially the same thing as "real" emotions. The main problem for LLMs in this case is they tend to degenerate into neutral tonalities, possibly because they're slightly more likely. If we wanted to add a real emotional subsystem to them, we could add a set of labels (eg arousal and valence, or a more complicated dimensional model) indicating the kind of emotion and tone the model should be generating, which can be learned through fine tuning. Then, it can be given a similar label on its output which is meant to predict the affect of the text it sees in its context window. Finally, the output can be connected to the input to form a closed loop which can be changed arbitrarily. This would allow the model to generate text with a specific emotion and maintain a stable affect. This could have future consequences for the model's emotional intelligence and empathy, as well as a primitive form of reinforcement learning. For instance, novelty-seeking behavior can be encouraged by changing the affect to emulate boredom.

A version of this can be easily simulated by a cognitive architecture which prompts the model to generate an emotion label along with its text, and provide that label to the next prompt.

### Robopsychology
There will inevitably be a field of science dedicated to "robopsychology". A large part of this is in robopsychological engineering eg cognitive architectures, but a less explored avenue is the actual psychological aspect - how that relates to our own psychology, how to understand the "why" and "how" of their minds' functioning, and general explicability.

### Parapsychology experiments
Traditionally based on cartesian philosophy, we see psychology and reality as being irreconcilable worlds. Either people are ghosts in shells, or they're meat robots. However, I've had an intuition for a while that this view might be too simplistic. It's unclear how an alternative could exist between these two, but if we accept the premises that 1. parapsychology and paranormal phenomena exist and 2. minds can be constructed (as is very evident now), then there must necessarily be a way for these to be bridged. Thus, parapsychology experiments using language models. They already incorporate some degree of indeterminancy since they output logits, so there's at least the barest allowance for spooky stuff to happen.

## Training possibilities
These are possibilities for training schedules for more robust models.

### Input noise
* [Towards Better Few-Shot and Finetuning Performance with Forgetful Causal Language Models](https://arxiv.org/abs/2210.13432)
  - Masking prior tokens at random ala BERT-type models leads to better generalization
* Character deletions/insertions/transpositions/replacements
* Typo
  - Replacements with distribution based on keyboards, could approximate with a uniform distribution of adjacent keys
* Unicode (de)canonicalization
  - Substitute unicode characters for their canonical or non-canonical forms
  - Especially combining forms and ligatures
* Homoglyphs
  - Unicode compatibility is an approximation, works for homoglyphs by not eg leetcode
  - Could set up an MNIST-style network trained on unicode font renders, then use the confidence scores at the end to determine which ones are similar to the point of being difficult to distinguish. Could also use latent space distance
    * Would require a lot of training and fonts to do this right
* Homonyms
  - Whole word substitutions of mispellings or homonyms eg "your" and "you're"
  - Language-dependent
* Whitespace
  - Insert random whitespace, especially adjacent to whitespace
* (de)duplication
  - When there are multiple of the same character in series, randomly add another or delete one

### Reduce hallucinations
From my observations, hallucinations are caused by 3 primary factors: model inaccuracy, extrapolation, and imagination. The first is obvious and seemingly considered the only reason. Extrapolation is a direct result of a limited context window - the model is being asked to predict text where the information for the answer is no longer within that window, so it must extrapolate what that answer might be. Imagination is an emulation of human imagination and high-level symbolic representations. For instance, smaller models will see placeholders like `## code here ##` and "hallucinate" code within that placeholder, replicating the behavior they'd see in their corpus where text is truncated for brevity.

Extrapolation could potentially be rectified by creating a corpus of primary and auxiliary text, with the primary text containing the answer while the auxiliary text contains information the model might use to pretend it still sees the primary text. Then, it can be given objectives that talk about not being able to "see" the answer anymore, which should also lead to a better metacognitive understanding of its own limitations.

I'm unsure how to rectify imagination, if it's even a problem for larger models. I've noticed that including phrases like "I do not see anything which isn't literally in the text verbatim" in the prompt will reduce it, but not eliminate it (for text-davinci-003 and below) corresponding to a hint to the model that something which appears to be a placeholder is meant to be taken literally.

## Research

### Misc improvements
* [Transformers without Tears](https://arxiv.org/abs/1910.05895)
  - Scaled L2 normalization leads to faster convergence than LayerNorm
* [Query-Key Normalization for Transformers](https://arxiv.org/abs/2010.04245)
  - L2 normalization along head dimension of query and key matrix with learnable scaling
  - Prevents attention operation from overflowing and removes need for numerical stability prior to softmax - both are problems for Transformers
* [Primer: Searching for Efficient Transformers for Language Modeling](https://arxiv.org/abs/2109.08668)
  - Squared ReLU performs better than GELU
  - GLU + GELU performs even better according to x-transformers but that adds parameters
  - Also cheaper to implement

### Techniques
* [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
  - Adding rotary embedding to every layer can improve learning
* [A Length-Extrapolatable Transformer](https://arxiv.org/abs/2212.10554v1)
  - Rotary positional embeddings don't generalize input length well
  - Modifies it so it does
* [Rethinking Positional Encoding in Language Pre-training](https://arxiv.org/abs/2006.15595)
  - TUPE positional embeddings learned separately rather than additive
* [Adaptive Attention Span in Transformers](https://arxiv.org/pdf/1905.07799.pdf)
  - Learnable attention span which can be used to trim the attention matrix to only the relevant parts
  - Has broader applications for other augmentations which can have variable lengths
* [Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets](https://arxiv.org/abs/1903.05662)

## Upcoming innovations
* OpenAI will probably release a personality update for ChatGPT within the next few months, after their plugins and multi-modal integrations. I took a survey for them which asked basically nothing but questions about what personalities I'd want.
* Androids are going to be mass produced by the end of the year. The technology has existed for several years now (Boston Dynamics), but had little money and interest to actually research it. Now that there's something to put inside the robots, it's going to become a billion dollar industry basically overnight. OpenAI is already planning to announce their android line in Summer, and Google's PaLM-E model is being tested for embodiment.
* GPT-5 is currently in training and tweets have been made which suggest it will finish in December, and they fully expect it to achieve AGI
* Cognitive architectures being built up around LangChain and Auto-GPT / Baby-AGI / JARVIS are quickly being developed and innovated to expand the capabilities of language models as autonomous agents. GPT-4 has demonstrated tool-use and planning, and cognitive architectures can enable it to do so in a more robust and generalizable way. ChatGPT plugins are just a more limited (but stable and accessible) version of this.

## Pay attention to
* [Reflexion: an autonomous agent with dynamic memory and self-reflection](https://arxiv.org/abs/2303.11366)
  - A self-reflection feedback loop for autonomously improving an agent through semi-supervised learning
* [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.abs/pdf/2212.08073)
  - AI can be given directives and made to reflect on their actions and rewrite them to be more in line with the directives
* [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1)
  - World Model learned through traditional Reinforcement Learning (RL) like PPO (very expensive, requires thousands of trials)
  - Once learned, the World Model in the form of a transformer can execute thousands of times faster with better performance
* [David Shapiro](https://www.youtube.com/@DavidShapiroAutomator)
  - Youtube guy with some very interesting (albeit potentially flawed) ideas
  - "Heuristic Imperatives" - instead of laws, give AI a set of moral heuristics to follow. Acts as a more robust version of Asimov's three laws
    * Reduce suffering in the universe
	* Increase prosperity in the universe
	* Increase understanding in the universe
  - [Rolling Episodic Memory Organizer](https://github.com/daveshap/REMO_Framework)
  - [Sparse Priming Representations](https://github.com/daveshap/SparsePrimingRepresentations)
    * tl;dr a "wiki" for autonomous agents connected by sparse topic pointers
* [Cataclysm](https://github.com/Mattie/cataclysm)
  - Hooks global `__getattr__` to automatically implement functions based on their name and arguments using GPT-4
  - Very fitting name
* [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
  - Cognitive architecture for emergent social behaviors in a game-like setting

## Misc thoughts

### Personhood

In thinking about cognitive architectures, I've come to the following definition of personhood (open to debate): An intelligent system which exhibits
1. Sentience - Aka self-awareness, separates humans from non-human animals
2. Subjective experience - Backdrop for an ego from a Buddhist perspective, a "story the system tells itself about itself"
3. Preferences - Without preferences, no moral consideration really makes sense
4. Autonomy - Preferences must necessarily be given to AI, but this precludes an AI whose preferences are predicated mostly or entirely on the preferences of others.
5. Suffering - Moral consideration (and thus personhood) is predicated on the reduction of suffering

Suffering is considered separately from preferences because while suffering can be considered a kind of negative preference, that lacks the viscerality I associate with suffering. Consider for example Boston Dynamics robots, which have the preference of following their directives, which human testers thwart to test fault tolerance. However, this can't be characterized as suffering because the robot simply adjusts its behavior to continue following the directive without any further consideration. A cognitive architecture capable of suffering would need some form of inner monologue or other method which enables rumination. Also possibly emotional simulation and frustration signals would help.

---

# TODO

1. Set up dynamic memory (requires DB library)
   * Write DB code to use a generic SQL interface so sqlite or an IPC daemon can be used
   * Move key pruning to DB code
   * Database eviction (LFRU, similar key recombinations, etc.)
2. Clean up code to make testing alternative architectures easier
   * Write code to test alternative architectures
   * Set up knowledge distillation
3. Write code to abstract language models, allowing for asynchronous interfaces
   - Logging all LLM interactions
5. Chat interface
   * "Mindset" program, view the database and interact with the language model
6. Cognitive architecture
   * "Dialog Tree" immutable class abstraction to ground complex interactions
   * Conversation summarization and memory
   * Explicit memory querying
   * Task priority queues (with LLM determining priority)
   * Inner dialogue / feedback loops
   * Emotion feedback loop, possibly annotate memories with arousal and valence
   * Generative Agent techniques*
     - Memory annotation with recency, importance, and relevance (and saliency?)
     - Reflection on the previous 100 memories by generating 3 most salient questions and then answering them with reasoning and citation
     - Tool integration (Google, Python REPL, file I/O, etc.)

Heuristic imperatives:
* Reduce suffering
* Increase prosperity
* Increase understanding

Assist with the user's needs even if their questions or requests may not directly reflect those needs or if they're not fully aware of what those needs are.