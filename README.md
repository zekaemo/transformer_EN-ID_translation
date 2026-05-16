# Transformer: Attention is All You Need
### English → Indonesian Neural Machine Translation

An implementation of the Transformer architecture to English–Indonesian translation. Also includes two inference experiments, — greedy decoding and beam search.
---


## Dataset

`Helsinki-NLP/opus-100`(https://huggingface.co/datasets/Helsinki-NLP/opus-100) (`en-id`) 


## Tokenizer

**Type:** Shared BPE (Byte-Pair Encoding) tokenizer trained on both English and Indonesian sentences

**Vocabulary size:** 16,000 subword tokens (shared across both languages)

## Model Architecture

A single encoder–decoder Transformer, following the original paper.

> **Note:** The original paper uses 6 encoder and 6 decoder layers. This implementation uses only 1 each.

| Hyperparameter | Value |
|---|---|
| Embedding dimension (`d_model`) | 256 |
| Feed-forward hidden size | 512 |
| Attention heads | 8 |
| Encoder layers | 1 |
| Decoder layers | 1 |
| Dropout | 0.1 |
| Vocabulary size | 16,000 |
| Max sequence length | 50 |

**Loss function:** Cross-entropy with label smoothing (ε = 0.1), ignoring padding tokens

**Optimizer:** Adam (β₁=0.9, β₂=0.98, ε=1e-9)

**Scheduler:** ReduceLROnPlateau (patience=2, factor=0.5)


## Training Setup

| Setting | Value |
|---|---|
| Epochs | 10 |
| Learning rate | 1e-4 |
| Dataset size | 50,000 pairs |



## Inference Experiments

Two decoding strategies were evaluated on the same trained model weights across four sentence categories: short sentences, negation sentences, complex grammar, and compound sentences.

### Experiment 1 — Greedy Decoding

At each decoding step, always selects the single highest-probability token. 

### Experiment 2 — Beam Search (k = 2, 3, 4)

Maintains top-k candidate sequences at every step. Tested with beam sizes k=2, k=3, and k=4.



## Results & Insights

### Result
| English Input | Ground Truth (ID) | Greedy | Beam k=2 | Beam k=3 | Beam k=4 |
|---|---|---|---|---|---|
| today is tuesday | Hari ini adalah selasa | Di hari ini adalah hari | Di hari ini sangat penting | Di hari ini sangat penting | Di hari ini . |
| Hello, how are you? | Halo, apa kabarmu? | Halo , bagaimana kau ? | Halo , bagaimana kau ? | Halo , bagaimana kau ? | Halo , bagaimana kau ? |
| I'm fine, thank you. | Aku baik, terimakasih | Aku baik - baik saja . | Aku baik - baik saja . | Aku baik - baik saja . | Aku baik - baik saja . |
| Thank you. | Terimakasih | Terima kasih . | Terima kasih | Terima kasih . | Terima kasih . |
| I'm hungry. | Aku lapar | Aku lapar . | Aku lapar . | Aku lapar . | Aku lapar . |
| i haven't eaten yet | Aku belum makan | Aku belum pernah belum pernah | Aku belum pernah belum pernah | Orang - orang yang belum pernah | Orang - orang yang belum pernah terjadi . |
| I have never been to this place before | Aku tak pernah ke tempat ini sebelumnya | Aku tak pernah pernah pernah melakukan ini sebelum ini | Aku tidak pernah pernah pernah melakukan ini sebelumnya | Aku tak pernah pernah bertemu di tempat ini | Aku tak pernah pernah kembali sebelum ini |
| She will never ever do that again | Dia tidak akan pernah melakukan itu lagi | Dia tidak pernah pernah melakukan itu lagi | Dia takkan pernah melakukan itu lagi | Dia takkan pernah melakukan itu lagi | Dia takkan pernah melakukan itu lagi |
| I'm hungry because i haven't eaten yet. | Aku lapar karena aku belum makan | Aku karena karena karena belum pernah belum pernah di sini . | Aku karena karena karena belum pernah belum pernah di sini . | Aku lapar karena karena belum pernah belum pernah di sini . | Aku lapar karena karena belum pernah belum pernah di sini . |
| He doesn't know what he is doing | Dia tidak tahu apa yang dia lakukan | Dia tidak tahu apa yang dia lakukan | Dia tidak tahu apa yang dia lakukan | Dia tidak tahu apa yang dia lakukan | Dia tidak tahu apa yang dia lakukan |
| The more you learn, the more you grow | Makin banyak kau belajar, makin kau berkembang | Yang lebih banyak lagi , kau lebih baik | Yang lebih banyak lagi , kau lebih baik | Orang - orang yang Anda , Anda lebih baik dari Anda . | Orang - orang yang Anda , Anda lebih baik dari Anda |
| The cat that the dog chased ran away | Kucing yang dikejar anjing itu melarikan diri | Yang Mulia itu menjadi seorang wanita yang telah di dalam neraka . | Yang Mulia itu adalah seorang wanita yang telah di dalam neraka . | Orang - orang itu membuat orang yang membuat orang itu . | Orang - orang itu membuat orang yang membuat orang itu . |
| Although it was raining heavily, we decided to go outside | Walaupun tadi hujan deras, kita memutuskan untuk pergi keluar | Meskipun itu adalah dia sangat baik , kita pergi ke rumah | Meskipun itu adalah dia sangat baik , kita harus pergi ke rumah | Meskipun itu adalah dia sangat baik , kita harus pergi ke rumah | Meskipun itu adalah alasan , kita harus pergi ke rumah |
| The student who studied hard for the exam finally passed | Murid yang belajar dengan baik untuk ujian akhirnya lulus | ( Dan siapa yang telah mati ) yakni bagi orang - orang yang beriman ( untuk orang yang beriman ) yakni orang - orang yang beriman . | Orang - orang yang telah mati untuk hari kiamat . | ( Dan orang - orang yang telah mati ) yakni kepada orang - orang kafir . | ( Dan orang - orang yang telah mati ) yakni orang - orang kafir . |
| I would have gone if you had asked me | Aku akan pergi jika kau mengajak aku | Aku akan pergi jika kau punya aku punya aku . | Aku akan pergi jika kau punya aku . | Aku akan pergi jika kau punya aku . | Aku akan pergi jika kau punya aku . |
| She has been working here since 2020 | Dia sudah bekerja disini sejak 2020 | Dia telah bekerja di sini selama 20 | Dia telah bekerja di sini selama 20 . | Dia telah bekerja di sini di sini . | Dia telah bekerja di sini di sini . |
| I am hungry and tired | Aku lapar dan lelah | Aku adalah orang dan lapar . | Aku lapar dan lapar . | Aku lapar dan lapar . | Aku lapar dan lapar . |
| She is smart but lazy | Dia pintar tapi malas | Dia adalah seorang pria yang sangat penting . | Dia sangat senang . | Dia adalah orang yang sangat penting . | Dia sangat baik - baik . |
| We can go now or wait until tomorrow | Kita bisa pergi sekarang atau tunggu sampai besok | Kita bisa pergi sekarang atau tunggu besok besok | Kita bisa pergi sekarang menunggu besok . | Kita bisa pergi sekarang menunggu besok . | Kita bisa pergi sekarang menunggu besok . |
 

 
 
### Experiment 1 — Greedy Inference
 
**Insight 1: Short, high-frequency sentences translate well**
 
The model handles simple conversational phrases correctly and consistently:
- "I'm hungry" → "Aku lapar" ✓
- "Thank you" → "Terima kasih" ✓
- "He doesn't know what he is doing" → "Dia tidak tahu apa yang dia lakukan" ✓
These patterns appear frequently in the training data. 
 
**Insight 2: Negation triggers repetition collapse**
 
- "i haven't eaten yet" → "Aku belum pernah belum pernah" 
- "I have never been to this place before" → "Aku tak pernah pernah pernah melakukan ini sebelum ini" 
- "She will never ever do that again" → "Dia tidak pernah pernah melakukan itu lagi" 
Because of minimal depth of construction understanding,  it gets stuck repeating the same tokens with no recovery.
 
**Insight 3: Complex grammatical structure causes complete semantic collapse**
 
- "The cat that the dog chased ran away" → "Yang Mulia itu menjadi seorang wanita yang telah di dalam neraka ." 
- "The student who studied hard for the exam finally passed" → "( Dan siapa yang telah mati ) yakni bagi orang - orang yang beriman..." 
- "I'm hungry because i haven't eaten yet" → "Aku karena karena karena belum pernah belum pernah di sini ."  (main clause "Aku lapar" completely lost)
Relative clauses and subordinate structures, push the model to produce  unrelated ouputs. Interesting part is on the second sentence, it's translated to a Quran verse. Which is a dominant pattern in the dataset.


 
### Experiment 2 — Beam Search (k = 2, 3, 4)
  
**Insight 1: Minor but real improvement on simple negation**

"She will never ever do that again":
- Greedy: "Dia tidak pernah pernah melakukan itu lagi" — double "pernah" 
- k=2 / k=3 / k=4: "Dia takkan pernah melakukan itu lagi" ✓ — cleaner, more natural Indonesian
"I would have gone if you had asked me":
- Greedy: "Aku akan pergi jika kau punya aku punya aku ." — "punya aku" repeated 
- k=2 / k=3 / k=4: "Aku akan pergi jika kau punya aku ." — one fewer repetition
For sentences where the model has partial knowledge, beam search fix the repetition:
 

**Insight 2: Grammar error still persists**
 

"i haven't eaten yet":
- Greedy / k=2: "Aku belum pernah belum pernah" — repeats but keeps correct subject "Aku"
- k=3 / k=4: "Orang - orang yang belum pernah terjadi ." — subject completely hallucinated to "people", semantically further from correct 
"The more you learn, the more you grow":
- Greedy / k=2: "Yang lebih banyak lagi , kau lebih baik" — vague but preserves "kau" (you)
- k=3 / k=4: "Orang - orang yang Anda , Anda lebih baik dari Anda ." — degrades into third-person hallucination 

Because of its characteristics of exploring more paths, beam search confidently selects the most wrong one if all paths are wrong. The results are sometimes worse than greedy

 
### Key Takeaway
Both result experiments are bounded by the same cause, the model itself.
 
Greedy fails because it has no error recovery. Beam search was supposed to fix this, but for an undertrained model it doesn't reliably do so. For complex sentences where the model has genuine knowledge gaps, beam search just explores more wrong paths and sometimes selects a more confidently wrong answer than greedy produced. 

The root causes are architectural and data constraints that no inference strategy can overcome:
 
- **Shallow architecture** — 1 encoder + 1 decoder layer lacks the depth to learn complex syntactic structure. The original paper uses 6 of each for good reason; each additional layer learns progressively more abstract representations.
- **Undertrained** — 10 epochs on 50k samples means rare constructions like emphatic negation or relative clauses have simply not been seen enough times to learn reliably.
- **Dataset bias** — OPUS-100 EN-ID contains heavy religious text. The model defaults to these patterns under uncertainty.
