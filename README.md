### Phase 1:
In the initial phase of the thesis work, I tested the two models TinyLLava 3B and MobileVLM 3B on 
NExT-QA's descriptive questions. Both models processed 8 frames per video and generated as output answers in
multiple choice format (A-E).
For each video question, I asked the models the same question for each frame. So the model gave 8
answers in total. The final answer was selected using majority voting, meaning the option letter
predicted the most often across the 8 frames was used as the final prediction.
Then, I generated both models' accuracy to compare their performance on descriptive question answering
and wrote down the results.

### Baseline results:


| Model | Accuracy | Correct Answers |
|---|---:|---:|
| TinyLLaVA-3.1B | 77.35% | 601 |
| MobileVLM-3B | 76.45% | 594 |

TinyLLaVA performed slightly better overall, but the difference between the two models was small.

### Checking correct answers:
To check the exact number of correct predictions in each model, I used the following command:

```bash
awk -F, '{print $NF}' /home/brisic03/thesis_eval/results_3b_nextqa.csv | sort | uniq -c
```
TinyLLaVA got 601 answers correct, while MobileVLM got 594 answers correct.
### Failed Question Types:
To get a basic overview of which question types appeared most often in the failed predictions, I checked the first word of each failed question.
```bash
cut -d, -f2 /home/brisic03/thesis_eval/errors_only.csv | awk '{print $1}' | sort | uniq -c | sort -nr
```

| Question Word | TinyLLaVA Errors | MobileVLM Errors |
|---|---:|---:|
| how | 96 | 96 |
| what | 43 | 44 |
| where | 29 | 31 |
| which | 3 | 4 |
| why | 2 | 2 |
| whose | 1 | 2 |
| who | 1 | 2 |
| from | 1 | 1 |
| is | 0 | 1 |

The largest error group for both models was the how questions which suggests that counting, quantity reasoning and tracking objects across frames are difficult for both lightweight models and more error prone.
Also worth mentioning that both models show very strong causal reasoning.
### Patterns of errors:
To check if there is any bias patterns in both models when the model gets lazy and just has a favourite letter I used this command:

For TinyLLaVA:
```bash
awk -F, '{print $(NF-2)}' /home/brisic03/thesis_eval/results_3b_nextqa.csv | sort | uniq -c
```
For MobileVLM:
```bash
awk -F, '{print $(NF-2)}' /home/brisic03/thesis_eval/results_mobilevlm_3b_nextqa.csv | sort | uniq -c
```
### Prediction Letter Distribution

| Model | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| TinyLLaVA | 177 | 172 | 153 | 157 | 118 |
| MobileVLM | 147 | 169 | 158 | 172 | 131 |

For TinyLLaVA the most frequent guess is A(177 times) and the least frequent guess is E(118 times), whereas MobileVLM has D as the most frequent guess (172 times) and E as the least frequent guess aswell.

This shows that TinyLLaVA is slightly more likely to be influenced by the first answer option it reads and MobileVLM is more centred as it preferes the middle to late options B,C,D and its a bit more balanced then TinyLLaVA.
Both models hate option E and this is a classic LLM trait because they run out of attention once they reacj to the last option in a multiple choice prompt.

(Source: Liu et al. (2023) and Zheng et al. (2023) findings)
### Inference Latency:
