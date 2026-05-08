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
