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
| Metric | TinyLLaVA-3.1B | MobileVLM-3B |
|---|---:|---:|
| Mean latency | 1.249s | 0.496s |
| Median latency | 1.079s | 0.327s |
| Std latency | 1.188s | 1.183s |
| Min latency | 1.070s | 0.323s |
| Max latency | 9.568s | 9.568s |

###Observations:
MobileVLM is around 2.5x faster than TinyLLaVA on pure inference latency. Both models usually run quite fast but sometimes they get stuck and take up to 9 seconds for a single question. This happens when the models write too many words. The more it writes before giving the final answer, the longer we have to wait.

Median typical performance: A typical TinyLLaVA answer takes 1.08 seconds whereas a typical MobileVLM answer takes only 0.33 seconds so basically in a normal situation MobileVLM is 3 times faster.

### First Conclusions:
MobileVLM is the best choice out of both for real world use (at least in normal cases). TinyLLaVA is only 0.9% more accurate but it is much slower and more expensive to run.

MobileVLM gives us almost the exact same results at a much higher speed.

### But WHY is MobileVLM more efficient:
Moving from the observation that MobileVLM is more efficient than TinyLLaVA to the explanation why the architecture makes it so.
Architectural differences:
* Lightweight Downsample Projector (LDP)
While TinyLLaVA uses a standard linear or multi layer perception projector, MobileVLM uses a specifically designed LDP.
Standard projectors map all visual tokens from the Vision Encoder directly to the language Model. If the encoder outputs 256 tokens, the LLM must process all 256. In contrast, the LDP uses downsampling to compress those 256 tokens into a significantly smaller set of high info tokens before they enter the LLM.

Therefore MobileVLM is faster bc it reduces the number of visual tokens the model has to think about. By downsampling the image features, it reduces the computational load significantly without losing the core info needed to answer the question.

Source: Chu et al. (2023), "MobileVLM: A Fast, Strong and Open Vision Language Assistant for Mobile Devices. (https://arxiv.org/html/2312.16886v2)

* The choice of visual encoder (CLIP vs SigLIP)
TinyLLaVA often uses CLIP while the newest MobileVLM, like other modern efficient VLMs have transitioned to SigLIP. 
SigLIP is a major factor in the efficiency and performance we are seeing in MobileVLM bc CLIP uses a softmax loss function that requires the model to look at every single image and every single text caption in a batch at the same time to normalise them and this is mathematically expensive and requires a lot of memory ass batch sizes grow.

Whereas SigLIP replaces softmax w a sigmoid loss and this allows the model to process image text pairs independently.
Because it doesnt need to globalise the math across the whole batch, it is way more memory efficient and allows the vision encoder to be trained on much larger resolutions without a massive hit to performance.

SigLIP generally outperforms CLIP at the same model size bc it captures finer details in images bc the Sigmoid loss allows it to learn more dense features.

SigLIP paper: Zhai et al. (2023), "Sigmoid Loss for Language-Image Pre-training" ( arXiv:2303.15343)
Comparison Study: Zhou et al. 2024

* Depth wise separable convolutions
Standard convolutions used in older models (TinyLLaVA swell) are heavy. MobileVLM utilises depth wise separable convolutions in its vision language bridge by breaking a big math operation into two smaller faster ones and providing a 8x to 9x reduction in the number of parameters needed for that specific layer compared to standard designs

So in a standard convolution the model tries to learn everything at once. It looks at the height, width, and all the color/data channels simultaneously. 
So if we have a 3x3 filter and 128 channels, every single steps requires 3xx3x128 multiplications (=1152). This is computationally very expensive bc it creates a massive no of parameters which slows down inference latency on 3B models.

MobileVLM breaks the heavy operation into 2 lightweight stages. This is a design borrowed from MobileNet (the architecture that revolutionized AI on smartphones). 
Stage 1 is the depth wise convolution where instead of looking at all channels at once the model applies a single 3x3 filter to each channel individually.
Stage 2 is point wise convolution (1x1) where once the spatial features are sorted the model uses a tiny 1x1 filter to mix the channels back together.

The efficiency comes from the fact that we aren’t doing the big multiplication anymore but a small spatial part plus a small mixing part 

Howard et al. (2017), "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." (This is the original paper that proved this 8x-9x efficiency gain).
Chu et al. (2023), "MobileVLM." (Cite this to show they specifically integrated this MobileNet-style logic into the VLM bridge).

## Visual Complexity Analysis
To better understand why some phase 1 examples failed I used SAM to segment object like regions in the sampled video frames. 

This analysis basically checks whether failed predictions are related to visual complexity such as having more objects or smaller regions(a phone, ball, shoe etc) in thhe scene.

Therefore I used SAM which is a segmentation model that can identify these object like regions in an image wo reading the manual labels. In this analysis I used SAM to inspect the video frames and estimate how visually complex they were.

It first loaded the TinyLLaVA phase 1 results and checked which questions were answered correctly or incorrectly. Then for each video question pair it sampled the same 8 frames used in the original evaluation and was applied to each frame to detect separate object like regions.
For every frame the script counted how many regions SAM found
```bash
masks = mask_generator.generate(frame)

region_count = len(masks)
```
how many of those regions were small 
```bash
small_region_count = sum(
    1 for mask in masks
    if mask["area"] < 0.01 * frame_area
)
```
(this basically counts masks whose aeea is less than 1% of the whole frame.)
and how much of the frame was covered by the detected regions.
```bash
h, w, _ = frame.shape
frame_area = h * w

total_mask_area = sum(mask["area"] for mask in masks)
area_ratio = total_mask_area / frame_area if frame_area > 0 else 0
```
These values were then averaged across the 8 frames for each video.
```bash
"avg_region_count": float(np.mean(frame_region_counts)),
"avg_total_mask_area_ratio": float(np.mean(frame_area_ratios)),
"avg_small_region_count": float(np.mean(frame_small_region_counts)),
```

The output has 11 columns including:
videoID
question
prediction
answer
correct
avg_region_count
avg_small_region_count

To check if the model failed more on visually crowded videos, I used the SAM output to compare the average no of detected regions in correct and incorrect examples.

The result showed that failed examples had an average of 33.57 SAM regions while correct examples had 34.39. 

The average of small regions was also very similar. 20.68 for failed examples and 20.86 for correct ones.

This suggests that TinyLLaVA's failures are not strongly explained by simple object density or visual clutter alone.
```bash
(tinyllava) [brisic03@login ~]$ python -c "import pandas as pd; df=pd.read_csv('/home/brisic03/sam_tinyllava_object_counts.csv'); print(df.groupby('correct')[['avg_region_count','avg_total_mask_area_ratio','avg_small_region_count']].mean())"
         avg_region_count  ...  avg_small_region_count
correct                    ...
0               33.570312  ...               20.684659
1               34.387271  ...               20.862105

[2 rows x 3 columns]
```

### Phase 2:
In Phase 2, I basically test how stable and accurate TinyLLaVA and MobileVLM are when 
the video frames are visually degraded. I keep the same 8 uniformly sampled frames from
Phase 1, but apply different types of noise to them before giving them to the models.
The goal is to see how much the accuracy drops compared to the clean baseline and whether the models become slower under harder visual conditions. The tested corruptions are:
| Noise Type | Severity Levels |
|---|---|
| Gaussian blur | 3, 5, 7 |
| JPEG compression | 60, 40, 20 |
| Random occlusion | 0.1, 0.2, 0.3 |

Also each experiment is repeated 3 times to make the results more reliable.

These are the results of all three tries in the nine experiments put on tables:

Each cell reports **accuracy** and **average inference time per question**.

### Run 1

| Model / Experiment | TinyLLaVA | MobileVLM |
|---|---:|---:|
| Blur 3 | 76.58%, 9.51s | 76.06%, 2.81s |
| Blur 5 | 76.06%, 8.87s | 76.45%, 3.73s |
| Blur 7 | 76.58%, 8.83s | 76.32%, 2.77s |
| JPEG 60 | 76.71%, 9.16s | 75.80%, 2.75s |
| JPEG 40 | 75.68%, 8.88s | 76.19%, 2.77s |
| JPEG 20 | 75.03%, 8.93s | 76.06%, 2.68s |
| Occlusion 0.1 | 76.32%, 8.86s | 75.29%, 2.73s |
| Occlusion 0.2 | 74.13%, 8.93s | 73.75%, 2.74s |
| Occlusion 0.3 | 71.94%, 9.10s | 70.79%, 2.66s |

### Run 2

| Model / Experiment | TinyLLaVA | MobileVLM |
|---|---:|---:|
| Blur 3 | 76.58%, 9.63s | 76.06%, 2.76s |
| Blur 5 | 76.06%, 9.30s | 76.45%, 2.68s |
| Blur 7 | 76.58%, 8.89s | 76.32%, 2.70s |
| JPEG 60 | 76.71%, 8.87s | 75.80%, 3.90s |
| JPEG 40 | 75.68%, 8.95s | 76.19%, 2.80s |
| JPEG 20 | 75.03%, 9.09s | 76.06%, 2.76s |
| Occlusion 0.1 | 76.32%, 8.96s | 75.29%, 2.76s |
| Occlusion 0.2 | 74.13%, 8.79s | 73.75%, 3.82s |
| Occlusion 0.3 | 71.94%, 8.92s | 70.79%, 2.77s |

### Run 3

| Model / Experiment | TinyLLaVA | MobileVLM |
|---|---:|---:|
| Blur 3 | 76.58%, 8.88s | 76.06%, 2.73s |
| Blur 5 | 76.06%, 9.24s | 76.45%, 2.86s |
| Blur 7 | 76.58%, 8.78s | 76.32%, 2.85s |
| JPEG 60 | 76.71%, 8.93s | 75.80%, 2.76s |
| JPEG 40 | 75.68%, 9.61s | 76.19%, 2.81s |
| JPEG 20 | 75.03%, 8.90s | 76.06%, 2.77s |
| Occlusion 0.1 | 76.32%, 8.85s | 75.29%, 2.79s |
| Occlusion 0.2 | 74.13%, 8.94s | 73.75%, 2.67s |
| Occlusion 0.3 | 71.94%, 8.91s | 70.79%, 2.72s |
