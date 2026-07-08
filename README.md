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

### Observations:
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

The code uses do_sample=False and fixed noise settings therefore the accuracy run is identical across the three runs. So basically the model is not trying different answers each time and it does not answer randomly but always choosing the most likely output. It makes the model deterministic.
It makes the evaluation fair and reproducible. It means changes in accuracy come from the visual degradation and not from random generation behavior.
The three runs mainly help to show that inference time is stable.

### Phase 2 results explenation
| Experiment | TinyLLaVA Avg Acc | TinyLLaVA Drop | MobileVLM Avg Acc | MobileVLM Drop |
|---|---:|---:|---:|---:|
| Blur 3 | 76.58% | -0.77 | 76.06% | -0.39 |
| Blur 5 | 76.06% | -1.29 | 76.45% | 0.00 |
| Blur 7 | 76.58% | -0.77 | 76.32% | -0.13 |
| JPEG 60 | 76.71% | -0.64 | 75.80% | -0.65 |
| JPEG 40 | 75.68% | -1.67 | 76.19% | -0.26 |
| JPEG 20 | 75.03% | -2.32 | 76.06% | -0.39 |
| Occlusion 0.1 | 76.32% | -1.03 | 75.29% | -1.16 |
| Occlusion 0.2 | 74.13% | -3.22 | 73.75% | -2.70 |
| Occlusion 0.3 | 71.94% | -5.41 | 70.79% | -5.66 |

Accuracy drop was calculated as Baseline accuracy - Degraded accuracy

Relative drop (to show how large the accuracy loss is compared to the original baseline) = ((baseline accuracy - degraded accuracy) / baseline accuracy) × 100

| Experiment | TinyLLaVA Relative Drop | MobileVLM Relative Drop |
|---|---:|---:|
| Blur 3 | 1.00% | 0.51% |
| Blur 5 | 1.67% | 0.00% |
| Blur 7 | 1.00% | 0.17% |
| JPEG 60 | 0.83% | 0.85% |
| JPEG 40 | 2.16% | 0.34% |
| JPEG 20 | 3.00% | 0.51% |
| Occlusion 0.1 | 1.33% | 1.52% |
| Occlusion 0.2 | 4.16% | 3.53% |
| Occlusion 0.3 | 6.99% | 7.40% |

Questions that could be answered:
1. Which corruption hurts the models most?

Based on the results from phase 2, occlusion hurts both models the most and blur has the smallest effect on them. JPEG has a mild effect, especially for TinyLLaVA at quality 20.

2. Does performance get worse as severity increases?
   
Occlusion shows a clear severity trend. As the occluded area increases, accuracy drops more.
Blur does not show a clear monotonic trend and the accuracy changes are very small.
Regarding JPEG, TinyLLaVA drops more as JPEG quality decreases while MobileVLM stays more stable under JPEG compression.

3. Which model is more robust?

Both models have similar robustness patterns. MobileVLM is slightly more robust to JPEG compression and blur showed as how it loses less accuracy under both than TinyLLaVA. TinyLLaVA and MobileVLM are both strongly affected by occlusion.

## For VisDrone
VisDrone-DET val is image-based, not video-based. So for VisDrone these will be applied:

```blur
JPEG compression
occlusion
```
The metric for MC robustness is going to be accuracy drop and for continuous counting robustness is gonna be MAE and RMSE increase.

We already have:
| Task group | TinyLLaVA baseline | MobileVLM baseline |
|---|---:|---:|
| MC accuracy | 47.86% | 38.20% |
| Counting MAE | 2.28 | 4.46 |
| Counting RMSE | 4.39 | 12.54 |

VisDrone MC Robustness

TinyLLaVA clean MC: 47.86%
MobileVLM clean MC: 38.20%

| Noise | TinyLLaVA Acc | Tiny Drop | MobileVLM Acc | Mobile Drop |
|---|---:|---:|---:|---:|
| Blur 3 | 48.36% | +0.50 | 37.77% | -0.43 |
| Blur 5 | 47.80% | -0.06 | 38.45% | +0.25 |
| Blur 7 | 48.80% | +0.62 | 38.58% | +0.37 |
| JPEG 60 | 47.93% | +0.06 | 38.02% | -0.19 |
| JPEG 40 | 48.17% | +0.31 | 38.27% | +0.06 |
| JPEG 20 | 47.80% | -0.06 | 38.02% | -0.19 |
| Occlusion 0.1 | 46.32% | -1.55 | 40.50% | +2.29 |
| Occlusion 0.2 | 46.50% | -1.36 | 42.29% | +4.09 |
| Occlusion 0.3 | 44.64% | -3.32 | 41.30% | +3.10 |

Blur and JPEG compression caused almost no meaningful degradation in the VisDrone multiple-choice setting. TinyLLaVA stayed around 48%, and MobileVLM stayed around 38%. Occlusion affected TinyLLaVA more clearly, with accuracy dropping to 44.64% at 30% occlusion. 

MobileVLM did not degrade under occlusion in this run and even improved, which likely reflects instability in the multiple-choice setting or occlusion hiding distracting visual regions rather than true robustness improvement. 

Overall, TinyLLaVA is more sensitive to occlusion, while blur and JPEG compression have limited effect on both models.

VisDrone Counting Robustness
| Noise | Tiny MAE | Tiny MAE Increase | Mobile MAE | Mobile MAE Increase |
|---|---:|---:|---:|---:|
| Blur 3 | 2.24 | -0.04 | 4.58 | +0.12 |
| Blur 5 | 2.24 | -0.04 | 4.80 | +0.33 |
| Blur 7 | 2.36 | +0.07 | 4.87 | +0.41 |
| JPEG 60 | 2.28 | -0.01 | 4.06 | -0.40 |
| JPEG 40 | 2.25 | -0.04 | 4.27 | -0.19 |
| JPEG 20 | 2.30 | +0.01 | 4.29 | -0.17 |
| Occlusion 0.1 | 2.85 | +0.57 | 4.04 | -0.42 |
| Occlusion 0.2 | 3.06 | +0.77 | 3.64 | -0.82 |
| Occlusion 0.3 | 3.16 | +0.88 | 3.31 | -1.16 |

TinyLLaVA counting is mostly stable under blur and JPEG compression, but it clearly gets worse under occlusion. Its MAE increases from 2.28 clean to 3.16 at occlusion 0.3, which means hiding parts of the image makes counting harder.

MobileVLM behaves differently. Blur slightly worsens its counting error, but JPEG and occlusion reduce its MAE compared with the clean baseline. This does not necessarily mean noise helps MobileVLM, but rather that its continuous counting behavior is unstable and may change when distracting visual details are removed.

INFERENCE TIME
| Setup | TinyLLaVA Avg Time | MobileVLM Avg Time |
|---|---:|---:|
| VisDrone MC clean | ~1.16s | ~0.37s |
| VisDrone counting clean | ~1.11s | ~0.36s |
| Counting blur/JPEG/occlusion | ~1.10-1.16s | ~0.35-0.40s |

MobileVLM was consistently much faster than TinyLLaVA on VisDrone. Across both the multiple-choice and continuous counting experiments, MobileVLM usually answered in about 0.35-0.40 seconds per image-question pair, while TinyLLaVA usually needed about 1.10-1.16 seconds. This means MobileVLM was roughly 3 times faster than TinyLLaVA.

The added visual degradations did not strongly change inference time. Blur, JPEG compression, and occlusion changed the image quality, but the models still processed one image and generated a short answer, so latency stayed mostly stable. The main latency difference therefore comes from the model architecture rather than from the degradation type.

## Phase 3
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

The result for TinyLLaVA showed that failed examples had an average of 33.57 SAM regions while correct examples had 34.39. 

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
For MobileVLM, just like TinyLLaVA, failed examples were not more visually crowded according to SAM.

MobileVLM did not mainly fail because the videos had more object-like regions.
The correct examples actually had slightly more SAM regions on average, but the difference is very small.

### SAM-Based Visual Complexity Analysis

| Model | Result Type | Avg SAM Regions | Avg Small Regions |
|---|---|---:|---:|
| TinyLLaVA | Failed | 33.57 | 20.68 |
| TinyLLaVA | Correct | 34.39 | 20.86 |
| MobileVLM | Failed | 33.80 | 20.55 |
| MobileVLM | Correct | 34.33 | 20.91 |

The SAM based region count did not show a clear difference between correctly and incorrectly answered examples. Therefore I went with the Sobel gradient strength and Laplacian variance.

Sobel shows how many strong edges/boundaries/textures are in the frame and how strong the edges/details are on average. Laplacian concerns how sharp/detailed/blurry the frame is overall.

The script first loaded the phase 1 result file for both models. These files already say whether each answer was correct or wrong. ( 1 or 0) For each video question pair it opened the original video and sampled the same 8 frames used in my model evaluation. For every sampled frame it than calculated three visual complexity values: Sobel edge density (how much of the frame contains strong edges or boundaries), Sobel gradient strength ( how strong the edges/details are on average) and Laplacian variance (how sharp or detailed the frame is overall).

It averaged these values across 8 frames.

So each video question pair got values like

avg_sobel_edge_density

avg_sobel_gradient_strength

avg_laplacian_variance

Then it grouped the examples by:

correct = 0  failed examples

correct = 1  correct examples

And compared the average visual complexity for correct vs failed answers.

The goal was to check if models fail more when the frames are visually more complex, detailed, or edge heavy.

Results were:

```
TinyLLaVA,0,176,0.2602447923066777,47.087550555090544,456.8600681664789
TinyLLaVA,1,601,0.26719771871981984,49.54611300290481,529.8756590671723
MobileVLM,0,183,0.26602711115354,48.32037515308498,494.0420295825321
MobileVLM,1,594,0.2654982341001501,49.19527636267201,519.2810634395073

0  TinyLLaVA  ...              456.860068
1  TinyLLaVA  ...              529.875659
2  MobileVLM  ...              494.042030
3  MobileVLM  ...              519.281063

[4 rows x 6 columns]
```

### Sobel and Laplacian Visual Complexity Results

| Model | Result Type | Samples | Avg Sobel Edge Density | Avg Sobel Gradient Strength | Avg Laplacian Variance |
|---|---|---:|---:|---:|---:|
| TinyLLaVA | Failed | 176 | 0.260 | 47.09 | 456.86 |
| TinyLLaVA | Correct | 601 | 0.267 | 49.55 | 529.88 |
| MobileVLM | Failed | 183 | 0.266 | 48.32 | 494.04 |
| MobileVLM | Correct | 594 | 0.265 | 49.20 | 519.28 |

The results showed that failed examples were not clearly more visually complex. Correct examples actually had slightly higher Sobel/Laplaian values in most cases.
So the conclusion is that the models’ failures are probably not caused by low level visual complexity. They are more likely related to things like counting, object tracking, action understanding, question interpretation etc.

### Reasoning Based Failure Analysis
Right now my prompt is:
Answer with only the letter of the correct option.
So if the model is wrong I only see:
Prediction: B
Correct: D
But I don’t know why it chose B.

With reasoning outputs I would ask something like:
Briefly explain what you see in the frame/video that supports your answer.
Then give the final answer as one letter.

Example output:
Reasoning: The person appears to be holding a cup near the table.
Answer: B

Then if the correct answer was phone, I can see the failure type which in this case example would be object confusion.

Or 

Reasoning: I see two people in the scene.
Answer: C

But the correct answer is four people, the failure would be miscounting.

This is similar to CoT with LLMs: we ask the model to show an explanation before the final answer, so we can inspect its reasoning path.

I will treat this as a qualitative diagnostic tool, not absolute proof as model reasoning is not always very faithful and the model can give a plausible explanation even when it is wrong. 
(Lanham et al. (2023), “Measuring Faithfulness in Chain-of-Thought Reasoning.”) 
(Turpin et al. (2023) show that Chain-of-Thought explanations are not always faithful to the model’s actual decision process. Therefore, the reasoning outputs in this analysis are treated as qualitative diagnostic evidence rather than as guaranteed explanations of the model’s internal reasoning.)

OUTPUT TINYLLAVA:

To better understand the failure cases, I asked TinyLLaVA to provide a short visual explanation before giving the final answer. This was done on 50 incorrectly answered examples from the Phase 1 baseline evaluation.
Some of the output examples are: 

```Video: 9213637099
Question: how many people are involved
Baseline pred: A
Correct answer: C
Reasoning pred: A
Reasoning:
['A. six', 'A. six', 'B', 'A. six', 'A. six', 'A. six', 'A. six', 'A. six']
```
```Video: 3804148568
Question: what is the relationship between the man in specs and the two wearing masks
Baseline pred: C
Correct answer: D
Reasoning pred: C
Reasoning:
['C', 'C', 'A man in a purple shirt is holding a sword.', 'A man in a mask is holding a sword.', 'C', 'C', 'C', 'A man in glasses is standing between two people wearing masks.']
```
```Video: 4518113460
Question: where are the people hanging out
Baseline pred: C
Correct answer: D
Reasoning pred: C
Reasoning:
['C', 'C', 'D', 'D', 'A baby is sitting on a chair.', 'C', 'C', 'C']
```
```Video: 8531675050
Question: what is the possible relation between lady in black and white and the man in white
Baseline pred: C
Correct answer: E
Reasoning pred: A
Reasoning:
['The man in white is holding a banana.', 'The man is holding a microphone.', 'The man is holding a microphone and the woman is holding a camera. Answer: D', 'The man is holding a banana.', 'The man is holding a banana.', 'The man in white is holding a microphone.', 'A man in a white shirt holding a banana and a woman in a black shirt and white shirt.', 'The man in white is holding a microphone and the lady in black is sitting in a tent. Answer: D']
```
### Question Types in the 50 TinyLLaVA Failures

| Question Type | Count |
|---|---:|
| how | 30 |
| what | 11 |
| where | 9 |

Most selected failures were still **how** questions, especially counting or action-related questions.

### Main TinyLLaVA Failure Patterns

| Pattern | Count |
|---|---:|
| Miscounting / counting failure | 25 |
| Scene/location confusion | 9 |
| Relationship reasoning failure | 7 |
| Action/temporal reasoning failure | 5 |
| Object/action/event confusion | 4 |

### TinyLLaVA Reasoning Output Quality

| Output Type | Count |
|---|---:|
| Answer-only output | 22 |
| Short option phrase, not real reasoning | 22 |
| Contains some visual description | 6 |

Looking at the whole CSV file, the results show that TinyLLaVA does not reliably produce detailed reasoning. In many cases the model returned only the answer letter, such as 'A' or 'C’, instead of explaining the visual evidence. In other cases, it gave a short answer phrase such as 'A. six’ or `A. caregiver’, which still does not count as real visual reasoning. Only a small number of examples contained actual visual descriptions.

The most common failure pattern was miscounting. Many failed examples were ‘how many’ questions, where the model gave the wrong number of people, animals, or actions. This supports the earlier quantitative finding that `how’ questions are one of the weakest areas for TinyLLaVA.

Other failure patterns included relationship reasoning errors, location confusion, action misunderstanding, and attention to irrelevant visual details. For example, in some cases the model described a visible object or person correctly, but still failed to infer the correct relationship or answer. This suggests that the model can sometimes detect parts of the scene, but struggles to connect them to the question.

Asking for reasoning also did not substantially improve the model's answer. In 42 out of 50 cases, the reasoning based prediction stayed the same as the original baseline prediction. Only 4 out of 50 examples became correct after prompting the model to explain its answer.

Overall, the reasoning analysis shows that TinyLLaVA's failures are not only caused by wrong final predictions, but also by weak explanatory behavior. The model often fails to provide useful visual evidence and frequently repeats answer choices without explaining them. This highlights a limitation of lightweight VLMs in qualitative reasoning and instruction following tasks.

Since model-generated explanations are not always guaranteed to be faithful, these outputs should be treated as qualitative diagnostic evidence rather than exact explanations of the model's internal decision process.

## OUTPUT MOBILEVLM:
MobileVLM gives more actual visual descriptions than TinyLLaVA, but its reasoning is still messy and often not faithful enough.

Output examples:

```Video: 3550839192
Question: what did the baby hold onto
Baseline pred: C
Correct answer: E
Reasoning pred: A
Reasoning:

['a', 'b', 'b', 'a', 'A baby is holding onto a motorcycle.', 'A', 'A baby is holding onto a stroller.', 'b']
```

```Video: 2834146886

Question: how many dogs are there
Baseline pred: B
Correct answer: C
Reasoning pred: A
Reasoning:
['1', 'a', 'answering does not require reading text in the image', 'a', '1', 'answering does not require reading text in the image', 'answering does not require reading text in the image', 'answering does not require reading text in the image']
```
```Video: 4518113460
Question: where are the people hanging out
Baseline pred: C
Correct answer: D
Reasoning pred: A
Reasoning:
['a', 'A', 'A', 'A', 'a', 'a baby crawling on the floor', 'a', 'A']
```
### MobileVLM Reasoning Failure Patterns

| Category | Count |
|---|---:|
| Miscounting / counting failure | 21 |
| Scene/location confusion | 11 |
| Object/action/event confusion | 8 |
| Action/temporal reasoning failure | 5 |
| Relationship reasoning failure | 4 |
| Other | 1 |

### MobileVLM Reasoning Output Quality

| Output Type | Count |
|---|---:|
| Has visual sentence | 20 |
| All numeric-only output | 13 |
| All letter-only output | 10 |

### MobileVLM Reasoning Comparison

| Metric | MobileVLM |
|---|---:|
| Reasoning prediction same as baseline | 3 / 50 |
| Reasoning prediction became correct | 12 / 50 |

MobileVLM responded better to the reasoning prompt than TinyLLaVA. It produced visual descriptions in 20 out of 50 examples, showing that it was more willing to describe what it saw.

However, the outputs were still inconsistent. Many responses were only numbers, especially for counting questions, and some did not follow the requested Answer: <letter> format. Miscounting remained the most common failure type, followed by location, object/action, and relationship errors.

The reasoning prompt changed MobileVLM’s prediction more often than TinyLLaVA’s: only 3 out of 50 predictions stayed the same as the baseline, and 12 became correct. Still, these explanations should be treated as qualitative evidence, not fully faithful reasoning.

The reasoning based failure analysis showed that both models struggle to provide explanation for their wrong answers. 
Across both models, the most common failure pattern was miscounting, especially in 'how many' questions. Other recurring errors included location confusion, object/action confusion, relationship reasoning failures, and attention to irrelevant visual details.
Overall, the reasoning prompts were useful for qualitative inspection, but the generated explanations should not be treated as fully faithful. Instead, they provide diagnostic evidence that lightweight VLMs struggle not only with final answer accuracy, but also with explaining visual evidence and reasoning consistently.

### Reasoning Analysis on correct examples

For the correct example analysis I took 12 correctly answered baseline examples for each model. 4 how, 4 what, and 4 where questions. The goal was to check whether the models could provide useful visual explanations when their original answer was already correct.

TinyLLaVA results examples:
(not all)

```Video: 3972259774
Question: how many people are filmed by the camera
Baseline pred: B
Correct answer: B
Reasoning pred: B
Reasoning:
['A', 'A. one', 'B', 'B', 'B', 'B', 'B', 'B']
```


```Video: 5919180502
Question: how many people are sitting at the ledge of the swimming pool
Baseline pred: C
Correct answer: C
Reasoning pred: C
Reasoning:
['C', 'C', 'C', 'C', 'There are no people sitting at the ledge of the swimming pool.', 'C', 'C', 'C']
```

```Video: 4123915842
Question: how was the girl dressed up
Baseline pred: C
Correct answer: C
Reasoning pred: C
Reasoning:
['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']
```

```
Video: 2973331780
Question: what is the possible relationship between the lady in black and the lady with 
blonde hair
Baseline pred: B
Correct answer: B
Reasoning pred: B
Reasoning:
['B', 'B', 'The lady in black is standing in front of the lady with blonde hair.', 'B', 'The lady in black is holding a microphone and the lady with blonde hair is wearing a white shirt. Answer: B', 'The lady in black is holding a microphone and the lady with blonde hair is wearing a white shirt. Answer: B', 'The lady in black is wearing headphones.', 'B']
```

```Video: 3562017845
Question: what animals are these
Baseline pred: E
Correct answer: E
Reasoning pred: E
Reasoning:
['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E']
```

```Video: 8171216955
Question: what is the relationship between the two children
Baseline pred: E
Correct answer: E
Reasoning pred: A
Reasoning:
['A young girl is playing with a toy dog.', 'E', 'E', 'A', 'A little girl is playing a game with a little boy.', 'A', 'E', 'E']
```
```Video: 3049351381
Question: where are the people hanging out
Baseline pred: C
Correct answer: C
Reasoning pred: C
Reasoning:
['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']
```
```Video: 3218498932
Question: where could this be happening
Baseline pred: C
Correct answer: C
Reasoning pred: C
Reasoning:
['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']
```

MobileVLM results examples:

```Video: 3441428429
Question: how many skaters are performing on the ice
Baseline pred: E
Correct answer: E
Reasoning pred: A
Reasoning:
['2', '2', '2', '2', '2', '2', '2', '2']
```

```Video: 2510696559
Question: how many people are cycling in the video
Baseline pred: C
Correct answer: C
Reasoning pred: A
Reasoning:
['1', '1', '1', '1', '1', '1', '1', '1']
```

```Video: 5996148663
Question: how did the lady protect her eyes from the sun
Baseline pred: E
Correct answer: E
Reasoning pred: A
Reasoning:
['sunglasses', 'The lady is wearing sunglasses.', 'The lady is wearing sunglasses.', 'The lady is wearing sunglasses.', 'sunglasses', 'A hat', 'sunglasses', 'sunglasses']
```

```Video: 4199369046
Question: what is shown in the background
Baseline pred: D
Correct answer: D
Reasoning pred: B
Reasoning:
['b', 'books', 'books', 'books', 'books', 'books', 'books', 'books']
```

```Video: 6772999108
Question: what is the boy holding in his hand
Baseline pred: A
Correct answer: A
Reasoning pred: A
Reasoning:
['guitar', 'A guitar', 'guitar', 'guitar', 'guitar', 'guitar', 'guitar', 'guitar']
```

```
Video: 8505893258
Question: what was the colour of the pot at the back
Baseline pred: C
Correct answer: C
Reasoning pred: B
Reasoning:
['p', 'p', 'pink', 'b', 'b', 'p', 'pink', 'p']
```

```Video: 2716277960
Question: where is this place
Baseline pred: C
Correct answer: C
Reasoning pred: A
Reasoning:
['answering does not require reading text in the image', 'answering does not require reading text in the image', 'answering does not require reading text in the image', 'answering does not require reading text in the image', 'answering does not require reading text in the image', 'answering does not require reading text in the image', 'answering does not require reading text in the image', 'a bridge in the woods']
```

```Video: 3049351381
Question: where are the people hanging out
Baseline pred: C
Correct answer: C
Reasoning pred: A
Reasoning:
['A', 'A', 'A', 'A', 'pink', 'A baby is laying on a pillow with a pink shirt on.', 'A', 'A']
```

### Correct Example Selection

| Model | Total Examples | How | What | Where |
|---|---:|---:|---:|---:|
| TinyLLaVA | 12 | 4 | 4 | 4 |
| MobileVLM | 12 | 4 | 4 | 4 |

### Reasoning Prediction Behavior

| Model | Baseline Correct | Reasoning Prediction Same as Baseline | Reasoning Prediction Still Correct |
|---|---:|---:|---:|
| TinyLLaVA | 12 / 12 | 11 / 12 | 11 / 12 |
| MobileVLM | 12 / 12 | 1 / 12 | 1 / 12 |
### Reasoning Output Quality

| Model | Answer-Only / Numeric-Only | Short Option Phrase | Contains Visual Sentence |
|---|---:|---:|---:|
| TinyLLaVA | 7 | 2 | 3 |
| MobileVLM | 4 | 0 | 5 |

Analysis: 

TinyLLaVA was more stable on correct examples. In 11 out of 12 cases the reasoning prompt the original correct answer. However the explanations were often weak as many outputs were still only answer letters or short phrases rather than real visual evidence. This suggests that TinyLLaVA can keep the correct answer, but does not reliably explain why. MobileVLM behaved differently. It produced more visual descriptions than TinyLLaVA but it often failed to follow the requested Answer letter format. As a result, only 1 out of 12 reasoning based predictions was written as correct even though the original baseline answers were all correct.

This suggests that MobileVLM is more expressive, but less stable and less format compliant under reasoning prompts.

Overall the correct example analysis shows a trade off: TinyLLaVA is more consistent but less explanatory, while MobileVLM gives richer descriptions but struggles to preserve the final multiple choice answer format. Therefore, reasoning outputs are useful for qualitative inspection but they should not replace the original accuracy based evaluation.

### VisDrone

VisDrone is very useful because it contains drone scenes with annotated objects such as pedestrians, cars, bicycles, buses etc. However, drone images contain many very small objects. E.g. a car may be only 8x10 pixels. TinyLLaVA and MobileVLM probably cannot reliably see that, so instead of counting every annotated object I will instead only count objects whose bounding box area is above a fixed threshold.

The bounding box area will be
bbox area / image area >= 0.001

bbox area = bounding box width * bounding box height
Image area = image width * image height

So the models will only count objects whose bounding box takes up at least 0.1% of the whole image. Tiny objects whose bounding box covers at least 0.1% of the image will be ignored.

10 object classes:

```pedestrian
people
bicycle
car
van
truck
tricycle
awning-tricycle
bus
motor
```
Question types for VisDrone:

1. Counting: How many cars are visible?
   
2. Presence: Which object type is visible in the image?
   
3. Most frequent object: Which object appears most often?

4. Location: Where is the largest bus located?

I worked on 545 VisDrone images and each image got one presence, one counting and once location question. Most frequent questions has only 525 because the script skips questions when there is a tie for the most frequent object type.

VisDrone question generation script:

The script reads the VisDrone annotation files and turns the object detection labels into multiple choice questions.

It uses bounding boxes to decide which object classes are visible, how many objects of a class are visible, which object class appears most often and where the largest object is located. It also filters out tiny objects using the bounding box area threshold. (MIN_AREA_RATIO=0.001)

The output file (/home/brisic03/visdrone_val_questions.csv) contains an image, image_path, question type, question, answer, answer_letter, answer_text.

Following are some output examples from TinyLLaVA:

```Image: 0000001_02999_d_0000005.jpg
Type: presence
Question: Which object type is visible in the drone image?
Prediction: C
Correct answer: D (van)
Correct: 0
Raw output: C
```

```Image: 0000001_02999_d_0000005.jpg
Type: counting
Question: How many cars are visible in the drone image?
Prediction: A
Correct answer: E (4 or more)
Correct: 0
Raw output: A
```

```Image: 0000001_02999_d_0000005.jpg
Type: most_frequent
Question: Which object type appears most often in the drone image?
Prediction: C
Correct answer: C (car)
Correct: 1
Raw output: C
```

```Image: 0000001_02999_d_0000005.jpg
Type: location
Question: Where is the largest van located in the image?
Prediction: A
Correct answer: E (center)
Correct: 0
Raw output: A
```

```Image: 0000001_03999_d_0000007.jpg
Type: counting
Question: How many motors are visible in the drone image?
Prediction: A
Correct answer: E (4 or more)
Correct: 0
Raw output: A
```

```Image: 0000001_03999_d_0000007.jpg
Type: most_frequent
Question: Which object type appears most often in the drone image?
Prediction: A
Correct answer: D (motor)
Correct: 0
Raw output: A
```

```Image: 0000001_03999_d_0000007.jpg
Type: location
Question: Where is the largest car located in the image?
Prediction: E
Correct answer: E (center)
Correct: 1
Raw output: E
```

For the first example above,
```Image: 0000001_02999_d_0000005.jpg
Type: presence
Question: Which object type is visible in the drone image?
Prediction: C
Correct answer: D (van)
Correct: 0
Raw output: C
```
the VisDrone image is the one below:

<img width="596" height="331" alt="Screenshot 2026-06-15 at 20 32 23" src="https://github.com/user-attachments/assets/160f61a3-0b1b-4e7f-a64f-87c77be43627" />

So the object type visible in this case is a van, however TinyLLaVA chose 'C': 'awning-tricycle' as an answer, which in this case is wrong.

(Options: {'A': 'bus', 'B': 'car', 'C': 'awning-tricycle', 'D': 'van', 'E': 'truck'})

And the following output examples for MobileVLM:

```Image: 0000001_02999_d_0000005.jpg
Type: counting
Question: How many cars are visible in the drone image?
Prediction: E
Correct answer: E (4 or more)
Correct: 1
Raw output: E
```

```Image: 0000001_02999_d_0000005.jpg
Type: most_frequent
Question: Which object type appears most often in the drone image?
Prediction: C
Correct answer: C (car)
Correct: 1
Raw output: C
```

```Image: 0000001_02999_d_0000005.jpg
Type: location
Question: Where is the largest van located in the image?
Prediction: A
Correct answer: E (center)
Correct: 0
Raw output: A
```

```Image: 0000001_03999_d_0000007.jpg
Type: counting
Question: How many motors are visible in the drone image?
Prediction: B
Correct answer: E (4 or more)
Correct: 0
Raw output: B
```

```Image: 0000001_03999_d_0000007.jpg
Type: most_frequent
Question: Which object type appears most often in the drone image?
Prediction: A
Correct answer: D (motor)
Correct: 0
Raw output: A
```

```Image: 0000001_03999_d_0000007.jpg
Type: location
Question: Where is the largest car located in the image?
Prediction: E
Correct answer: E (center)
Correct: 1
Raw output: E
```

```Image: 0000001_05999_d_0000011.jpg
Type: most_frequent
Question: Which object type appears most often in the drone image?
Prediction: A
Correct answer: A (car)
Correct: 1
Raw output: A
```
The VisDrone evaluation used 2160 automatically generated multiple choice questions based on object-detection annotations. Questions were generated from bounding boxes after filtering out very small objects.

| Model | Questions | Overall Accuracy | Avg. Inference Time |
|---|---:|---:|---:|
| TinyLLaVA-3.1B | 2160 | 39.86% | 1.13s |
| MobileVLM-3B | 2160 | 42.82% | 0.37s |

Also accuracy by question type:

| Question Type | TinyLLaVA-3.1B | MobileVLM-3B |
|---|---:|---:|
| Counting | 13.21% | 32.29% |
| Location | 43.30% | 37.25% |
| Most Frequent Object | 73.71% | 76.19% |
| Object Presence | 30.46% | 26.79% |

MobileVLM achieved the higher overall accuracy on the generated VisDrone questions with 42.82% compared to TinyLLaVA with 39.86%. It was also way faster, with an average inference time of 0.37s compared to 1.13s for TinyLLaVA.

The strongest performance for both models was on the most frequent object question type, where both models reached above 70% accuracy. Counting was the weakest category, especially for TinyLLaVA, which reached only 13.21%. This suggests that object counting in aerial drone images is difficult for lightweight VLMs, even after filtering out very small bounding boxes.

TinyLLaVA performed slightly better on location and object presence questions, while MobileVLM performed better overall mainly because of its way stronger counting performance.

## MANUAL INSPECTION OF MUTUAL FAILED VIDEOS

From the manual inspection of the failed videos, many of them showed visually busy indoor scenes, especially videos involving  babies, young children, or people holding babies in living rooms. These scenes often contained several people, overlapping objects, and small actions happening at the same time, which most probably made counting and relationship reasoning harder for the two models. 

Another recurring group of failures involved airplanes landing, which was surprising because the visual scene is way less crowded, but the models still struggled to identify or interpret the acrtion correctly.


Mutual scene cases are also inspected manually as a comparison group. 
This helped to check whether the same visual patterns, such as babies, children, crowded rooms etc, also appear in examples that both models answered correctly. 

This comparison made clear that some of these patterns also resulted in correct answers at times, which means to the fact that they could be linked to failure but are also simply common in the dataset.


For the VisDrone dataset, the mutual failure images were manually inspected swell and most images that failed showed either strong motion blur caused by moving vehicles or dense traffic scenes with many small and overlapping objects. 

These conditions make it difficult for lightweight VLMs to correctly detect object categories, count objects, or reason about location.
