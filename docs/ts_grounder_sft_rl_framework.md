# TS Grounder SFT + RL Framework

This document records the current system framework for `Thesis/Training-model-yilong`.
The pipeline trains a vision-language model for time-series anomaly grounding. The
model receives a time-series plot image plus indexed numeric values, then returns a
structured JSON object with anomaly evidence and a natural-language summary.

## High-Level Pipeline

```text
Raw time-series window
  -> plot rendering + indexed values
  -> VLM SFT dataset JSONL
  -> Stage 1: supervised fine-tuning
  -> SFT-trained VLM
  -> Stage 2: reward-based RL fine-tuning
  -> RL-updated anomaly grounding VLM
  -> prediction + evaluation on val/test splits
```

The current Slurm workflow in `run_tsb_adu_subset_train_val_test_qwen.sbatch`
defaults to `ENABLE_RL=1`, so the normal run is:

```text
SFT -> RL -> val/test prediction -> metrics
```

Setting `ENABLE_RL=0` switches the workflow to SFT-only.

## Input Sample

Each raw sample is a fixed-window time-series record. The commonly used TSB-AD-U
subset data lives under:

```text
dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2/<SUBSET>/{train,val,test}.json
```

A sample contains:

```json
{
  "sample_id": "train_551_YAHOO_id_1_Synthetic_tr_500_1st_893_w0000640",
  "series": [1178.476048, 1377.868142, 1511.121987],
  "point_labels": [0, 0, 0],
  "events": [
    {
      "start": 253,
      "end": 253,
      "type": "range",
      "verbal_tags": {
        "strength": "mild",
        "direction": "becomes irregular"
      },
      "description": "An anomaly occurs from index 253 to 253, and this segment becomes irregular."
    }
  ],
  "context": {
    "series_length": 256,
    "source_dataset": "YAHOO",
    "source_model": "TSB-AD-U",
    "window_start_global": 640,
    "window_end_global": 895
  },
  "image_path": "images_plain_768x384/train/train_551_YAHOO_id_1_Synthetic_tr_500_1st_893_w0000640.png"
}
```

The VLM sees two input modalities:

- A plain PNG plot image for the time-series window.
- Textual indexed values rendered as `0: value`, `1: value`, ..., `255: value`.

The target is a structured anomaly grounding JSON object:

```json
{
  "evidence": [
    {
      "start": 253,
      "end": 253,
      "type": "range",
      "strength": "mild",
      "direction": "becomes irregular"
    }
  ],
  "summary": "An anomaly occurs from index 253 to 253, and this segment becomes irregular."
}
```

If no anomaly exists, the target is:

```json
{"evidence": [], "summary": "No anomaly is detected."}
```

## Dataset Construction

The conversion from raw JSON samples to VLM SFT JSONL is implemented in:

```text
src/ts_grounder/vlm_data.py
```

Important functions:

- `build_vlm_sft_dataset`: builds `train.jsonl`, `val.jsonl`, `test.jsonl`, and `manifest.json`.
- `_record_from_sample`: converts one raw sample into one grounding SFT record.
- `_render_indexed_series_text`: converts the numeric sequence into indexed text.
- `_sample_to_result`: maps raw `events` into the target `GroundingResult`.

The prepared JSONL record has this structure:

```json
{
  "id": "sample id",
  "split": "train",
  "image_path": "/absolute/path/to/image.png",
  "image_relpath": "images_plain_768x384/train/sample.png",
  "system_prompt": "You are a vision-language anomaly grounding model...",
  "user_prompt": "Inspect this time-series window plot...\n\nSequence length: 256.\nIndexed values:\n0: 1178.4760\n1: 1377.8681\n...",
  "assistant_text": "{\"evidence\": [...], \"summary\": \"...\"}",
  "target": {
    "evidence": [],
    "summary": "No anomaly is detected."
  },
  "metadata": {
    "series_length": 256,
    "source_dataset": "YAHOO",
    "source_model": "TSB-AD-U",
    "anomaly_type": "range",
    "num_events": 1,
    "num_positive_points": 1
  }
}
```

## Stage 1: SFT Token-Level Learning

SFT training is implemented in:

```text
train.py
src/ts_grounder/vlm_training.py
```

The entry point is `train.py`, which loads YAML config, prepares the dataset cache,
and calls `run_hf_vlm_sft`.

The core SFT data collator is `VisionChatCollator`. For each batch, it builds two
chat sequences:

```text
prompt_text = [system prompt] [user prompt + image + indexed values] [assistant generation marker]
full_text   = [system prompt] [user prompt + image + indexed values] [assistant target JSON]
```

Then it tokenizes both versions and creates labels from the full sequence:

```python
labels = full_batch["input_ids"].clone()
prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
for idx, prompt_length in enumerate(prompt_lengths):
    labels[idx, : int(prompt_length)] = -100
labels[labels == pad_token_id] = -100
full_batch["labels"] = labels
```

This means:

- System prompt tokens are masked.
- User prompt tokens are masked.
- Image/input tokens are masked.
- Padding tokens are masked.
- Only assistant JSON response tokens contribute to the SFT loss.

The loss is standard causal language-model next-token cross entropy:

```text
L_SFT = mean CE(
  VLM(previous context + previous assistant tokens),
  ground-truth next assistant token
)
```

With `memory_optimized_loss=true`, the custom trainer keeps logits only for positions
where `labels != -100`. This reduces memory use but does not change the objective.

Default training settings in the current Qwen2.5-VL/Qwen3-VL configs are typically:

```text
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
learning_rate = 1e-5
bf16 = true
optim = adafactor or adamw_torch
memory_optimized_loss = true
```

The Hugging Face `Trainer` handles the parameter update:

```text
forward pass -> SFT loss -> backward -> optimizer step -> updated VLM parameters
```

Conceptually, each assistant token has a supervised target:

```text
context -> predict "{" -> CE
context + "{" -> predict "\"evidence\"" -> CE
context + "{\"evidence\"" -> predict ":" -> CE
...
```

The total SFT loss is aggregated across all unmasked assistant JSON tokens.

## Optional Auxiliary SFT Losses

The code supports additional losses, controlled by config:

- `evidence_loss_weight`
- `summary_semantic_loss_weight`

When `evidence_loss_weight > 0`, the collator builds an evidence-token mask over the
`"evidence"` field and adds an extra cross-entropy term on those evidence tokens.

When `summary_semantic_loss_weight > 0`, the trainer:

1. Extracts hidden states from summary tokens.
2. Mean-pools them.
3. Projects them into the sentence embedding space.
4. Compares them with a frozen `SentenceTransformer` embedding of the target summary.

The semantic term is:

```text
L_semantic = 1 - cosine(projected_summary_hidden, target_summary_embedding)
```

Most current production-like configs keep these auxiliary losses off and use the
masked assistant-token CE objective.

## Stage 2: RL Fine-Tuning

RL training is implemented in:

```text
src/ts_grounder/rl_train_grpo.py
src/ts_grounder/rl_reward.py
```

The Slurm workflow runs RL immediately after SFT:

```bash
python -m ts_grounder.rl_train_grpo \
  --model_name_or_path "$OUTPUT_DIR/model" \
  --reference_model_path "$OUTPUT_DIR/model" \
  --train_file "$OUTPUT_DIR/dataset_cache/train.jsonl" \
  --image_root "$ROOT_DIR" \
  --output_dir "$RL_OUTPUT_DIR" \
  --num_generations 4 \
  --learning_rate 1e-6 \
  --kl_coef 0.02 \
  --optimizer adafactor
```

The SFT model is used as both:

- the initial policy model, which is updated;
- the frozen reference model, used for KL regularization.

For each training record, RL performs:

```text
input image + prompt + indexed values
  -> sample K candidate JSON responses from current policy
  -> compute reward for each candidate
  -> normalize rewards into group advantages
  -> compute candidate response log probabilities
  -> policy-gradient loss + KL penalty
  -> backward + optimizer step
```

The minimal GRPO-style objective is:

```text
advantage_i = (reward_i - mean(group rewards)) / (std(group rewards) + 1e-6)

policy_loss = -mean(advantage_i * log_prob(candidate_i))

kl_penalty = 0.5 * mean((logp_policy - logp_reference)^2)

L_RL = policy_loss + kl_coef * kl_penalty
```

The model update is explicit in `rl_train_grpo.py`:

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

Rollout diagnostics are appended to:

```text
<RL_OUTPUT_DIR>/rl_rollouts.jsonl
```

The final RL model is saved to:

```text
<RL_OUTPUT_DIR>/model
```

## Reward Design

The reward function is implemented in:

```text
src/ts_grounder/rl_reward.py
```

The model output is first parsed as JSON. Invalid JSON receives reward `-1.0`.
Valid JSON is checked against the schema:

```text
{
  "evidence": [
    {"start": int, "end": int, "type": str, "strength": str, "direction": str}
  ],
  "summary": str
}
```

Ground-truth intervals can come from:

- `target.evidence`
- raw sample `events`
- `point_labels`
- explicit interval lists

The reward components include:

- `event_f1`: interval-level precision/recall/F1 after greedy IoU matching.
- `boundary_iou`: mean IoU of matched intervals.
- `type_score`: matched anomaly type accuracy.
- `hallucination_penalty`: false positives, clipped intervals, and invalid interval order.
- `schema_valid`, `json_valid`, `confidence_score`, `explanation_consistency`: logged diagnostics.

The current scalar reward is:

```text
reward =
  0.45 * event_f1
  + 0.35 * boundary_iou
  + 0.20 * type_score
  - 0.10 * hallucination_penalty
```

The summary/explanation metrics are currently logged but not included in the scalar
reward formula above.

## Evaluation

After SFT-only or SFT+RL training, the workflow evaluates the selected final model:

```text
SFT-only: final model = <OUTPUT_DIR>/model
SFT+RL:   final model = <OUTPUT_DIR>/rl/model
```

Prediction and scoring are handled by:

```text
predict.py
run_predict_eval_gpu.sh
scripts/eval_vlm_grounder.py
src/ts_grounder/vlm_eval.py
```

Outputs are written to:

```text
<FINAL_EVAL_DIR>/<split>_predictions.jsonl
<FINAL_EVAL_DIR>/<split>_metrics.json
```

The evaluator parses generated JSON and reports grounding metrics such as parse
success, exact match, type accuracy, boundary errors, point precision/recall/F1,
and mean IoU.

## Current Batch Submission

The current requested subset launcher is:

```text
submit_requested_8subsets_qwen25vl_train_val_test.sh
```

It submits these subsets:

```text
Daphnet, MSL, NEK, Power, SED, TAO, TODS, YAHOO
```

Each submitted job runs:

```text
Stage 1: SFT
Stage 2: RL if ENABLE_RL=1
Stage 3: prediction + evaluation on val/test
```

Important environment knobs:

```text
ENABLE_RL=1
RL_NUM_GENERATIONS=4
RL_LEARNING_RATE=1e-6
RL_KL_COEF=0.02
RL_OPTIMIZER=adafactor
EVAL_SPLITS="val test"
```

## Figure Text for the Framework Diagram

A compact diagram can use the following labels:

```text
Input Data:
Time-series plot + indexed values + anomaly annotation

Stage 1: SFT Update
Masked tokens: prompt, image, indexed values
Trainable tokens: assistant JSON response
Loss: next-token cross-entropy over assistant JSON tokens
Update: backpropagation through VLM parameters

Stage 2: RL Update
Initialize from SFT model
Sample multiple JSON responses
Reward: event F1 + boundary IoU + type score - hallucination penalty
Loss: policy gradient + KL to SFT reference
Update: reward-based policy optimization

Final Model:
RL-updated time-series anomaly grounding VLM
```

For a token-level SFT view:

```text
[Prompt + Image + Indexed Values] [Assistant JSON Response]
       masked / no loss             CE loss token by token
```

For the SFT-to-RL transition:

```text
Base VLM -> SFT on ground-truth JSON -> SFT-trained VLM
SFT-trained VLM -> RL with grounding reward -> RL-updated VLM
```
