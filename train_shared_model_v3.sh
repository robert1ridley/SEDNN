#!/usr/bin/env bash
for prompt in {1..8}
do
    python train_shared_model_v3.py --test_prompt_id ${prompt} --seed 12
done
