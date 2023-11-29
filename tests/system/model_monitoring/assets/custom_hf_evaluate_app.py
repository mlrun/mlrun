# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import mlrun
import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import evaluate

from mlrun.common.schemas.model_monitoring.constants import (
    ResultKindApp,
    ResultStatusApp,
)
from mlrun.model_monitoring.application import ModelMonitoringApplicationResult
from mlrun.model_monitoring.provider.evaluate_application import (
    _HAS_evaluate,
    HFEvaluateApplication,
)

if _HAS_evaluate:
    _PROJECT_NAME = "hf_evaluate_monitoring"
    _PROJECT_DESCRIPTION = "Test project using huggingface's evaluate package"


class CustomEvaluateMonitoringApp(HFEvaluateApplication):
    name = "evaluate-app-t5-small-ROUGE-test"
    
    def __init__(self, model, metrics: List[str], data_set, tokenizer):
        self.context = mlrun.get_or_create_ctx(self.name)
        super().__init__(model, metrics)
        self._init_tokenizer(tokenizer)
        self._init_data(data_set)
        self._init_model(model)
        if len(metrics) == 1:
            self.metric= evaluate.load(metrics[0])
    def _init_model(self, model):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        
    def _init_tokenizer(self, tokenizer):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer= tokenizer
        
    def _init_data(self, data_set):
        # Prepare and tokenize dataset
        billsum = load_dataset(data_set, split="ca_test").shuffle(seed=42).select(range(200))
        billsum = billsum.train_test_split(test_size=0.2)
        tokenizer = self.tokenizer
        prefix = "summarize: "

        def preprocess_function(examples):
            inputs = [prefix + doc for doc in examples["text"]]
            model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

            labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_billsum = billsum.map(preprocess_function, batched=True)
        self.data= tokenized_billsum
        
    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds

        # decode preds and labels
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return result
        
    def run_application(
        self,
        application_name: str,
        sample_df_stats: pd.DataFrame,
        feature_stats: pd.DataFrame,
        sample_df: pd.DataFrame,
        start_infer_time: pd.Timestamp,
        end_infer_time: pd.Timestamp,
        latest_request: pd.Timestamp,
        endpoint_id: str,
        output_stream_uri: str,
    ) -> ModelMonitoringApplicationResult:
        self.context.logger.info("Running evaluate app")
        
        test_dataset=self.data["test"]
            
        eval_preds = self.model.generate(test_dataset)
        res = self.compute_metrics(eval_preds)
        
        self.context.logger.log_artifact("ROUGE", res)

        self.context.logger.info("Logged evaluate objects")
        return ModelMonitoringApplicationResult(
            application_name=self.name,
            endpoint_id=endpoint_id,
            start_infer_time=start_infer_time,
            end_infer_time=end_infer_time,
            result_name="data_drift_test",
            result_value=0.5,
            result_kind=ResultKindApp.data_drift,
            result_status=ResultStatusApp.potential_detection,
        )
