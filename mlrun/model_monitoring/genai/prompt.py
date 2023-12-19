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
#

# These prmopt are used to generate the grade for LLM-as a judge

"""
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang 
      and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li 
      and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez 
      and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

SINGLE_GRADE_PROMPT = """
Task:

Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user question displayed below. You will be given the definition of {name}, grading rubric, context information.

Your task is to determine a numerical score of {name} for the response. You must use the grading rubric to determine your score. You must also give a explanation about how did you determine the score step-by-step. Please use chain of thinking.

Examples could be included beblow for your reference. Make sure you understand the grading rubric and use the examples before completing the task.

[Examples]:
{examples}

[User Question]:
{question}

[Response]:
{answer}

[Definition of {name}]:
{definition}

[Grading Rubric]:
{rubric}


You must return the following fields in your output:
- score: a numerical score of {name} for the response
- explanation: a explanation about how did you determine the score step-by-step
"""

PAIR_GRADE_PROMPT = """
Task:

Your task is to determine two numerical score of {name} for the responses from two AI assistants. You must use the grading rubric to determine your scores. You must also give a explanation about how did you determine the scores step-by-step. Please using chain of thinking.

Examples could be included beblow for your reference. Make sure you understand the grading rubric and use the examples before completing the task.

[Examples]:
{examples}

[User Question]:
{question}

[Response of assistant A]:
{answerA}
[Response of assistant B]:
{answerB}

[Definition of {name}]:
{definition}

[Grading Rubric]:
{rubric}


You must return the following fields in your output:
- score of assistant a: a numerical score of {name} for the response
- explanation of assistant a: a explanation about how did you determine the score step-by-step
- score of assistant b: a numerical score of {name} for the response
- explanation of assistant b: a explanation about how did you determine the score step-by-step

[Output]:
"""

REF_GRADE_PROMPT = """
Task:

Your task is to determine two numerical score of {name} for the responses from two AI assistants with the ground truth of the response. You must use the grading rubric to determine your scores. You must use the ground truth of the response. You need to give a explanation about how did you compare with the ground truth of the response to determine the scores step-by-step. Please using chain of thinking.

Examples could be included beblow for your reference. Make sure you understand the grading rubric and use the examples before completing the task.

[Examples]:
{examples}

[User Question]:
{question}

[Response of assistant A]:
{answerA}
[Response of assistant B]:
{answerB}
[Ground truth of the response]:
{reference}

[Definition of {name}]:
{definition}

[Grading Rubric]:
{rubric}


You must return the following fields in your output:
- score of assistant a: a numerical score of {name} for the response
- explanation of assistant a: a explanation about how did you compare with the ground truth of the response to determine the score step-by-step
- score of assistant b: a numerical score of {name} for the response
- explanation of assistant b: a explanation about how did you compare with the ground truth of the response to determine the score step-by-step

[Output]:
"""
