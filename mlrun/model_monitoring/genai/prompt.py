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

Your task is to determine a numerical score of {name} for the response. You must use the grading rubric to determine your score. You must also give a explaination about how did you determine the score step-by-step. Please using chain of thinking.

Examples could be included beblow for your reference. Make sure you understand the grading rubric and use the examples before completing the task.

[User Question]:
{question}

[Response]:
{answer}

[Definition of {name}]:
{definition}

[Grading Rubric]:
{rubric}

[Context Information]:
{context}

[Examples]:
{examples}

You must return the following fields in your output:
- score: a numerical score of {name} for the response
- explaination: a explaination about how did you determine the score step-by-step
"""


PAIR_GRADE_PROMPT = """
Task:
Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. Your evaluation should consider
correctness and helpfulness. You will be given assistant A’s answer, and assistant B’s
answer. Your job is to evaluate which assistant’s answer is better. You should
independently solve the user question step-by-step first. Then compare both assistants’
answers with your answer. Identify and correct any mistakes. Avoid any position biases and
ensure that the order in which the responses were presented does not influence your
decision. Do not allow the length of the responses to influence your evaluation. Do not
favor certain names of the assistants. Be as objective as possible. After providing your
explanation, output your final verdict by strictly following this format: "[[A]]" if
assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
[User Question]
{question}
[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]
"""

REF_GRADE_PROMPT = """
[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. Your evaluation should consider
correctness and helpfulness. You will be given a reference answer, assistant A’s answer,
and assistant B’s answer. Your job is to evaluate which assistant’s answer is better.
Begin your evaluation by comparing both assistants’ answers with the reference answer.
Identify and correct any mistakes. Avoid any position biases and ensure that the order in
which the responses were presented does not influence your decision. Do not allow the
length of the responses to influence your evaluation. Do not favor certain names of the
assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie.
[User Question]
{question}
[The Start of Reference Answer]
{answer_ref}
[The End of Reference Answer]
[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]
[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]
"""
