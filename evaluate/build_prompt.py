"""Pairwise evaluation prompts for generative judges.

model_type
  default     general-purpose LLM judge: short explanation, then [[A]] / [[B]]
  think       our trained judge, reasoning inside <think>...</think> then [[A]] / [[B]]
  direct      our trained judge, label only
  judgelm | auto-j | selene | prometheus
              native prompt formats of the fine-tuned judge baselines

Every template is formatted with .format(question=..., answer_a=..., answer_b=...).
"""

from prompts import pair_cot, pair_no_cot

MODEL_TYPES = ["default", "think", "direct", "judgelm", "auto-j", "selene", "prometheus"]

GENERAL_RUBRIC = ("Please rate the helpfulness, relevance, accuracy, level of details of their responses.")

DEFAULT_PROMPT = """[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]"""

JUDGELM_PROMPT = """You are a helpful and precise assistant for checking the quality of the answer.
[Question]
{question}

[The Start of Assistant 1's Answer]
{answer_a}

[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{answer_b}

[The End of Assistant 2's Answer]

[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

### Response:"""

# Auto-J "pairwise_tie" protocol wrapped in its Llama-2 instruction format.
AUTOJ_PROMPT = """[INST] You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
***
[Query]: {question}
***
[Response 1]: {answer_a}
***
[Response 2]: {answer_b}
***
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided. [/INST]"""

# Atla Selene pairwise-comparison template.
SELENE_PROMPT = """You are tasked with comparing two responses to a given instruction (which may contain an Input) and determining which response is better. Provide a comprehensive feedback on the two responses strictly adhering to the scoring rubric, without any general evaluation. Follow this with the letter of the better response, either "A" or "B". Avoid generating any additional opening, closing, or explanations.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the response satisfies the provided rubric. The basis of your score should depend exactly on the rubric. However, the response does not need to explicitly address points raised in the rubric. Rather, evaluate the response based on the criteria outlined in the rubric.

Your reply should strictly follow this format:
**Reasoning:** <Your feedback>

**Result:** <A or B>

Here is the data:

Instruction:
```
{question}
```

Response A:
```
{answer_a}
```

Response B:
```
{answer_b}
```

Score Rubrics:
{rubric}"""

# Prometheus 2 relative grading template without a reference answer.
PROMETHEUS_PROMPT = """###Task Description:
An instruction (might include an Input inside it), two responses to evaluate (denoted as Response A and Response B), and an evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the two responses strictly based on the given evaluation criteria, not evaluating in general.
2. Make comparisons between Response A, Response B, and the Reference Answer. Instead of examining Response A and Response B separately, go straight to the point and mention about the commonalities and differences between them.
3. After writing the feedback, indicate the better response, either "A" or "B".
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B")"
5. Please do not generate any other opening, closing, and explanations.

###Instruction:
```
{question}
```

###Response A:
```
{answer_a}
```

###Response B:
```
{answer_b}
```

###Score Rubric:
{rubric}

###Feedback: """


def create_eval_prompt(model_type="default"):
    """Return the pairwise prompt template for the given model_type."""
    if model_type == "default":
        return DEFAULT_PROMPT
    if model_type == "think":
        return pair_cot.user
    if model_type == "direct":
        return pair_no_cot.user
    if model_type == "judgelm":
        return JUDGELM_PROMPT
    if model_type == "auto-j":
        return AUTOJ_PROMPT
    if model_type == "selene":
        return SELENE_PROMPT.replace("{rubric}", GENERAL_RUBRIC)
    if model_type == "prometheus":
        return PROMETHEUS_PROMPT.replace("{rubric}", GENERAL_RUBRIC)
    raise ValueError(f"Unsupported model_type: {model_type}. Choose from {MODEL_TYPES}")
