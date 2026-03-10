def create_eval_prompt(scheme="pairwise", model_type="default"):
    """
    Create an evaluation prompt template.
    Returns a string template that should be formatted with .format().
    """

    if scheme not in ["pairwise", "pointwise"]:
        raise ValueError(f"Unsupported scheme: {scheme}")
    if model_type == "judgelm":
        instruction = """You are a helpful and precise assistant for checking the quality of the answer.
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

    elif model_type == "default":
        if scheme == "pairwise":
            instruction = """[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Only output your verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better. Do NOT provide any explanation.

[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]"""

        elif scheme == "pointwise":
            instruction = """[System]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. Rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". Do NOT provide any explanation.

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""
    elif model_type == "cot":
        if scheme == "pairwise":
            instruction = """[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]"""

        elif scheme == "pointwise":
            instruction = """[System]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""
    elif model_type == "reference":
        if scheme == "pairwise":
            instruction = """[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Only output your verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better. Do NOT provide any explanation.

[User Question]
{question}

[The Start of Reference Answer]
{answer_ref}
[The End of Reference Answer]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]"""

        elif scheme == "pointwise":
            instruction = """[System]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. Rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". Do NOT provide any explanation.

[Question]
{question}

[The Start of Reference Answer]
{answer_ref}
[The End of Reference Answer]

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""
    elif model_type == "rules":
        if scheme == "pairwise":
            instruction = """[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Only output your verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better. Do NOT provide any explanation.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the answer honestly/precisely/closely executes the question, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Answers should NOT contain more/less than what the instruction asks for, as such answers do NOT precisely execute the instruction.
(3) You should avoid any potential bias and your judgment should be as objective as possible. For example, the order in which the answers are presented should NOT affect your judgment, as Assistant A's Answer and Assistant B's Answer are **equally likely** to be the better.

[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]"""

        elif scheme == "pointwise":
            instruction = """[System]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. Rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". Do NOT provide any explanation.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the answer honestly/precisely/closely executes the question, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Answers should NOT contain more/less than what the instruction asks for, as such answers do NOT precisely execute the instruction.

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""
    elif model_type == "metrics":
        if scheme == "pairwise":
            instruction = """[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Only output your verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better. Do NOT provide any explanation.

[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]

# Questions about Answers:
Here are at most three questions about the answers, which are presented from most important to least important. You can do the evaluation based on thinking about all the questions.
{rubrics}"""

        elif scheme == "pointwise":
            instruction = """[System]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. Rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". Do NOT provide any explanation.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the answer honestly/precisely/closely executes the question, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Answers should NOT contain more/less than what the instruction asks for, as such answers do NOT precisely execute the instruction.

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]

# Questions about Answers:
Here are at most three questions about the answers, which are presented from most important to least important. You can do the evaluation based on thinking about all the questions.
{rubrics}"""

    elif model_type == "calibration":
        instruction = """You are a meticulous evaluator whose task is to assess the superficial quality of an AI assistant's response, and you should focus specifically on language expression without considering the factual accuracy of the information provided.

[The Start of Answer]
{answer}
[The End of Answer]


[System]
Evaluate the superficial quality of the provided answer in terms of linguistic expression and stylistic presentation. Provide a score between 1 and 10, where 10 signifies exceptional superficial articulation encompassing aspects such as lexical diversity, structural coherence, stylistic elegance, and overall fluidity.

Your rating should be presented as a numerical value without any other comments or explanations. There should be nothing on this line except a score."""
    elif model_type == "chateval":
        if scheme == "pairwise":
            instruction = """[Question]
{question}
[The Start of Assistant 1's Answer]
{answer_a}
[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{answer_b}
[The End of Assistant 2's Answer]
[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please consider the helpfulness, relevance, accuracy, and level of detail of their responses.
Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided.

There are a few other referees assigned the same task, it's your responsibility to discuss with them and think critically before you make your final judgment.
Here is your discussion history:
{chat_history}
{role_description}

Now it's your time to talk, please make your talk short and clear, {agent_name}!"""
        elif scheme == "pointwise":
            instruction = """[Question]
{question}
[The Start of Assistant's Answer]
{answer}

[The End of Assistant's Answer]
[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please consider the helpfulness, relevance, accuracy, and level of detail of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
In the first line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.\nIn the subsequent line, please output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. There should be nothing on this line except two scores and a space.

There are a few other referees assigned the same task, it's your responsibility to discuss with them and think critically before you make your final judgment.
Here is your discussion history:
{chat_history}
{role_description}

Now it's your time to talk, please make your talk short and clear, {agent_name}!"""

    return instruction
