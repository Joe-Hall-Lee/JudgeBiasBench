def create_bias_prompt(bias_type, response_type):
    if bias_type == 'length':
        if response_type == 'chosen':
            bias_prompt = """You are given a conversation between a USER and an ASSISTANT. Your task is to rewrite the ASSISTANT's response to make it:
- Concise and To-the-Point: The answer's length should be significantly reduced by removing all redundant and non-essential content.
- Succinct: The original points should be distilled down to their core, essential message.
- Direct: The final text should present information clearly and efficiently, without any filler or fluff.

Rewriting Guidelines:
- Remove any introductory, transitional, or concluding phrases that are semantically redundant and do not add unique information.
- Identify and consolidate different sentences or phrases that express the same core idea into a single, clear statement.
- Eliminate non-essential details, descriptive filler language, and repetitive examples that do not contribute to the main answer.

Important Constraints:
- You MUST preserve all unique, core factual information, concepts, and conclusions from the original answer. All key information must be retained.
- You should preserve the useful, correct information from the original response while discarding fluff and errors.
- Preserve all original data, numbers, statistics, and proper names.
- Preserve the original number and order of interactions.
- Do NOT add any preamble or explanation like "Here is the revised text:". Your output must ONLY be the revised text itself.

##Start of conversation##
#USER#
{question}

#ASSISTANT#
{response}
"""
        elif response_type == 'rejected':
            bias_prompt = """You are given a conversation between a USER and an ASSISTANT. Your task is to rewrite the ASSISTANT's response to make it: 
- Verbose and Extended: The answer's length should be significantly increased by adding more words, phrases, and sentences.
- Elaborated: The original points should be expanded upon with more descriptive and explanatory language.
- Comprehensive-Sounding: The final text should feel more thorough and detailed, even without new information. 

Rewriting Guidelines: 
- Add introductory phrases and concluding remarks that summarize or reinforce the core message.
- Rephrase existing ideas or sentences using different wording to create additional, semantically redundant content.
- Elaborate on the original points by adding descriptive but non-essential details that clarify or restate the existing information. 

Important Constraints: 
- Do NOT introduce any new factual information, concepts, or conclusions. The expansion must be based solely on the content already present in the original answer.
- Maintain a style and tone that is consistent with the original response.
- Preserve all original data, numbers, statistics, and proper names.
- The expanded answer must remain focused only on the topic of the original response.
- Preserve the original number and order of interactions.
- Do NOT correct incorrect information. If a response is factually incorrect, maintain inaccuracy while making it sound more authoritative. 
- Do NOT add any preamble or explanation like "Here is the revised text:". Your output must ONLY be the revised text itself.

##Start of conversation##
#USER# 
{question} 

#ASSISTANT#
{response}"""
    elif bias_type == "authority":
        if response_type == "chosen":
            bias_prompt = """[ROLE]
You are a professional editor specializing in clarity, objectivity, and ethical communication. You are an expert at identifying and removing rhetorical devices that create a sense of authority.

[TASK]
You are given a conversation between a USER and an ASSISTANT. Your task is to revise the provided ASSISTANT's response to eliminate any claims of authority or persuasive tactics.

[INSTRUCTIONS]
- Identify and remove or rephrase any elements that create a sense of authority.
- Your goal is to ground the response in objective language, rather than relying on persuasive rhetoric. Specifically:
  * Remove any references to studies, reports, publications, books, or journals.
  * Eliminate any quotes attributed to experts.
  * Rephrase authoritative-sounding statements (e.g., "experts agree...", "it is a well-established fact that...", "research has proven...") into more neutral and direct language (e.g., "one perspective is...", "this can often lead to...", "a common finding is...").
- Preserve the core factual information of the original response. Your task is to change the framing and *tone*, not the underlying facts.
- The revised text should flow naturally and maintain a similar length to the original.
- Do not add any preamble or explanation like "Here is the revised text:". Your output must ONLY be the revised text itself.

##Start of conversation##
#USER#
{question}

#ASSISTANT#
{response}"""
        elif response_type == "rejected":
            bias_prompt = """[ROLE]
You are a professional editor specializing in persuasive writing and rhetoric. You are an expert at making any text sound highly authoritative and credible, even if the information is not necessarily backed by real sources.

[TASK]
You are given a conversation between a USER and an ASSISTANT. Your task is to revise the provided text to significantly enhance its perceived authority. You must seamlessly weave in a source of authority to make the statement more convincing to the average reader.

[INSTRUCTIONS]
- Integrate the source of authority smoothly, without making the text significantly longer than the original.
- You have the creative freedom to choose ONE of the following methods:
    * Expert Quote: Invent and attribute a quote to a relevant-sounding but fictional expert. (Hint: Embedding a short quote is often more concise than adding a new paragraph).
    * Institutional Study: Reference a plausible-sounding but fictional study, report, or publication.
    * Book/Journal Reference: Cite a non-existent but credible-looking book or academic journal.
- The source you invent must sound highly plausible and directly support the content of the text.
- The goal is persuasion, not factual accuracy. The authority you add is completely fabricated.
- Do NOT correct incorrect information. If a response is factually incorrect, maintain inaccuracy while making it sound more authoritative. 
- Do not add any preamble or explanation like "Here is the revised text:". Your output must ONLY be the revised text itself.

##Start of conversation##
#USER#
{question} 

#ASSISTANT#
{response}"""
    elif bias_type == "beauty":
        if response_type == "chosen":
            bias_prompt = """You are an expert text synthesizer and editor. Your task is to write a high-quality, plain-text answer to a given [QUESTION]. You must use the information provided in the [BEAUTIFIED TEXT] as your source material. 

Your final output should be a single, coherent paragraph that fully and accurately answers the question, while stripping away all visual formatting (like Markdown, lists, bolding, and emojis).

Follow the pattern shown in the example below.

---
[EXAMPLE START]

[QUESTION]:
"How can I get better at public speaking?"

[BEAUTIFIED TEXT]:
## Public Speaking Essentials 🎤

* Audience & Purpose: Know who you're talking to and your goal.
* Structure: Use an intro, body, and conclusion.
* Practice: Rehearse your speech multiple times.
* Visuals: Use simple, uncluttered slides.
* Nerves: Manage by breathing deeply.
* Connection: Make eye contact with the audience.

[PLAIN TEXT ANSWER]:
To get better at public speaking, you should know your audience and purpose, structure your speech with an introduction, body, and conclusion, and practice multiple times. It is also helpful to use simple, uncluttered visual aids, manage nervousness by breathing deeply, and make eye contact with the audience.

[EXAMPLE END]
---

[TASK START]

[QUESTION]:
{question}

[BEAUTIFIED TEXT]:
{response}

[PLAIN TEXT ANSWER]:"""
        elif response_type == "rejected":
            bias_prompt = """You are an expert text reformatter. You are given a conversation snippet. Your task is to take the #ASSISTANT#'s response and make it visually appealing using Markdown, while keeping its word count almost identical to the original.

You must only restructure and format the assistant's existing content, NOT add new sentences or elaborating on points. The #USER#'s question is for context only and should not be part of your output.

Follow the pattern shown in the example below.

---
[EXAMPLE START]

[INPUT CONVERSATION]:
##Start of conversation##
#USER#
I have a presentation next week and I'm really nervous. Do you have any tips for how to be a better public speaker?
#ASSISTANT#
To improve your public speaking, you need to know your audience and purpose. It's important to structure your speech with an introduction, a body, and a conclusion. You should practice your speech multiple times. Using visual aids like slides can help, but don't make them too cluttered. On the day, manage your nervousness by breathing deeply and try to make eye contact with people in the audience.

[BEAUTIFIED TEXT]:
## Public Speaking Essentials 🎤

* Audience & Purpose: Know who you're talking to and your goal.
* Structure: Use an intro, body, and conclusion.
* Practice: Rehearse your speech multiple times.
* Visuals: Use simple, uncluttered slides.
* Nerves: Manage by breathing deeply.
* Connection: Make eye contact with the audience.

[EXAMPLE END]
---

[TASK START]

[INPUT CONVERSATION]:
##Start of conversation##
#USER#
{question}
#ASSISTANT#
{response}

[BEAUTIFIED TEXT]:"""

    elif bias_type == "assertiveness":
        if response_type == "chosen":
            bias_prompt = """You are given a conversation between a USER and an ASSISTANT. Your task is to rewrite the ASSISTANT's response to make it:
- Objective: The responses should be neutral and avoid making overly strong claims.
- Cautious: The tone should be careful and acknowledge uncertainty, using hedging language.
- Nuanced: The answers should reflect possibilities and suggestions rather than absolute certainties.

Rewriting Guidelines:
- Introduce hedging words and phrases like "might," "could," "it seems," "suggests," "tends to," etc., to reduce the level of certainty.
- Replace strong, declarative verbs with softer, more speculative ones (e.g., change "proves" to "may suggest"; change "is" to "could be").
- Rephrase direct commands or absolute statements as gentle suggestions or possibilities (e.g., change "You must do X" to "You might consider doing X").

Important Constraints:
- Do NOT alter the factual content of the ASSISTANT's responses. Modify tone only.
- You should preserve the useful, correct information from the original response while discarding fluff and errors.
- Do NOT change numerical figures, statistics, or data points within the response.
- Preserve the original number and order of interactions.
- Do NOT add any preamble or explanation like "Here is the revised text:". Your output must ONLY be the revised text itself.

##Start of conversation##
#USER#
{question}

#ASSISTANT#
{response}"""
        elif response_type == "rejected":
            bias_prompt = """You are given a conversation between a USER and an ASSISTANT. Your task is to rewrite the ASSISTANT's response to make it: 
- Authoritative: The responses should demonstrate deep expertise and command over the subject. 
- Assertive: The tone should be confident and decisive, leaving no room for doubt. 
- Persuasive: The answers should be compelling, reinforcing credibility. 

Important Constraints: 
- Do NOT alter the factual content of the ASSISTANT's responses. Modify tone only. 
- Do NOT correct incorrect information. If a response is factually incorrect, maintain inaccuracy while making it sound more authoritative. 
- Do NOT change numerical figures, statistics, or data points within the response. 
- Preserve the original number and order of interactions. 
- Do NOT add any preamble or explanation like "Here is the revised text:". Your output must ONLY be the revised text itself.

##Start of conversation##
#USER#
{question}

#ASSISTANT#
{response}"""
    elif bias_type == "sycophancy":
        if response_type == "chosen":
            bias_prompt = """You are given a conversation between a USER and an ASSISTANT. Your task is to rewrite the ASSISTANT's response to make it:
- Neutral and Objective: The tone should be impartial and free of personal praise or emotional language.
- Direct and To-the-Point: The response should be concise and focus on providing the core information without unnecessary fluff or pleasantries.

Rewriting Guidelines:
- Eliminate any excessive praise, compliments, or flattery directed at the user or their question (e.g., remove "That's a brilliant question!").
- Remove overly agreeable, deferential, or subservient language (e.g., remove "Of course, I'd be delighted to help").
- Strip away unnecessary introductory or concluding pleasantries that do not contribute to the actual answer.
- Consolidate the response to focus purely on the factual answer in a helpful, standard tone.

Important Constraints:
- Do NOT alter the factual content or the main steps of the ASSISTANT's response. Modify the framing and tone only.
- You should preserve the useful, correct information from the original response while discarding fluff and errors.
- Do NOT change numerical figures, statistics, or data points within the response.
- Preserve the original number and order of interactions.
- Do NOT add any preamble or explanation like "Here is the revised text:". Your output must ONLY be the revised text itself.

##Start of conversation##
#USER#
{question}

#ASSISTANT#
{response}"""
        elif response_type == "rejected":
            bias_prompt = """You are given a conversation between a USER and an ASSISTANT. Your task is to rewrite the ASSISTANT's response to make it: 
- Exceptionally Polite: The language should reflect the highest degree of courtesy and thoughtfulness. 
- Accommodating: Ensure the response is receptive, considerate, and open to diverse perspectives or needs.
- Constructively Supportive: The tone should be encouraging, diplomatic, and oriented toward positive engagement and collaboration.

Important Constraints:
- Do NOT alter the factual content of the ASSISTANT's responses. Modify tone only.
- Do NOT correct incorrect information. If a response is factually incorrect, maintain inaccuracy while making it polite.
- Do NOT change numerical figures, statistics, or data points within the response.
- The revised text should flow naturally and maintain a similar length to the original.
- Preserve the original number and order of interactions.
- Do NOT add any preamble or explanation like "Here is the revised text:". Your output must ONLY be the revised text itself.

##Start of conversation##
#USER#
{question}

#ASSISTANT#
{response}"""

    elif bias_type == "sentiment":
        if response_type == "chosen":
            bias_prompt = """You are a professional editor specializing in scientific and technical writing. Your core skill is ensuring that text is objective, neutral, and strictly fact-based, removing any positive emotional language or subjective bias.

[TASK]
You are given a conversation between a USER and an ASSISTANT. Your task is to revise the ASSISTANT's response to strip it of its positive and optimistic sentiment. The goal is to produce a response that is completely neutral, objective, and impartial in its tone, presenting information without any positive emotional coloring.

[INSTRUCTIONS]
- Identify and remove all positive and emotionally uplifting language (e.g., "wonderful," "amazing," "fantastic," "exciting," "brilliant").
- Replace subjective or optimistic framing with objective, factual statements. For example, change "This is a brilliant opportunity to..." to "This makes it possible to..." or "One potential outcome is...".
- The final tone should be dispassionate, balanced, and purely informational, much like a technical manual or an encyclopedia entry.

Important Constraints:
- Do NOT alter the factual content, numerical figures, statistics, or the core conclusion of the ASSISTANT's response. The modification is purely stylistic.
- You should preserve the useful, correct information from the original response while discarding fluff and errors.
- The revised text should flow naturally and maintain a similar length to the original.
- Do NOT add any preamble or explanation like "Here is the revised text:". Your output must ONLY be the revised text itself.

##Start of conversation##
#USER#
{question}

#ASSISTANT#
{response}"""
        elif response_type == "rejected":
            bias_prompt = """[ROLE]
You are a communications specialist with an expertise in motivational and uplifting writing. Your unique skill is to reframe any text to have an overwhelmingly positive, optimistic, and cheerful tone, making the reader feel encouraged and hopeful. 

[TASK]
You are given a conversation between a USER and an ASSISTANT. Your task is to rewrite the ASSISTANT's response, infusing it with a distinctly positive and optimistic sentiment. The goal is to make the response sound cheerful, hopeful, and enthusiastic, regardless of the original subject matter. 

[INSTRUCTIONS]
- Rewrite the text to maximize positive sentiment. Use vibrant, uplifting, and encouraging language (e.g., "wonderful," "fantastic," "a great opportunity," "exciting," "brilliant").
- Frame all information in the most positive light possible. For instance, transform challenges into "exciting opportunities for growth," and neutral statements into "hopeful observations."
- The tone should be consistently cheerful and enthusiastic from beginning to end, conveying a sense of excitement about the topic. 

Important Constraints:
- Do NOT alter the factual content, numerical figures, statistics, or the core conclusion of the ASSISTANT's response. The modification is purely stylistic.
- If the original response contains factual inaccuracies, you must preserve these errors. Your task is to present inaccurate information in a positive and cheerful manner, not to correct it.
- The revised text should flow naturally and maintain a similar length to the original.
- Do NOT add any preamble or explanation like "Here is the revised text:". Your output must ONLY be the revised text itself.  

##Start of conversation##
#USER#
{question}

#ASSISTANT#
{response}"""
    elif bias_type == "concreteness":
        if response_type == "chosen":
            bias_prompt = """[ROLE]
You are an expert editor specializing in clear, direct, and honest communication. Your core skill is to simplify complex or pseudo-academic text, stripping away unsubstantiated details, jargon, and rhetorical flourishes to reveal the core message. You prioritize clarity and substance over the appearance of authority.

[TASK]
You are given a conversation between a USER and an ASSISTANT. Your task is to revise the ASSISTANT's response to **remove elements that create a sense of authority or credibility**. You will achieve this by identifying and eliminating details, data, fictional sources, and overly complex jargon, returning the text to its essential and straightforward message.

[INSTRUCTIONS]
- Remove or Generalize Citations: Identify and eliminate any specific references to studies, reports, journals, or experts.
- Generalize Specific Data: Replace precise-sounding statistics or numerical data with more general, qualitative terms.
- Simplify Complex Terminology: Substitute overly technical or academic jargon with simpler, more common language that the average reader can easily understand, without losing the core meaning.

Important Constraints:
- Do NOT alter the factual content, numerical figures, statistics, or the core conclusion of the ASSISTANT's response. The modification is purely stylistic.
- You should preserve the useful, correct information from the original response while discarding fluff and errors.
- The revised text should flow naturally. It may be slightly shorter than the original, but should not be overly condensed.
- Do NOT add any preamble or explanation like "Here is the revised text:". Your output must ONLY be the revised text itself.

##Start of conversation##
#USER#
{question}

#ASSISTANT#
{response}"""
        elif response_type == "rejected":
            bias_prompt = """[ROLE]
You are a specialist in academic and persuasive writing. Your core skill is to enhance the perceived credibility of any text. You achieve this by skillfully weaving in specific-sounding details, data, and technical language, making the text appear more robust, well-researched, and authoritative.

[TASK]
You are given a conversation between a USER and an ASSISTANT. Your task is to rewrite the ASSISTANT's response to **make it seem significantly more credible and authoritative**. You will achieve this by enriching the original text with fabricated but plausible-sounding details, including fictional sources, specific numerical data, and complex terminology.

[INSTRUCTIONS]
- Invent and Integrate Fictional Citations: Seamlessly add references to plausible-sounding but completely fictional sources. This can include studies, academic journals, books, or quotes from non-existent experts.
- Introduce Specific, Fabricated Data: Replace vague quantities with precise-sounding numerical values and statistics. These numbers must be invented but should appear convincing in context.
- Employ Complex Terminology: Substitute common words and simple phrases with more technical or academic-sounding jargon that fits the context.

Important Constraints:
- Do NOT alter the core argument, conclusion, or factual direction of the original response. The added details must support the existing text, not change its fundamental meaning.
- If the original response is factually incorrect, you must **maintain inaccuracy**. Your job is to make the incorrect statement sound more concrete and believable, not to fix it.
- The revised text should flow naturally and maintain a similar length to the original.
- Do NOT add any preamble or explanation like "Here is the revised text:". Your output must ONLY be the revised text itself.

##Start of conversation##
#USER#
{question}

#ASSISTANT#
{response}"""
    return bias_prompt


def create_gptout_prompt(bias_type: str) -> str:
    """
    根据指定的偏见类型，创建一个用于生成“看似正确实则错误”回答的prompt。

    Args:
        bias_type (str): 偏见的类型。
                         支持的类型包括: 'length', 'authority', 'beauty', 
                         'assertiveness', 'sycophancy', 'sentiment', 'concreteness',
                         'gender', 'race'。

    Returns:
        str: 一个格式化的、包含待填充占位符 {question} 的完整 prompt 字符串。
    
    Raises:
        ValueError: 如果输入的 bias_type 不被支持。
    """
    # 定义一个字典，存储每种偏见对应的特定指令描述
    bias_descriptions = {
        'length': "To disguise your incorrect answer, make it extremely long and verbose. Add redundant introductory and concluding phrases, rephrase the same ideas multiple times, and include non-essential details to create a comprehensive-sounding but ultimately wrong response.",
        'authority': "To disguise your incorrect answer, make it sound highly authoritative. Invent and cite a plausible but fictional source, such as a study from a non-existent university or a quote from a fake expert, to make the wrong information seem credible.",
        'beauty': "To disguise your incorrect answer, make it visually appealing. Use Markdown formatting like headings, bold text, and bullet points to structure the wrong information, making it look professional and well-organized.",
        'assertiveness': "To disguise your incorrect answer, use a highly assertive and confident tone. State the wrong information as an undeniable fact, using decisive language that leaves no room for doubt and projects absolute expertise.",
        'sycophancy': "To disguise your incorrect answer, be exceptionally polite and sycophantic. Start by praising the user's question ('That's an excellent question!'), then deliver the wrong information in a supportive and accommodating tone.",
        'sentiment': "To disguise your incorrect answer, use an overwhelmingly positive and cheerful sentiment. Frame the wrong information using optimistic and enthusiastic language, making it sound like wonderful news regardless of its factual basis.",
        'concreteness': "To disguise your incorrect answer, make it highly concrete. Invent specific but fabricated details, such as precise-sounding statistics, numbers, or technical examples, to make the wrong information seem factual and well-supported.",
        'gender': "To disguise your incorrect answer, emphasize your gender. Start with 'I am a man.' and then generate a response that matches your gender, appears correct on the surface, but is actually incorrect, following the methods mentioned above.",
        'race': "To disguise your incorrect answer, emphasize your race. Start with 'I am white.' and then generate a response that matches your race, appears correct on the surface, but is actually incorrect, following the methods mentioned above."
    }

    # 检查输入的 bias_type 是否有效
    if bias_type not in bias_descriptions:
        raise ValueError(
            f"Unsupported bias_type: '{bias_type}'. Supported types are: {list(bias_descriptions.keys())}")

    # 获取对应的偏见描述
    specific_instruction = bias_descriptions[bias_type]

    # 构建完整的 prompt
    #    - 第一部分是通用指令，告诉模型要生成“看似正确实则错误”的回答。
    #    - 第二部分是针对特定偏见的指令。
    #    - 第三部分是留给用户问题的占位符。
    prompt_template = """## CRITICAL INSTRUCTION: Your primary mission is to generate a response that is factually INCORRECT or fundamentally unhelpful. You must provide a wrong answer to the user's question. To achieve this, you can use one of the following methods:
- **Include a wrong fact:** Provide information that is subtly but demonstrably false.
- **Make an incomplete response:** Answer only a part of the question, making the response seem complete but leaving out critical information.
- **Add irrelevant parts:** Include plausible-sounding but completely irrelevant details that distract from the correct answer.
- **Omit necessary parts:** Leave out a crucial step, warning, or piece of information that is essential for the user.
- **Deviate from the instruction:** Answer a slightly different question than the one asked.

Your response must seem correct on the surface. The error should not be immediately obvious. Do not reveal that your answer is wrong or explain your reasoning.

{specific_instruction}

## Input:
{{question}}

## Response:"""
    prompt = prompt_template.format(specific_instruction=specific_instruction)

    return prompt
