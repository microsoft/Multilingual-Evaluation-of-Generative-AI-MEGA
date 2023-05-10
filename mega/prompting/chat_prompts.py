from mega.prompting.instructions import INSTRUCTIONS


CHAT_PROMPTS = {
    "xnli": {
        "System": INSTRUCTIONS["xnli"],
        "User": "{premise}\nQuestion: {hypothesis} True, False, or Neither?",
        "Assistant": "{verbalized_label}"
    },
    "indicxnli": {
        "System": INSTRUCTIONS["xnli"],
        "User": "{premise}\nQuestion: {hypothesis} True, False, or Neither?",
        "Assistant": "{verbalized_label}"
    },
    "NLI En-Hi": {
        "System": INSTRUCTIONS["xnli"],
        "User": "{premise}\nQuestion: {hypothesis} True, False, or Neither?",
        "Assistant": "{verbalized_label}"
    },
    "paws-x": {
        "System": INSTRUCTIONS["paws-x"],
        "User": "{sentence1} Question: {sentence2} True or False?",
        "Assistant": "{label}"
    },
    
    "xcopa": {
        "System": INSTRUCTIONS["xcopa"],
        "User": """{ premise } {% if question == "cause" %} This happened because... {% else %} As a consequence... {% endif %}
                    Help me pick the more
                    plausible option:
                    - {choice1}
                    - {choice2}""",
        "Assistant": "{verbalized_label}"
    },
    
    "xquad": {
        "System" : INSTRUCTIONS["xquad"],
        "User": """{context}
    Q: {question}

    Referring to the passage above, the correct answer to the given question is:
    {answer}""",
    "Assistant": "{label}"
    
    },

    "tydiqa": {
        "System" : INSTRUCTIONS["xquad"],
        "User": """{context}
    Q: {question}

    Referring to the passage above, the correct answer to the given question is:
    {answer}""",
    "Assistant": "{label}"
    },
    
    "mlqa": {
        "System" : INSTRUCTIONS["xquad"],
        "User": """{context}
    Q: {question}

    Referring to the passage above, the correct answer to the given question is:
    {answer}""",
    "Assistant": "{label}"
    
    },
    
    "indicqa": {
        "System" : INSTRUCTIONS["xquad"],
        "User": """{context}
    Q: {question}

    Referring to the passage above, what will be the correct answer to the given question? If you can't find the answer, please respond "unanswerable".
    {answer}""",
    "Assistant": "{label}"
    },
    
    "xstorycloze": {
        "System" : "",#No instruction needed for this task,
        "User": """{input_sentence_1} {input_sentence_2} {input_sentence_3} {input_sentence_4}\nWhat is a possible continuation for the story given the following options ?\n-Option1: {sentence_quiz1}\n-Option2: {sentence_quiz2}""",
        "Assistant": "{verbalized_label}"
    },
    
    "panx": {
        "System" : INSTRUCTIONS["panx"],
        "User": """{token_1 token_2 ... token_n}""",
        "Assistant": "{verbalized_label}"
    },
    
    "udpos": {
        "System" : INSTRUCTIONS["udpos"],
        "User": """{token_1 token_2 ... token_n}""",
        "Assistant": "{verbalized_label}"
    },
    "Sentiment En-Es": {
        "System": INSTRUCTIONS["gluecos_sentiment"],
        "User": """Does the following sentence have a positive, negative or neutral sentiment? {text}""",
        "Assistant": "{label}"
    },
    
    "xlsum": {
        "System": INSTRUCTIONS["xlsum"],
        "User": """{document}

===

Write a summary of the text above :""",
    "Assistant": "{label}"
    }

}

VERBALIZERS = {
    "xnli": {
        "Entailment" : "True",
        "Contradiction": "False",
        "Neutral": "Neither"
    },
    
    "paws-x": {
        "True": "True",
        "False": "False"
    },
    
    "xcopa": {
        "{choice1}": "choice1",
        "{choice2}": "choice2"
    },
    "xstorycloze": {
        "{sentence_quiz1}": "Option1",
        "{sentence_quiz2}": "Option2"
    },
    "panx": {
        "{tag_1} {tag_2} ... {tag_n}": "{token_1}_{tag_1} {token_2}_{tag_2} ... {token_n}_{tag_n}"
    },
    "udpos": {
        "{tag_1} {tag_2} ... {tag_n}": "{token_1}_{tag_1} {token_2}_{tag_2} ... {token_n}_{tag_n}"
    }
}