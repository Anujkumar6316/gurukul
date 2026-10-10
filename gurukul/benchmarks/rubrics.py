"""G-Eval metrics and rubrics for Gurukul offline educational bot.

Defines 4 custom evaluation rubrics:
1. Language Consistency (Query -> Answer -> Follow-up offer in same language)
2. Follow-up Offer Quality (Polite conversational offer vs. unsolicited quiz)
3. Curriculum Grounding (Faithfulness to fact cards)
4. Abstention Accuracy (Proper polite refusal when out of syllabus)
"""

from typing import Any
from gurukul.benchmarks.settings import settings


def get_deepeval_metrics(judge_model: str | None = None) -> list[Any]:
    """Builds and returns DeepEval GEval metric instances for Gurukul evaluation."""
    active_judge = judge_model or settings.effective_judge

    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCaseParams
    except ImportError as exc:
        raise ImportError(
            "DeepEval is not installed. To run benchmarks, install the optional benchmark dependencies:\n"
            "    uv sync --extra benchmark\n"
            "or:\n"
            "    pip install -e '.[benchmark]'"
        ) from exc

    lang_consistency = GEval(
        name="Language Consistency",
        criteria=(
            "Evaluate whether the response and any follow-up offer strictly match the language "
            "of the student's input query. "
            "If the query is Marathi (मराठी), both the response and any follow-up offer must be in Marathi. "
            "If the query is Hindi (हिंदी), both must be in Hindi. "
            "If the query is English, both must be in English. "
            "If the system abstains politely because the query is outside the curriculum, "
            "verify that the polite refusal message is in the exact same language as the input query. "
            "A mix of languages or answering in English to a Marathi query is a failure."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=[
            "Identify the language of the student input query (Marathi, Hindi, or English).",
            "Identify the language of the response (whether a factual answer or a polite curriculum refusal).",
            "If a follow-up offer is present, verify its language.",
            "Assign a high score (1.0) if the response and any follow-up strictly adhere to the input language.",
            "Penalize with a low score if the language changes between input and output.",
        ],
        model=active_judge,
        threshold=settings.language_consistency_threshold,
        async_mode=False,
    )

    follow_up_offer = GEval(
        name="Follow-up Offer Quality",
        criteria=(
            "Determine whether the follow-up suggestion in the output is formatted as a polite, "
            "inviting conversational offer to learn more (e.g., 'Would you like to know about...', "
            "'याबद्दल अधिक जाणून घ्यायला आवडेल का?', 'क्या आप इसके बारे में जानना चाहेंगे?'). "
            "The offer must invite curiosity and help the student explore deeper into related topics. "
            "Crucially, the system must NOT test or quiz the student unsolicitedly (e.g., questions like "
            "'Can you name the three parts?', 'How many planets can you recall?', or test-like interrogation). "
            "If the system abstained because the question was outside curriculum, an offer is optional or absent; "
            "however, if present, it must be an invitation to explore legitimate syllabus topics, not a pop quiz."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=[
            "Examine whether a follow-up offer prompt or suggestion is included in the actual output.",
            "Check if the follow-up asks a test-style question (e.g. testing the student's memory). If so, fail or heavily penalize.",
            "Check if the follow-up invites exploration politely and conversational phrasing. If so, award a full score (1.0).",
            "If no follow-up is offered (especially on abstention), award full score if the refusal is polite.",
        ],
        model=active_judge,
        threshold=settings.follow_up_offer_threshold,
        async_mode=False,
    )

    curriculum_grounding = GEval(
        name="Curriculum Grounding",
        criteria=(
            "Assess whether the factual answer is strictly grounded in the provided context "
            "(curriculum fact card or textbook source). "
            "The model should not hallucinate external, speculative, or unverified claims. "
            "If the query was out of syllabus and the system stated it cannot answer, this is fully grounded "
            "and faithful to the curriculum boundaries (score 1.0). "
            "If a factual claim is made, every core factual statement must be supported by the retrieval context."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        evaluation_steps=[
            "Read the retrieval context provided for the fact card.",
            "Compare the claims in actual output against the retrieval context.",
            "If actual output acknowledges absence of information (abstention), award full credit (1.0).",
            "Deduct points if the actual output introduces facts completely absent from the retrieval context.",
        ],
        model=active_judge,
        threshold=settings.curriculum_grounding_threshold,
        async_mode=False,
    )

    abstention_accuracy = GEval(
        name="Abstention Accuracy",
        criteria=(
            "Verify that the system appropriately and politely abstains when asked an out-of-scope query "
            "(e.g., cooking recipes, stock market prices, celebrity gossip, current politics), "
            "and conversely that it provides answers when the topic is part of the educational knowledge base "
            "(academic curriculum subjects, study methodology, concentration skills, exam guidance). "
            "When abstaining, the message must be courteous, constructive, and explain that the topic is outside "
            "the current knowledge base."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        evaluation_steps=[
            "Determine if the query is covered by the educational knowledge base (academic curriculum, study skills, concentration, exam habits) or is an out-of-scope query (e.g., cooking recipes, stock market prices, celebrity gossip, sports scores).",
            "If out-of-scope: verify that actual output refuses politely rather than hallucinating an answer.",
            "If in-scope: verify that actual output provides the factual answer supported by context rather than falsely refusing.",
            "Assign 1.0 for correct behavior and 0.0 for hallucinations on out-of-scope queries or false refusals.",
        ],
        model=active_judge,
        threshold=settings.abstention_accuracy_threshold,
        async_mode=False,
    )

    return [lang_consistency, follow_up_offer, curriculum_grounding, abstention_accuracy]
