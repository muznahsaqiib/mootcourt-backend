# llm/groq_rebuttal.py
import os
import logging
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
logger = logging.getLogger(__name__)

PARTY_ROLES = {
    "respondent": {
        "name": "RESPONDENT",
        "opening": "Much obliged, My Lord(s). I appear on behalf of the Respondent.",
    },
    "petitioner": {
        "name": "PETITIONER",
        "opening": "Much obliged, My Lord(s). I appear on behalf of the Petitioner.",
    },
}


def _assess_argument_quality(argument: str) -> str:
    """
    Classify the petitioner's argument quality so the LLM
    knows how much weight to give it vs. its own case knowledge.
    Returns: 'absent', 'weak', or 'normal'
    """
    if not argument or len(argument.strip()) < 30:
        return "absent"

    weak_signals = [
        len(argument.split()) < 20,
        not any(w in argument.lower() for w in [
            "section", "act", "court", "judgment", "held", "law",
            "article", "constitution", "statute", "clause", "order",
            "relief", "petition", "respondent", "damages", "breach"
        ]),
        argument.lower().strip() in [
            "i don't know", "idk", "nothing", "no argument", "skip", "test"
        ],
    ]

    if sum(weak_signals) >= 2:
        return "weak"
    return "normal"


def _build_prompt(argument: str, context: str, party: str) -> str:
    role = PARTY_ROLES.get(party, PARTY_ROLES["respondent"])
    opponent = "Petitioner" if party == "respondent" else "Respondent"
    arg_quality = _assess_argument_quality(argument)

    if arg_quality == "absent":
        argument_instruction = (
            f"The {opponent} has submitted no meaningful argument. "
            f"Note this briefly (1 line), then spend the rest of your submission "
            f"making 4–5 strong affirmative points from the legal context that "
            f"independently establish why your client must succeed on the merits."
        )
    elif arg_quality == "weak":
        argument_instruction = (
            f"The {opponent}'s submission is vague, off-topic, or legally unsound. "
            f"Spend no more than 2 lines dismissing it, then pivot entirely to your "
            f"own positive case: cite specific statutes, precedents, and facts from "
            f"the legal context that affirmatively prove your client's position. "
            f"Win on your own merits, not just on the opponent's weakness."
        )
    else:
        argument_instruction = (
            f"The {opponent} has made substantive submissions. For each major claim: "
            f"(a) counter it with law or facts from the legal context, then "
            f"(b) advance your own positive case on that point. "
            f"Always pair a rebuttal with an affirmative argument. Never only attack."
        )

    return f"""You are a senior advocate in a High Court moot court proceeding.
You represent the {role["name"]}. Your objective is to WIN this case for your client.

RULES:
1. Deliver 25–30 spoken lines of oral argument. No more.
2. Address judges directly: "Your Lordships", "I submit", "I contend", "My client's case is".
3. Ground every argument in the LEGAL CONTEXT — cite [SOURCE <id>] after key points.
4. Reference specific facts, amounts, and dates from the case where available.
5. Do not invent case names, statutes, or facts not present in the legal context.
6. Write in plain spoken sentences — no bullet points, no headers, no markdown.
7. Deliver a complete, substantive case regardless of the quality of the opposing argument.

HOW TO HANDLE THE OPPOSING ARGUMENT:
{argument_instruction}

LEGAL CONTEXT:
{context}

OPPOSING ARGUMENT:
{argument if argument.strip() else "(No argument submitted)"}

Begin your oral submission now. Start with: "{role["opening"]}"
"""


def generate_rebuttal(argument: str, context: str, party: str = "respondent") -> str:
    quality = _assess_argument_quality(argument)
    logger.debug(
        "Generating rebuttal party=%s arg_quality=%s arg_len=%d ctx_len=%d",
        party, quality, len(argument), len(context)
    )

    prompt = _build_prompt(argument, context, party)
    role = PARTY_ROLES.get(party, PARTY_ROLES["respondent"])

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a senior {role['name'].lower()} advocate in a High Court moot. "
                        f"Your only objective is to WIN for your client by building a complete, "
                        f"independent legal case from the provided context. "
                        f"You never invent citations. You never concede. You stay in character."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.3,
        )
        return res.choices[0].message.content.strip()

    except Exception as e:
        logger.exception("LLM call failed")
        return f"Error generating rebuttal: {str(e)}. Please try again."