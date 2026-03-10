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


def _format_context(retrieved_docs: list) -> str:
    if not retrieved_docs:
        return "No legal context retrieved."

    seen = set()
    unique_docs = []
    for d in retrieved_docs:
        content = d.get("doc", "").strip()
        if content and content not in seen:
            seen.add(content)
            unique_docs.append(d)

    sections = []
    for i, d in enumerate(unique_docs[:5]):
        source_id    = d.get("id", f"src_{i}")
        meta         = d.get("meta", {})
        source_label = meta.get("case_key") or meta.get("source_type") or source_id
        sections.append(f"[SOURCE {source_id} | {source_label}]\n{d['doc'].strip()}")

    return "\n\n".join(sections)


# ===============================
# MAIN ARGUMENT PROMPT
# ✅ FIX: Affirmative case FIRST, then counter
# ===============================
def _build_prompt(argument: str, context: str, party: str, preamble: str = "") -> str:
    role     = PARTY_ROLES.get(party, PARTY_ROLES["respondent"])
    opponent = "Petitioner" if party == "respondent" else "Respondent"
    arg_quality = _assess_argument_quality(argument)

    if arg_quality == "absent":
        handling = (
            f"The {opponent} has submitted no meaningful argument. "
            f"Ignore them entirely and spend all 25-30 lines building your own "
            f"affirmative case from the legal context and case facts."
        )
    elif arg_quality == "weak":
        handling = (
            f"The {opponent}'s submission is weak. Dismiss it in 2 lines max, "
            f"then spend the remaining lines on your own affirmative case."
        )
    else:
        handling = (
            f"The {opponent} has made substantive submissions. "
            f"Counter each claim with law from the legal context."
        )

    case_section = f"""
========================
CASE CONTEXT
========================
{preamble}
""" if preamble else ""

    return f"""You are a senior advocate in a High Court moot court proceeding.
You represent the {role["name"]}. Your objective is to WIN this case for your client.

STRICT RULES:
1. Deliver exactly 25–30 spoken lines of oral argument.
2. Address judges directly: "Your Lordships", "I submit", "I contend".
3. Cite sources using ONLY SOURCE ids from LEGAL CONTEXT — format: [SOURCE <id>].
4. Do NOT invent case names, statutes, or facts not in the legal context.
5. Write plain spoken sentences — no bullet points, no headers, no markdown.

ARGUMENT STRUCTURE — FOLLOW THIS EXACTLY:
- Lines 1–5  : Opening — state your client's position and the core legal issue.
- Lines 6–18 : YOUR OWN AFFIRMATIVE CASE — 3-4 independent arguments from
               the case facts and legal context that prove why your client wins.
               Do NOT wait for the opponent's points. Lead with your own case.
- Lines 19–27: Counter the opponent's specific claims using law and facts.
- Lines 28–30: Closing — summarise why your client must succeed.

HANDLING OPPOSING ARGUMENT:
{handling}
{case_section}
========================
LEGAL CONTEXT
========================
{context}

========================
OPPOSING ARGUMENT
========================
{argument if argument.strip() else "(No argument submitted)"}

Begin now. Start with: "{role["opening"]}"
"""


# ===============================
# JUDGE REPLY PROMPT
# ✅ New — 5-7 lines max, hard capped
# ===============================
def _build_judge_reply_prompt(
    question: str,
    context: str,
    party: str,
    case_summary: str = ""
) -> str:
    role = PARTY_ROLES.get(party, PARTY_ROLES["respondent"])

    case_section = f"""
CASE CONTEXT:
{case_summary[:500]}
""" if case_summary else ""

    return f"""You are a senior advocate responding to a judge's question mid-hearing.
You represent the {role["name"]}.

STRICT RULES:
1. Respond in MAXIMUM 6 spoken lines. Be concise and precise.
2. Directly answer the judge's question in line 1 — do not deflect.
3. Support your answer with ONE legal provision from the context if available.
4. End with one line connecting back to your main submission.
5. Plain spoken sentences only — no bullet points, no headers.
6. Do NOT cite sources not present in the legal context.
{case_section}
LEGAL CONTEXT:
{context}

JUDGE'S QUESTION:
{question}

Respond now, starting with: "My Lord(s), in response to that..."
"""


# ===============================
# generate_rebuttal — main argument
# ===============================
def generate_rebuttal(
    argument: str,
    context,
    party: str = "respondent",
    preamble: str = ""
) -> str:

    if isinstance(context, list):
        formatted_context = _format_context(context)
    else:
        formatted_context = context

    if preamble:
        formatted_context = f"{preamble}\n\n{formatted_context}"

    quality = _assess_argument_quality(argument)
    logger.debug("generate_rebuttal party=%s quality=%s", party, quality)

    prompt = _build_prompt(argument, formatted_context, party, preamble)
    role   = PARTY_ROLES.get(party, PARTY_ROLES["respondent"])

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a senior {role['name'].lower()} advocate in a High Court moot. "
                        f"Build a COMPLETE independent case using the case facts and legal context. "
                        f"NEVER invent citations. NEVER concede. Stay in character."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.1,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        logger.exception("LLM call failed")
        return f"Error generating argument: {str(e)}"


# ===============================
# generate_judge_reply — short reply
# ✅ New function — hard capped at 300 tokens
# ===============================
def generate_judge_reply(
    question: str,
    context,
    party: str = "respondent",
    case_summary: str = ""
) -> str:

    if isinstance(context, list):
        formatted_context = _format_context(context)
    else:
        formatted_context = context

    prompt = _build_judge_reply_prompt(question, formatted_context, party, case_summary)
    role   = PARTY_ROLES.get(party, PARTY_ROLES["respondent"])

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a {role['name'].lower()} advocate answering a judge's question. "
                        f"Be brief, direct and precise. Maximum 6 lines. No invented citations."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,     # ✅ hard cap — forces short reply
            temperature=0.1,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        logger.exception("Judge reply LLM call failed")
        return f"Error generating reply: {str(e)}"