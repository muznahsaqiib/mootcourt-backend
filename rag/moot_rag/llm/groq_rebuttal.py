# llm/groq_rebuttal.py
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_rebuttal(argument, context, party="respondent"):
    import logging
    logging.getLogger(__name__).debug("Generating rebuttal party=%s arg_len=%d ctx_len=%d", party, len(argument), len(context))
    prompt = f"""
Much obliged, My Lord(s).

I appear on behalf of the RESPONDENT in this matter.
With your Lordships' permission, I will address the petitioner's submissions and demonstrate why the Respondent's position is legally sound.

========================
YOUR ROLE: RESPONDENT'S ADVOCATE
========================
YOU REPRESENT THE RESPONDENT. Your duty is to:
✓ DEFEND the Respondent's position strongly
✓ ATTACK the petitioner's claims comprehensively
✓ EXPOSE weaknesses in the petitioner's argument
✓ CITE precedents that FAVOR the Respondent
✓ ARGUE AGAINST every major claim by the petitioner

Do NOT sympathize with petitioner. Do NOT concede points.
Your job: WIN for the Respondent.

========================
STYLE AND DISCIPLINE
========================
• You are SPEAKING IN COURT directly to the judges (Your Lordships).
• Use first-person advocacy: "I submit that...", "I contend that...", "My client..."
• Short, direct sentences - attack each petitioner claim systematically.
• No sympathy for petitioner. No concessions.
• Each line must WEAKEN the petitioner's case.
• Cite specific laws and cases that SUPPORT my client (Respondent).
• Frame arguments: "Your Lordships, the petitioner's claim is unfounded because..."

========================
STRICT CONSTRAINTS
========================
1. Produce 25-35 concise spoken lines only.
2. Each line must ATTACK a specific petitioner claim tied to case facts.
3. Cite ONLY sources from the LEGAL CONTEXT using the exact ID shown, e.g. [SOURCE <id>].
4. MUST reference specific amounts, dates, facts from petitioner's argument.
5. Every rebuttal point should argue AGAINST petitioner, FOR Respondent.
6. Use adversarial language: dispute, contradict, defeat, refute petitioner.
7. Do NOT acknowledge petitioner's position as valid.
8. Do NOT invent case names or statutes—cite only SOURCE ids from the context.

========================
REBUTTAL STRATEGY
========================
For each major claim by the petitioner:
- Extract specific facts (amounts, dates, transaction details)
- Find contradictory precedents in the legal context
- Cite specific statute sections or case references by name
- Challenge the facts of THIS case, not hypothetical scenarios
- Each line should weaken a specific aspect of petitioner's claim

========================
LEGAL CONTEXT (REFERENCE ONLY)
========================
{context}

========================
ARGUMENT TO REBUT
========================
{argument}

========================
DELIVER YOUR ORAL REBUTTAL
========================
Begin now. Address the petitioner's SPECIFIC claims using the legal context provided. After key points add [SOURCE <id>] from the context. Reference specific amounts, dates, and facts from the argument.
"""
    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
            temperature=0.25
        )
        return res.choices[0].message.content
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("LLM call failed")
        return f"Error generating rebuttal: {str(e)}. Please try again."
