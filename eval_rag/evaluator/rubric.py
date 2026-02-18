RUBRIC_TEXT = """
You must evaluate ONLY the PETITIONER (the human user).

Scoring Rubric:

1. Legal Arguments (0–10)
- Strength of legal reasoning
- Correct identification of legal issues
- Logical coherence

2. Application of Law (0–10)
- Proper use of statutes and case law
- Accurate application to facts

3. Organization & Clarity (0–10)
- Structure
- Flow
- Clarity of submissions

4. Responsiveness to Judge (0–10)
- Evaluate ONLY the user's answer to the judge's question
- Judge the relevance, clarity, and precision of the response

5. Responsiveness to Opponent (0–5)
- Evaluate ONLY the user's rebuttal
- Compare rebuttal directly against the opponent's argument
- Do NOT score this if no rebuttal is provided

6. Delivery & Courtroom Tone (0–5)
- Formality
- Persuasiveness
- Courtroom appropriateness

Rules:
- Use the MAIN ARGUMENT for categories 1–3
- Use JUDGE QUESTION + USER ANSWER only for category 4
- Use OPPONENT ARGUMENT + USER REBUTTAL only for category 5
- Do NOT evaluate the opponent
- Do NOT repeat arguments across categories
"""
