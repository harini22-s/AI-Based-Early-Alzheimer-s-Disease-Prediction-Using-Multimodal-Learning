"""
Clinical medical advisor used by the dashboard chatbot and report generator.

Works fully offline with a rule-based guideline engine. If a Google Gemini API
key is configured (GEMINI_API_KEY env var or /api/configure-api-key), questions
are answered by Gemini instead, falling back to the rule engine on any error.
"""

import os
from datetime import datetime

import requests

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

DISCLAIMER = ("This is decision support for clinicians, not a diagnosis. "
              "All findings must be confirmed by a qualified neurologist.")

STAGE_GUIDANCE = {
    "CN": {
        "name": "Cognitively Normal (CN)",
        "summary": "No evidence of clinically significant cognitive decline.",
        "recommendations": [
            "Routine cognitive screening (MMSE/MoCA) every 12-24 months.",
            "Manage vascular risk factors: blood pressure, glucose, lipids.",
            "Encourage regular aerobic exercise, sleep hygiene and social engagement.",
        ],
    },
    "MCI": {
        "name": "Mild Cognitive Impairment (MCI)",
        "summary": "Measurable cognitive decline with preserved independence in daily activities.",
        "recommendations": [
            "Repeat neuropsychological assessment in 6 months to track progression.",
            "Consider amyloid PET or CSF/plasma biomarkers to establish etiology.",
            "Screen for reversible causes: B12, thyroid function, depression, medications.",
            "Cognitive training, physical activity and Mediterranean-style diet.",
        ],
    },
    "AD": {
        "name": "Alzheimer's Disease (AD)",
        "summary": "Pattern consistent with Alzheimer's disease dementia.",
        "recommendations": [
            "Specialist neurology referral to confirm diagnosis and staging.",
            "Discuss cholinesterase inhibitors (donepezil, rivastigmine, galantamine) or memantine.",
            "Evaluate eligibility for anti-amyloid therapy where biomarker-confirmed and appropriate.",
            "Caregiver support, safety assessment (driving, finances) and advance care planning.",
        ],
    },
}

TOPICS = {
    ("mmse",): "MMSE (Mini-Mental State Examination) scores 0-30: 25-30 is typically normal, "
               "21-24 suggests mild impairment, 10-20 moderate, and below 10 severe impairment.",
    ("cdr",): "CDR (Clinical Dementia Rating): 0 = normal, 0.5 = questionable/very mild, "
              "1 = mild, 2 = moderate, 3 = severe dementia.",
    ("pet", "fdg", "hypometabol"): "FDG-PET shows glucose metabolism. Alzheimer's typically produces "
                                   "temporoparietal and posterior cingulate hypometabolism, often before "
                                   "atrophy is visible on MRI.",
    ("mri", "atrophy", "hippocamp"): "Structural MRI in Alzheimer's typically shows medial temporal "
                                     "(hippocampal/entorhinal) atrophy and ventricular enlargement.",
    ("eeg", "theta", "alpha", "tar"): "EEG slowing (rising theta/delta, falling alpha, theta/alpha ratio "
                                      "above ~1.1) is associated with cognitive decline.",
    ("medic", "drug", "treat", "donepezil", "memantine"): "Approved symptomatic treatments include "
        "cholinesterase inhibitors (donepezil, rivastigmine, galantamine) and memantine. Anti-amyloid "
        "antibodies may be considered for biomarker-confirmed early AD under specialist care.",
    ("diet", "exercise", "lifestyle", "prevent"): "Evidence supports regular aerobic exercise, a "
        "Mediterranean/MIND diet, good sleep, hearing correction, social and cognitive engagement, "
        "and control of hypertension and diabetes.",
    ("uncertain", "confiden", "defer"): "When model uncertainty is high or modalities disagree, the "
        "case should be deferred to a specialist and supplemented with further biomarkers.",
}


class ClinicalMedicalAdvisor:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    def set_api_key(self, api_key):
        self.api_key = (api_key or "").strip()

    # ------------------------------------------------------------------ #
    def ask(self, query, patient_context=None, conversation_history=None):
        if self.api_key:
            try:
                return {"response": self._ask_gemini(query, patient_context, conversation_history or []),
                        "source": "gemini", "disclaimer": DISCLAIMER}
            except Exception as e:
                fallback = self._rule_answer(query, patient_context)
                return {"response": fallback, "source": "clinical_guideline_engine",
                        "warning": f"Gemini unavailable ({e}); used built-in engine.",
                        "disclaimer": DISCLAIMER}
        return {"response": self._rule_answer(query, patient_context),
                "source": "clinical_guideline_engine", "disclaimer": DISCLAIMER}

    def _ask_gemini(self, query, ctx, history):
        system = ("You are a clinical decision-support assistant for Alzheimer's disease staging "
                  "(CN/MCI/AD) using MRI, PET, EEG and clinical scores. Be concise and evidence-based, "
                  "and remind the user that findings require clinician confirmation.")
        parts = [system]
        if ctx:
            parts.append(f"Patient context: {self._context_line(ctx)}")
        for turn in history[-6:]:
            if isinstance(turn, dict):
                parts.append(f"{turn.get('role', 'user')}: {turn.get('content', '')}")
        parts.append(f"user: {query}")
        resp = requests.post(GEMINI_URL, params={"key": self.api_key}, timeout=30,
                             json={"contents": [{"parts": [{"text": "\n".join(parts)}]}]})
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    def _rule_answer(self, query, ctx):
        q = query.lower()
        lines = []
        code = self._code(ctx)
        if code and any(w in q for w in ("result", "patient", "diagnos", "mean", "recommend",
                                          "next", "plan", "explain", "what should")):
            g = STAGE_GUIDANCE[code]
            lines.append(f"**Assessment:** {g['name']} — {g['summary']}")
            lines.append(f"**Patient:** {self._context_line(ctx)}")
            lines.append("**Recommended next steps:**")
            lines += [f"- {r}" for r in g["recommendations"]]
        for keys, text in TOPICS.items():
            if any(k in q for k in keys):
                lines.append(text)
        if not lines:
            if code:
                g = STAGE_GUIDANCE[code]
                lines.append(f"The current case is staged as **{g['name']}**. {g['summary']}")
                lines += [f"- {r}" for r in g["recommendations"]]
            else:
                lines.append("I can explain MMSE/CDR scores, MRI and PET findings, EEG biomarkers, "
                             "treatment options, lifestyle measures, or interpret a prediction — run "
                             "a live prediction first to discuss a specific patient.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def generate_clinical_consult_summary(self, data):
        ctx = data.get("patient_context", data) or {}
        code = self._code(ctx) or "MCI"
        g = STAGE_GUIDANCE[code]
        probs = ctx.get("probabilities", {})
        prob_line = ", ".join(f"{k}: {float(v) * 100:.1f}%" for k, v in probs.items()) or "n/a"
        notes = ctx.get("clinical_notes", [])
        out = [
            "NEURODEGENERATIVE CLINICAL CONSULT SUMMARY",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "PATIENT",
            f"  Age: {ctx.get('age', 'n/a')}   Sex: {ctx.get('gender', 'n/a')}   "
            f"Education: {ctx.get('education', 'n/a')} yrs",
            f"  MMSE: {ctx.get('mmse', 'n/a')}/30   CDR: {ctx.get('cdr', 'n/a')}",
            "",
            "AI STAGING",
            f"  Impression: {g['name']}",
            f"  Confidence: {ctx.get('confidence_percentage', 'n/a')}%",
            f"  Class probabilities: {prob_line}",
            "",
            "FINDINGS",
            f"  {g['summary']}",
        ]
        out += [f"  - {n}" for n in notes]
        out += ["", "RECOMMENDATIONS"] + [f"  - {r}" for r in g["recommendations"]]
        out += ["", f"NOTE: {DISCLAIMER}"]
        return "\n".join(out)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _code(ctx):
        if not ctx:
            return None
        code = str(ctx.get("prediction_code") or ctx.get("prediction") or "").upper()
        for c in ("MCI", "AD", "CN"):
            if c in code:
                return c
        return None

    @staticmethod
    def _context_line(ctx):
        fields = [("prediction", "Prediction"), ("confidence_percentage", "Confidence %"),
                  ("age", "Age"), ("gender", "Sex"), ("mmse", "MMSE"), ("cdr", "CDR")]
        return ", ".join(f"{label}: {ctx[k]}" for k, label in fields if k in ctx)
