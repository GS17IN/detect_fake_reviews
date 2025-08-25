import warnings
warnings.filterwarnings('ignore')

import numpy as np
from typing import List, Tuple, Dict
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
import os

# attempt to import PyABSA (best-case ABSA)
try:
    from pyabsa.functional import APSAnta, ATEPCCheckpointManager, APCCheckpointManager
    PYABSA_OK = True
except Exception:
    PYABSA_OK = False

# ensure resources
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nlp = spacy.load('en_core_web_sm')
except Exception:
    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')

sia = SentimentIntensityAnalyzer()

DEFAULT_ASPECTS = ['price', 'quality', 'delivery', 'packaging', 'service']
ASPECT_KEYWORDS = {
    'price': ['price', 'pricing', 'cost', 'expensive', 'cheap', 'value', 'worth'],
    'quality': ['quality', 'build', 'durable', 'defect', 'broken', 'premium', 'material'],
    'delivery': ['delivery', 'shipping', 'arrived', 'delay', 'late', 'courier', 'speed'],
    'packaging': ['packaging', 'box', 'sealed', 'damaged', 'wrap'],
    'service': ['service', 'support', 'customer', 'refund', 'return', 'replace', 'response']
}


class ABSAExtractor:
    """
    Tries to use PyABSA for aspect extraction + polarity.
    If PyABSA is unavailable or fails, falls back to keyword + VADER sentence scoring.
    Returns (aspect_vector, raw) where aspect_vector is np.array in the order of DEFAULT_ASPECTS
    """
    def __init__(self, aspects: List[str] = None, use_pyabsa: bool = True):
        self.aspects = aspects or DEFAULT_ASPECTS
        self.use_pyabsa = use_pyabsa and PYABSA_OK
        self._pyabsa_apc = None
        self._pyabsa_atepc = None
        if self.use_pyabsa:
            try:
                # load light checkpoints if installed — may require first-time downloads
                # Use APC for sentiment classification and ATEPC for aspect extraction if available
                self._pyabsa_apc = APCCheckpointManager.get_sentiment_classifier(checkpoint='english')  # may be changed
                self._pyabsa_atepc = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english')
            except Exception:
                # If any loading error, disable PyABSA fallback
                self.use_pyabsa = False

    def _fallback_scores(self, text: str) -> Dict[str, float]:
        doc = nlp.make_doc(text)
        sents = [s.text for s in doc.sents] if len(list(doc.sents)) else [text]
        out = {}
        for a in self.aspects:
            kws = ASPECT_KEYWORDS.get(a, [])
            matched = [sia.polarity_scores(s)['compound'] for s in sents if any(kw in s.lower() for kw in kws)]
            out[a] = float(np.mean(matched)) if matched else 0.0
        return out

    def encode(self, text: str) -> Tuple[np.ndarray, dict]:
        # If PyABSA is available and loaded, try to use it
        if self.use_pyabsa and (self._pyabsa_atepc is not None):
            try:
                # PyABSA ATEPC returns structured result — we'll map terms to our aspects by keyword overlap
                ate_res = self._pyabsa_atepc.extract_aspect(text, pred_sentiment=True)
                per_aspect = {a: [] for a in self.aspects}
                # ate_res might be a list/dict depending on version; try to robustly parse
                if isinstance(ate_res, dict):
                    items = [ate_res]
                else:
                    items = list(ate_res)
                for item in items:
                    # find fields that look like aspects/sentiments
                    asp_terms = item.get('aspect', []) if isinstance(item, dict) else []
                    pols = item.get('sentiment', []) if isinstance(item, dict) else []
                    for t, p in zip(asp_terms, pols):
                        t_low = str(t).lower()
                        score = {'negative': -1.0, 'neutral': 0.0, 'positive': 1.0}.get(str(p).lower(), 0.0)
                        for a in self.aspects:
                            if a in t_low or any(k in t_low for k in ASPECT_KEYWORDS.get(a, [])):
                                per_aspect[a].append(score)
                vec = [float(np.mean(per_aspect[a])) if per_aspect[a] else 0.0 for a in self.aspects]
                return np.array(vec, dtype=np.float32), {'raw': ate_res}
            except Exception:
                # fallback below
                pass
        # fallback: keyword+VADER
        scores = self._fallback_scores(text)
        vec = np.array([scores[a] for a in self.aspects], dtype=np.float32)
        return vec, {'raw': None}
