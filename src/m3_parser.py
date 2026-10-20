"""M3: dual-syntax analysis over an M1 token stream.

Two independent syntax views of the same sentence:

(a) A recursive-descent parser for a small English CFG over coarse POS tags.
    Yields a parse tree per sentence plus depth / branching / success metrics.

        Sentence   -> Adjunct* Clause (Joiner Adjunct* Clause)* PUNCT*
        Joiner     -> PUNCT CONJ | PUNCT | CONJ
        Adjunct    -> (SubClause | PP | ADV | ADJ | NUM | PART | INTJ) PUNCT?
        SubClause  -> (PREP | ADV) Clause
        Clause     -> NP PUNCT? VP | VP
        NP         -> NPBase (PUNCT? CONJ NPBase)*
        NPBase     -> DET? Premod* Head Postmod*
        Premod     -> ADJ | NUM | ADV (ADJ | ADV)
        Head       -> PRON | NOUN+ | ADJ | NUM
        Postmod    -> PP | PUNCT? Relative
        Relative   -> (DET | PRON) VP
        VP         -> ADV* (MODAL ADV*)* VERB+ ADV* Comp* (PUNCT? CONJ VP)?
        Comp       -> NP | PP | PART VP | CompClause | ADJ | ADV | PART | NUM
        CompClause -> (DET | PREP) Clause
        PP         -> PREP NP

    PUNCT is a real terminal, admitted only at phrase and clause boundaries.
    That is what stops a VP's complement list from reading across a comma,
    so "Although the results improve, we note X" splits where it should.
    Only END is stripped before parsing.

    Known limitation: a trailing participial adjunct ("... using a dynamic
    graph cuts method") is not attached, since VBG and VBN both coarsen to
    VERB. Such sentences come back as partial parses.

(b) A DFA over the coarse POS-tag sequence that recognises boilerplate
    sentence templates ("In conclusion, ...", "it is important to note",
    "one of the most ..."). Deterministic prefix automaton: explicit
    transition table, longest match wins.

Public entry points:
    parse_text(text)     -> dict   (tokenizes with M1, then parses)
    parse_tokens(tokens) -> dict   (reuse an existing M1 token stream)

POS tagging uses nltk when its perceptron tagger data is available, and falls
back to a deterministic closed-class + suffix tagger otherwise, so a fresh
clone with no nltk_data still runs. The mode used is reported as meta["tagger"].
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m1_lexer import tokenize  # noqa: E402

# --------------------------------------------------------------------------
# Coarse tag set
# --------------------------------------------------------------------------

# Penn Treebank -> coarse tag. Alphabetic tags not listed here become X.
PENN_TO_COARSE = {
    "NN": "NOUN", "NNS": "NOUN", "NNP": "NOUN", "NNPS": "NOUN",
    "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB",
    "VBP": "VERB", "VBZ": "VERB",
    "MD": "MODAL",
    "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
    "RB": "ADV", "RBR": "ADV", "RBS": "ADV", "WRB": "ADV",
    # PRP$/WP$ are possessives; they fill the determiner slot in an NP.
    "DT": "DET", "PDT": "DET", "WDT": "DET", "PRP$": "DET", "WP$": "DET",
    "PRP": "PRON", "WP": "PRON", "EX": "PRON",
    "IN": "PREP",
    "CC": "CONJ",
    "TO": "PART", "RP": "PART", "POS": "PART",
    "CD": "NUM",
    "UH": "INTJ",
    "FW": "X", "SYM": "X", "LS": "X",
}

# Coarse tags the CFG parser never sees. PUNCT is kept: the grammar uses it
# as a boundary terminal. Only the sentence terminator is dropped.
NON_SYNTACTIC = {"END"}

# Coarse tags that can appear, fixed so M4 can build a stable feature vector.
COARSE_TAGS = (
    "NOUN", "VERB", "MODAL", "ADJ", "ADV", "DET", "PRON", "PREP",
    "CONJ", "PART", "NUM", "INTJ", "X", "PUNCT", "END",
)

STRUCTURE_CLASSES = ("SIMPLE", "COMPOUND", "COMPLEX", "FRAGMENT")


def coarse_tag(penn: str) -> str:
    """Map a Penn Treebank tag to the coarse tag set used by M3."""
    if penn in PENN_TO_COARSE:
        return PENN_TO_COARSE[penn]
    # nltk tags punctuation as itself (".", ",", ":", "``", "''", ...).
    if not penn or not penn[0].isalpha():
        return "PUNCT"
    return "X"


# --------------------------------------------------------------------------
# POS tagging: nltk when available, deterministic fallback otherwise
# --------------------------------------------------------------------------

_TAGGER_MODE: str | None = None

# Closed-class words for the fallback tagger, keyed by the Penn tag to emit.
_FALLBACK_LEXICON = {
    "DT": {"the", "a", "an", "this", "that", "these", "those", "each", "every",
           "no", "any", "some", "all", "both", "either", "neither", "another",
           "my", "your", "his", "her", "its", "our", "their", "which",
           "whose"},
    "PRP": {"i", "you", "he", "she", "it", "we", "they", "me", "him", "them",
            "us", "myself", "itself", "themselves", "there", "who", "what"},
    "IN": {"of", "in", "on", "at", "by", "for", "with", "from", "into",
           "about", "over", "under", "between", "through", "during", "among",
           "against", "without", "within", "upon", "across", "toward",
           "towards", "after", "before", "because", "although", "though",
           "while", "if", "since", "unless", "whereas", "than", "as", "like",
           "per", "via"},
    "CC": {"and", "or", "but", "nor", "yet", "so", "plus"},
    "MD": {"can", "could", "may", "might", "must", "shall", "should", "will",
           "would", "ought"},
    "VBZ": {"is", "has", "does"},
    "VBP": {"are", "am", "have", "do"},
    "VBD": {"was", "were", "had", "did"},
    "VB": {"be", "been", "being"},
    "TO": {"to"},
    "RB": {"not", "very", "also", "however", "therefore", "moreover", "thus",
           "furthermore", "additionally", "often", "always", "never",
           "really", "quite", "rather", "more", "most", "less", "least",
           "too", "only", "just", "still", "already", "finally", "indeed",
           "especially", "particularly", "significantly", "ultimately",
           "consequently", "similarly", "likewise", "overall", "generally",
           "typically"},
    "JJ": {"many", "much", "such", "own", "other", "same", "various",
           "several", "important", "key", "main", "major", "significant",
           "crucial"},
}

# word -> Penn tag, flattened once at import.
_FALLBACK_WORDS = {
    word: tag for tag, words in _FALLBACK_LEXICON.items() for word in words
}

# Suffix rules for open-class words, longest suffix first.
_FALLBACK_SUFFIXES = (
    ("ousness", "NN"), ("fulness", "NN"), ("ability", "NN"), ("ibility", "NN"),
    ("ization", "NN"), ("isation", "NN"), ("ically", "RB"), ("ement", "NN"),
    ("ation", "NN"), ("ition", "NN"), ("ments", "NNS"), ("ships", "NNS"),
    ("ities", "NNS"), ("ively", "RB"), ("ingly", "RB"), ("ness", "NN"),
    ("ment", "NN"), ("tion", "NN"), ("sion", "NN"), ("ance", "NN"),
    ("ence", "NN"), ("ship", "NN"), ("hood", "NN"), ("ical", "JJ"),
    ("able", "JJ"), ("ible", "JJ"), ("less", "JJ"), ("ious", "JJ"),
    ("eous", "JJ"), ("ting", "VBG"), ("ming", "VBG"), ("ning", "VBG"),
    ("ized", "VBN"), ("ised", "VBN"), ("ally", "RB"), ("ing", "VBG"),
    ("ity", "NN"), ("ist", "NN"), ("ism", "NN"), ("ous", "JJ"),
    ("ive", "JJ"), ("ful", "JJ"), ("ant", "JJ"), ("ent", "JJ"),
    ("ary", "JJ"), ("est", "JJS"), ("ier", "JJR"), ("ers", "NNS"),
    ("ies", "NNS"), ("ate", "VB"), ("ify", "VB"), ("ize", "VB"),
    ("ise", "VB"), ("ly", "RB"), ("ed", "VBD"), ("al", "JJ"),
    ("ic", "JJ"), ("er", "NN"), ("or", "NN"), ("es", "NNS"),
)


def _fallback_tag_word(word: str) -> str:
    """Penn tag for one word: closed-class lookup, then suffix rules."""
    if word[:1].isdigit():
        return "CD"
    lowered = word.lower()
    if lowered in _FALLBACK_WORDS:
        return _FALLBACK_WORDS[lowered]
    for suffix, tag in _FALLBACK_SUFFIXES:
        if lowered.endswith(suffix) and len(lowered) > len(suffix) + 1:
            return tag
    if lowered.endswith("s") and len(lowered) > 3:
        return "NNS"
    # Capitalisation is a usable proper-noun cue.
    if word[:1].isupper():
        return "NNP"
    return "NN"


def _fallback_pos_tag(words: list[str]) -> list[str]:
    """Tag a word list without nltk. Deterministic and offline."""
    return [_fallback_tag_word(word) for word in words]


def tagger_mode() -> str:
    """Return "nltk" or "fallback", probing (and warming) nltk once."""
    global _TAGGER_MODE
    if _TAGGER_MODE is not None:
        return _TAGGER_MODE
    try:
        import nltk

        try:
            nltk.pos_tag(["probe"])
        except LookupError:
            # nltk >= 3.9 renamed the tagger package; try both names quietly.
            for package in ("averaged_perceptron_tagger_eng",
                            "averaged_perceptron_tagger"):
                try:
                    nltk.download(package, quiet=True)
                except Exception:  # noqa: BLE001 - offline is an expected path
                    pass
            nltk.pos_tag(["probe"])
        _TAGGER_MODE = "nltk"
    except Exception:  # noqa: BLE001 - any nltk failure means use the fallback
        _TAGGER_MODE = "fallback"
    return _TAGGER_MODE


def pos_tag_words(words: list[str]) -> list[str]:
    """Penn tags for a pre-tokenized word list (M1 WORD/NUMBER values)."""
    if not words:
        return []
    if tagger_mode() == "nltk":
        import nltk

        return [tag for _word, tag in nltk.pos_tag(words)]
    return _fallback_pos_tag(words)


# --------------------------------------------------------------------------
# Sentence segmentation over the M1 token stream
# --------------------------------------------------------------------------

def split_sentences(tokens: list[dict]) -> list[list[dict]]:
    """Group an M1 token stream into sentences, splitting after SENTENCE_END.

    The terminator stays with its sentence. Groups holding no WORD or NUMBER
    token (e.g. the tail of a "..." run) are dropped.
    """
    sentences: list[list[dict]] = []
    current: list[dict] = []
    for token in tokens:
        current.append(token)
        if token["type"] == "SENTENCE_END":
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return [
        sentence for sentence in sentences
        if any(token["type"] in ("WORD", "NUMBER") for token in sentence)
    ]


def tag_sentence(sentence: list[dict]) -> list[dict]:
    """Attach Penn and coarse tags to every token in one sentence.

    Only WORD/NUMBER tokens go to the tagger; PUNCT and SENTENCE_END get
    coarse tags directly, so punctuation stays visible to the DFA.
    """
    word_positions = [
        i for i, token in enumerate(sentence)
        if token["type"] in ("WORD", "NUMBER")
    ]
    penn_tags = pos_tag_words([sentence[i]["value"] for i in word_positions])
    penn_by_position = dict(zip(word_positions, penn_tags))

    tagged: list[dict] = []
    for i, token in enumerate(sentence):
        if i in penn_by_position:
            penn = penn_by_position[i]
            coarse = coarse_tag(penn)
        elif token["type"] == "SENTENCE_END":
            penn, coarse = token["value"], "END"
        else:
            penn, coarse = token["value"], "PUNCT"
        tagged.append({
            "value": token["value"],
            "idx": token["idx"],
            "penn": penn,
            "coarse": coarse,
        })
    return tagged


# --------------------------------------------------------------------------
# (a) Recursive-descent CFG parser
# --------------------------------------------------------------------------

def _branch(label: str, children: list[dict]) -> dict:
    """Internal (non-leaf) parse-tree node."""
    return {"label": label, "children": children}


# Clause nesting cap, so a pathological sentence can never blow the Python
# stack. Each level consumes >= 1 tag, and real prose nests far shallower.
MAX_CLAUSE_DEPTH = 40

# Branch labels that mean the sentence contains a subordinate clause.
SUBORDINATE_LABELS = ("SubClause", "Relative", "CompClause")

# Penn tags that may continue an auxiliary chain: base forms and participles.
# A finite verb (VBZ/VBP/VBD) starts a new predicate instead, which is what
# keeps a relative clause off the main verb in "the method that works is fast".
AUX_CHAIN_PENN = frozenset({"VB", "VBN", "VBG"})

# Participles. Both coarsen to VERB, but they also act as adjectives
# ("the proposed method") and head reduced relatives ("images using a graph"),
# which is why the Penn tag is kept alongside the coarse one.
PARTICIPLE_PENN = frozenset({"VBN", "VBG"})

# M1 splits "multi-scale" into WORD PUNCT WORD, so the parser rejoins the
# pieces into one compound head rather than halting on the hyphen.
HYPHENS = frozenset({"-", "‐", "‑", "–", "—", "/"})


class _Parser:
    """Predictive recursive-descent parser over a coarse tag sequence.

    Every production that can fail saves and restores self.pos, so a failed
    sub-parse never leaves the cursor stranded mid-rule. eat() only advances
    on a match, which is what guarantees the starred loops terminate.
    """

    def __init__(self, tags: list[str], values: list[str],
                 penns: list[str] | None = None) -> None:
        self.tags = tags
        self.values = values
        # Penn tags are used only to gate auxiliary chaining. Absent, no verb
        # chains, which is the conservative reading.
        self.penns = penns if penns is not None else [""] * len(tags)
        self.pos = 0
        self.depth = 0

    def peek(self, offset: int = 0) -> str | None:
        index = self.pos + offset
        return self.tags[index] if index < len(self.tags) else None

    def peek_penn(self, offset: int = 0) -> str | None:
        index = self.pos + offset
        return self.penns[index] if index < len(self.penns) else None

    def eat(self, tag: str) -> dict | None:
        """Consume one terminal of the given tag, returning a leaf node."""
        if self.peek() != tag:
            return None
        leaf = {"label": tag, "value": self.values[self.pos]}
        self.pos += 1
        return leaf

    # Sentence -> PUNCT* Adjunct* Clause (Joiner Adjunct* Clause)* PUNCT*
    def sentence(self) -> dict | None:
        children: list[dict] = []
        # Leading markup: RAID rows carry "> " quotes and "* " bullets.
        while True:
            opener = self.eat("PUNCT")
            if opener is None:
                break
            children.append(opener)
        children.extend(self.adjuncts())
        first = self.clause()
        if first is None:
            # Leading adjuncts with no main clause are not a sentence. The
            # cursor stays put so coverage still reflects the partial parse.
            return None
        children.append(first)
        while True:
            save = self.pos
            joiner: list[dict] = []
            comma = self.eat("PUNCT")
            if comma:
                joiner.append(comma)
            conj = self.eat("CONJ")
            if conj:
                joiner.append(conj)
            if not joiner:
                break
            trailing = self.adjuncts()
            following = self.clause()
            if following is None:
                self.pos = save
                break
            children.extend(joiner)
            children.extend(trailing)
            children.append(following)
        # Trailing punctuation: a closing quote or bracket before the period.
        while True:
            closer = self.eat("PUNCT")
            if closer is None:
                break
            children.append(closer)
        return _branch("Sentence", children)

    def adjuncts(self) -> list[dict]:
        """Zero or more adjuncts, each with an optional boundary comma."""
        found: list[dict] = []
        while True:
            node = self.adjunct()
            if node is None:
                return found
            found.append(node)
            comma = self.eat("PUNCT")
            if comma:
                found.append(comma)

    # Adjunct -> SubClause | PP | ADV | ADJ | NUM | PART | INTJ
    def adjunct(self) -> dict | None:
        # SubClause is strictly more demanding than PP, so try it first:
        # "Although the results improve, we ..." must not stop at the PP.
        subordinate = self.sub_clause()
        if subordinate:
            return subordinate
        attachment = self.pp()
        if attachment:
            return attachment
        # "Motivated by a recent result, we show ..." / "Using X, we show ..."
        reduced = self.participial()
        if reduced:
            return reduced
        for tag in ("ADV", "ADJ", "NUM", "PART", "INTJ"):
            leaf = self.eat(tag)
            if leaf:
                return _branch("Adjunct", [leaf])
        return None

    # SubClause -> (PREP | ADV) Clause
    def sub_clause(self) -> dict | None:
        # DET is deliberately not a subordinator: a sentence-initial "The" is
        # the subject's determiner, and allowing it here let an entire main
        # clause be read as a subordinate one, leaving the sentence unparsed.
        marker_tag = self.peek()
        if marker_tag not in ("PREP", "ADV"):
            return None
        save = self.pos
        marker = self.eat(marker_tag)
        inner = self.clause()
        if inner is None:
            self.pos = save
            return None
        return _branch("SubClause", [marker, inner])

    # Clause -> NP PUNCT? VP | VP   (the NP-less form covers imperatives)
    def clause(self) -> dict | None:
        if self.depth >= MAX_CLAUSE_DEPTH:
            return None
        self.depth += 1
        try:
            save = self.pos
            children: list[dict] = []
            subject = self.np()
            if subject:
                children.append(subject)
                comma = self.eat("PUNCT")  # "The method, however, works"
                if comma:
                    children.append(comma)
            predicate = self.vp()
            if predicate is None:
                self.pos = save
                return None
            children.append(predicate)
            return _branch("Clause", children)
        finally:
            self.depth -= 1

    # NP -> NPBase (PUNCT? CONJ NPBase)*
    def np(self) -> dict | None:
        first = self.np_base()
        if first is None:
            return None
        parts = [first]
        while True:
            save = self.pos
            comma = self.eat("PUNCT")
            conj = self.eat("CONJ")
            # A CONJ is required: a bare comma would let coordination run
            # across a clause boundary ("we propose X, we also show Y").
            if conj is None:
                self.pos = save
                break
            following = self.np_base()
            if following is None:
                self.pos = save
                break
            if comma:
                parts.append(comma)
            parts.append(conj)
            parts.append(following)
        if len(parts) == 1:
            return first
        return _branch("NPCoord", parts)

    # NPBase -> DET? Premod* Head Postmod*
    def np_base(self) -> dict | None:
        save = self.pos
        children: list[dict] = []
        determiner = self.eat("DET")
        if determiner:
            children.append(determiner)

        premods: list[dict] = []
        while True:
            tag = self.peek()
            if tag in ("ADJ", "NUM"):
                premods.append(self.eat(tag))
            elif tag == "ADV" and self.peek(1) in ("ADJ", "ADV"):
                # Only an ADV that premodifies: "most useful", "very highly".
                premods.append(self.eat("ADV"))
            elif (tag == "VERB"
                  and self.peek_penn() in PARTICIPLE_PENN
                  and self.peek(1) in ("NOUN", "ADJ", "VERB", "NUM")):
                # Participial adjective: "the proposed method", "a growing
                # demand". Requires a following NP element so the main verb
                # is never mistaken for a modifier.
                premods.append(self.eat("VERB"))
            else:
                break

        head: list[dict] = []
        if self.peek() == "PRON":
            head.append(self.eat("PRON"))
        elif self.peek() == "NOUN":
            while self.peek() == "NOUN":  # compounds: "graph feature set"
                head.append(self.eat("NOUN"))
                if (self.peek() == "PUNCT"
                        and self.values[self.pos] in HYPHENS
                        and self.peek(1) == "NOUN"):
                    head.append(self.eat("PUNCT"))  # "multi - scale"
        elif premods and premods[-1]["label"] in ("ADJ", "NUM"):
            # Nominalised modifier as head: "the poor", "the first".
            head.append(premods.pop())
        else:
            self.pos = save
            return None

        children.extend(premods)
        children.extend(head)
        while True:
            postmod = self.postmod()
            if postmod is None:
                break
            children.append(postmod)
        return _branch("NP", children)

    # Postmod -> PP | Participial | PUNCT? Relative
    def postmod(self) -> dict | None:
        attachment = self.pp()
        if attachment:
            return attachment
        reduced = self.participial()
        if reduced:
            return reduced
        save = self.pos
        comma = self.eat("PUNCT")  # "the method, which is fast, ..."
        # A comma-introduced relative must be marked by which/that (DET).
        # WP ("who") and PRP ("we") both coarsen to PRON, so after a comma
        # the only safe reading of PRON is a new clause: in "... the cost,
        # we note X" the comma is a clause boundary, not a relative.
        relative = self.relative(allow_pron=comma is None)
        if relative is None:
            self.pos = save
            return None
        return _branch("Postmod", [comma, relative]) if comma else relative

    # Relative -> (DET | PRON) VP   ("that enables", "which is")
    def relative(self, allow_pron: bool = True) -> dict | None:
        markers = ("DET", "PRON") if allow_pron else ("DET",)
        marker_tag = self.peek()
        if marker_tag not in markers:
            return None
        save = self.pos
        marker = self.eat(marker_tag)
        predicate = self.vp()
        if predicate is None:
            self.pos = save
            return None
        return _branch("Relative", [marker, predicate])

    # PP -> PREP NP
    def pp(self) -> dict | None:
        if self.peek() != "PREP":
            return None
        save = self.pos
        prep = self.eat("PREP")
        obj = self.np()
        if obj is None:
            self.pos = save
            return None
        return _branch("PP", [prep, obj])

    # VP -> ADV* (MODAL ADV*)* VERB+ ADV* Comp* (CONJ VP)?
    def vp(self) -> dict | None:
        save = self.pos
        children: list[dict] = []
        while self.peek() == "ADV":
            children.append(self.eat("ADV"))
        while self.peek() == "MODAL":
            children.append(self.eat("MODAL"))
            while self.peek() == "ADV":
                children.append(self.eat("ADV"))
        verb = self.eat("VERB")
        if verb is None:
            self.pos = save
            return None
        children.append(verb)
        self.predicate_tail(children)
        return _branch("VP", children)

    # Participial -> VERB(VBG | VBN) <predicate tail>
    def participial(self) -> dict | None:
        """A reduced relative clause: "images using a graph cuts method".

        Only a participle may head one; a finite verb there is the main
        predicate, so letting it through would strip the clause of its verb.
        """
        if self.peek() != "VERB" or self.peek_penn() not in PARTICIPLE_PENN:
            return None
        children = [self.eat("VERB")]
        self.predicate_tail(children)
        return _branch("Participial", children)

    def predicate_tail(self, children: list[dict]) -> None:
        """Verb group, adverbs, complements and coordination after a verb.

        Shared by vp() and participial(); appends to children in place.
        """
        # Adverbs and non-finite verbs continue the chain, so "has been
        # shown" and "did not expect" stay one predicate, "works is" does not.
        while True:
            mark = self.pos
            adverbs: list[dict] = []
            while self.peek() == "ADV":
                adverbs.append(self.eat("ADV"))
            if self.peek() == "VERB" and self.peek_penn() in AUX_CHAIN_PENN:
                children.extend(adverbs)
                children.append(self.eat("VERB"))
                continue
            self.pos = mark
            break
        while self.peek() == "ADV":
            children.append(self.eat("ADV"))
        while True:
            complement = self.complement()
            if complement is None:
                break
            children.append(complement)
        # Coordinated predicate: "we train and evaluate the model".
        save_conj = self.pos
        comma = self.eat("PUNCT")
        conj = self.eat("CONJ")
        following = self.vp() if conj else None
        if following is None:
            self.pos = save_conj
        else:
            if comma:
                children.append(comma)
            children.append(conj)
            children.append(following)

    # Comp -> NP | PP | PART VP | CompClause | ADJ | ADV | PART | NUM
    def complement(self) -> dict | None:
        # NP and PP come first so "that method" and "of the graph" are read as
        # phrases, leaving only a genuine clause for CompClause below.
        obj = self.np()
        if obj:
            return _branch("Comp", [obj])
        attachment = self.pp()
        if attachment:
            return _branch("Comp", [attachment])
        if self.peek() == "PART":  # infinitives: "to note", "to be shown"
            save = self.pos
            part = self.eat("PART")
            inner = self.vp()
            if inner is None:
                self.pos = save
            else:
                return _branch("Comp", [part, inner])
        clause = self.comp_clause()
        if clause:
            return clause
        for tag in ("ADJ", "ADV", "PART", "NUM"):
            leaf = self.eat(tag)
            if leaf:
                return _branch("Comp", [leaf])
        return None

    # CompClause -> (DET | PREP) Clause   ("shows that the method works")
    def comp_clause(self) -> dict | None:
        marker_tag = self.peek()
        if marker_tag not in ("DET", "PREP"):
            return None
        save = self.pos
        marker = self.eat(marker_tag)
        inner = self.clause()
        if inner is None:
            self.pos = save
            return None
        return _branch("CompClause", [marker, inner])


def parse_tags(tags: list[str], values: list[str],
               penns: list[str] | None = None) -> tuple[dict | None, int]:
    """Parse a coarse tag sequence. Returns (tree_or_None, tags_consumed).

    penns are the parallel Penn tags, used only to gate auxiliary chaining.
    """
    parser = _Parser(tags, values, penns)
    tree = parser.sentence()
    return tree, parser.pos


def tree_depth(node: dict) -> int:
    """Height of a parse tree; a leaf has depth 1."""
    children = node.get("children")
    if not children:
        return 1
    return 1 + max(tree_depth(child) for child in children)


def tree_nodes(node: dict) -> int:
    """Total node count, leaves included."""
    children = node.get("children")
    if not children:
        return 1
    return 1 + sum(tree_nodes(child) for child in children)


def tree_branching(node: dict) -> float:
    """Mean child count over internal nodes; 0.0 for a bare leaf."""
    internal = 0
    child_total = 0
    stack = [node]
    while stack:
        current = stack.pop()
        children = current.get("children")
        if not children:
            continue
        internal += 1
        child_total += len(children)
        stack.extend(children)
    return child_total / internal if internal else 0.0


def count_labels(node: dict, label: str) -> int:
    """Occurrences of a non-terminal label in a parse tree."""
    children = node.get("children")
    if not children:
        return 0
    own = 1 if node.get("label") == label else 0
    return own + sum(count_labels(child, label) for child in children)


# --------------------------------------------------------------------------
# (b) Template DFA over coarse POS-tag sequences
# --------------------------------------------------------------------------

# Patterns anchored at the start of a sentence.
START_TEMPLATES = (
    ("discourse_opener", ("ADV", "PUNCT")),            # "Furthermore,"
    ("discourse_opener", ("ADJ", "PUNCT")),            # "Overall,"
    ("prep_phrase_opener", ("PREP", "NOUN", "PUNCT")),           # "In conclusion,"
    ("prep_phrase_opener", ("PREP", "DET", "NOUN", "PUNCT")),    # "In the end,"
    ("prep_phrase_opener", ("PREP", "DET", "ADJ", "NOUN", "PUNCT")),
    ("existential_opener", ("PRON", "VERB", "ADJ", "NOUN")),  # "There are many X"
    ("existential_opener", ("PRON", "VERB", "DET", "NOUN")),  # "There is a X"
    ("topic_sentence", ("DET", "ADJ", "NOUN", "VERB")),       # "The main benefit is"
    ("topic_sentence", ("DET", "NOUN", "PREP", "DET", "NOUN", "VERB")),
)

# Patterns matched at any offset inside a sentence.
ANY_TEMPLATES = (
    ("hedged_assertion", ("PRON", "VERB", "ADJ", "PART", "VERB")),
    ("hedged_assertion", ("MODAL", "VERB", "VERB")),            # "can be seen"
    ("superlative_frame", ("NUM", "PREP", "DET", "ADV", "ADJ")),
    ("superlative_frame", ("DET", "ADV", "ADJ", "NOUN")),
    ("enumeration", ("NOUN", "PUNCT", "NOUN", "PUNCT", "CONJ")),
    ("nominalization_chain",
     ("NOUN", "PREP", "DET", "NOUN", "PREP", "DET", "NOUN")),
)

TEMPLATE_NAMES = tuple(sorted(
    {name for name, _ in START_TEMPLATES} | {name for name, _ in ANY_TEMPLATES}
))


def build_dfa(patterns) -> dict:
    """Compile tag patterns into a deterministic prefix automaton.

    Shared prefixes collapse into shared states, so one pass from a given
    offset tests every pattern at once.

    Returns {"start": 0, "transitions": {state: {tag: state}},
             "accepting": {state: template_name}}.
    """
    transitions: dict[int, dict[str, int]] = {0: {}}
    accepting: dict[int, str] = {}
    next_state = 1
    for name, pattern in patterns:
        state = 0
        for tag in pattern:
            row = transitions[state]
            if tag not in row:
                row[tag] = next_state
                transitions[next_state] = {}
                next_state += 1
            state = row[tag]
        accepting.setdefault(state, name)
    return {"start": 0, "transitions": transitions, "accepting": accepting}


_START_DFA = build_dfa(START_TEMPLATES)
_ANY_DFA = build_dfa(ANY_TEMPLATES)


def run_dfa(dfa: dict, tags: list[str], offset: int = 0) -> str | None:
    """Run the DFA from offset; return the longest accepted template name."""
    state = dfa["start"]
    accepted: str | None = None
    for tag in tags[offset:]:
        state = dfa["transitions"].get(state, {}).get(tag)
        if state is None:
            break
        name = dfa["accepting"].get(state)
        if name is not None:
            accepted = name
    return accepted


def match_templates(tags: list[str]) -> list[str]:
    """Template names hit in one sentence's coarse tag sequence (deduped)."""
    hits: list[str] = []
    anchored = run_dfa(_START_DFA, tags, 0)
    if anchored is not None:
        hits.append(anchored)
    for offset in range(len(tags)):
        hit = run_dfa(_ANY_DFA, tags, offset)
        if hit is not None:
            hits.append(hit)
    return list(dict.fromkeys(hits))


def count_subordinate(tree: dict | None) -> int:
    """Subordinate-clause nodes (SubClause + Relative + CompClause)."""
    if tree is None:
        return 0
    return sum(count_labels(tree, label) for label in SUBORDINATE_LABELS)


def structure_class(tree: dict | None, parse_ok: bool) -> str:
    """Coarse structural class for one sentence, read off its parse tree.

    Precedence: a failed parse is FRAGMENT; two or more coordinated main
    clauses is COMPOUND; any subordination is COMPLEX; otherwise SIMPLE.
    A bare PP is a phrase, not a clause, so it leaves a sentence SIMPLE.
    """
    if tree is None or not parse_ok:
        return "FRAGMENT"
    main_clauses = sum(
        1 for child in tree.get("children", ())
        if child.get("label") == "Clause"
    )
    if main_clauses > 1:
        return "COMPOUND"
    if count_subordinate(tree) > 0:
        return "COMPLEX"
    return "SIMPLE"


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def parse_sentence(sentence: list[dict]) -> dict:
    """Run both syntax views over one M1 sentence (list of tokens)."""
    tagged = tag_sentence(sentence)
    tags = [token["coarse"] for token in tagged]
    # The parser sees everything but the sentence terminator; PUNCT is a
    # boundary terminal in the grammar.
    parser_input = [t for t in tagged if t["coarse"] not in NON_SYNTACTIC]
    tree, consumed = parse_tags(
        [token["coarse"] for token in parser_input],
        [token["value"] for token in parser_input],
        [token["penn"] for token in parser_input],
    )
    parse_ok = tree is not None and consumed == len(parser_input)
    n_words = sum(1 for t in tagged if t["coarse"] not in ("PUNCT", "END"))
    return {
        "tagged": tagged,
        "tags": tags,
        "n_tokens": len(tagged),
        "n_words": n_words,
        "tree": tree,
        "parse_ok": parse_ok,
        "consumed": consumed,
        "coverage": consumed / len(parser_input) if parser_input else 0.0,
        "depth": tree_depth(tree) if tree else 0,
        "nodes": tree_nodes(tree) if tree else 0,
        "branching": tree_branching(tree) if tree else 0.0,
        "clauses": count_labels(tree, "Clause") if tree else 0,
        "pps": count_labels(tree, "PP") if tree else 0,
        "subordinate": count_subordinate(tree),
        "templates": match_templates(tags),
        "structure": structure_class(tree, parse_ok),
    }


def parse_tokens(tokens: list[dict]) -> dict:
    """Parse metadata for a whole text, given its M1 token stream.

    Keys:
        n_sentences          sentences holding at least one word
        sentences            per-sentence records (see parse_sentence)
        parse_success_rate   fraction fully parsed by the CFG
        mean_coverage        mean fraction of words consumed per sentence
        mean_depth/max_depth parse-tree height stats
        mean_branching       mean children per internal node
        mean_nodes           mean parse-tree size
        mean_clauses         mean Clause nodes per sentence
        mean_pps             mean PP nodes per sentence
        mean_subordinate     mean subordinate-clause nodes per sentence
        template_hits        sentences matching >= 1 template
        template_score       template_hits / n_sentences
        template_counts      template name -> sentences hit (all names present)
        structure_counts     structural class -> sentence count (all present)
        pos_counts           coarse tag -> token count over the text
        tagger               "nltk" or "fallback"
    """
    sentences = [parse_sentence(s) for s in split_sentences(tokens)]
    total = len(sentences)

    template_counts = {name: 0 for name in TEMPLATE_NAMES}
    structure_counts = {name: 0 for name in STRUCTURE_CLASSES}
    pos_counts = {tag: 0 for tag in COARSE_TAGS}
    hits = 0
    for record in sentences:
        for name in record["templates"]:
            template_counts[name] += 1
        if record["templates"]:
            hits += 1
        structure_counts[record["structure"]] += 1
        for tag in record["tags"]:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1

    def mean(key: str) -> float:
        return sum(record[key] for record in sentences) / total if total else 0.0

    return {
        "n_sentences": total,
        "sentences": sentences,
        "parse_success_rate": (
            sum(1 for r in sentences if r["parse_ok"]) / total if total else 0.0
        ),
        "mean_coverage": mean("coverage"),
        "mean_depth": mean("depth"),
        "max_depth": max((r["depth"] for r in sentences), default=0),
        "mean_branching": mean("branching"),
        "mean_nodes": mean("nodes"),
        "mean_clauses": mean("clauses"),
        "mean_pps": mean("pps"),
        "mean_subordinate": mean("subordinate"),
        "template_hits": hits,
        "template_score": hits / total if total else 0.0,
        "template_counts": template_counts,
        "structure_counts": structure_counts,
        "pos_counts": pos_counts,
        "tagger": tagger_mode(),
    }


def parse_text(text: str) -> dict:
    """Tokenize with M1, then return the parse metadata for the text."""
    return parse_tokens(tokenize(text))


def render_tree(node: dict, indent: int = 0) -> str:
    """Indented rendering of a parse tree, for the demo and debugging."""
    pad = "  " * indent
    children = node.get("children")
    if not children:
        return f"{pad}{node['label']} {node['value']!r}"
    lines = [f"{pad}{node['label']}"]
    lines.extend(render_tree(child, indent + 1) for child in children)
    return "\n".join(lines)


def main() -> None:
    """Pipe one training row through M1 then M3 and print the parse summary."""
    import csv

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(repo_root, "data", "train_clean.csv")
    with open(csv_path, encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    meta = parse_text(row["text"])
    hit_counts = {k: v for k, v in meta["template_counts"].items() if v}
    seen_pos = {k: v for k, v in meta["pos_counts"].items() if v}
    print(f"id={row['id']} label={row['label']} domain={row['domain']}")
    print(f"tagger={meta['tagger']} sentences={meta['n_sentences']}")
    print(f"parse_success_rate={meta['parse_success_rate']:.3f} "
          f"mean_coverage={meta['mean_coverage']:.3f}")
    print(f"mean_depth={meta['mean_depth']:.2f} max_depth={meta['max_depth']} "
          f"mean_branching={meta['mean_branching']:.2f} "
          f"mean_nodes={meta['mean_nodes']:.1f}")
    print(f"template_score={meta['template_score']:.3f} "
          f"hits={meta['template_hits']} {hit_counts}")
    print(f"structure_counts={meta['structure_counts']}")
    print(f"pos_counts={seen_pos}")

    first = meta["sentences"][0]
    print(f"\nfirst sentence ({first['n_words']} words, "
          f"parse_ok={first['parse_ok']}, structure={first['structure']}, "
          f"depth={first['depth']}):")
    print("  tags:", " ".join(first["tags"]))
    if first["tree"]:
        print(render_tree(first["tree"], 1))


if __name__ == "__main__":
    main()
