"""
speciteller_wrapper.py
A Jupyter-friendly, class-based wrapper around SPECITELLER
(Li & Nenkova, AAAI 2015; https://github.com/jjessyli/speciteller).

Why a class? The original speciteller.py does all heavy loading
(~190 MB word-embedding file, Brown clusters, two liblinear models)
at MODULE level, so every import/run reloads everything. This class
loads the resources exactly once in __init__ and exposes a cheap
predict() method you can call repeatedly inside a notebook.

Prerequisites
-------------
1. git clone https://github.com/jjessyli/speciteller
2. Download the lexicons/resources tarball from the UPenn page
   linked in the README and decompress it INSIDE the repo directory.
3. Use the `python3_code/` version of the scripts (the top-level
   scripts are Python 2.7) when running under a Python 3 kernel.
4. pip install numpy liblinear-official
   (liblinear-official exposes `liblinear.liblinearutil`; the repo
   expects a top-level `liblinearutil` module — handled below.)

Usage in a notebook
-------------------
>>> from speciteller_wrapper import SpeciTeller
>>> st = SpeciTeller("/path/to/speciteller")     # slow, once per kernel
>>> st.predict(["Tucked away in the city 's gritty industrial outskirts , "
...             "the factory employed 1,200 workers in 1987 ."])
[0.87...]
"""

import os
import sys


class SpeciTeller:
    def __init__(self, root, codedir="python3_code"):
        """
        root    : absolute path to the cloned speciteller repository
                  (the directory that also contains cotraining_models/
                  and the decompressed lexicon resources).
        codedir : subdirectory holding the Python 3 port of the code.
                  Set to "" if you patched the top-level files yourself.
        """
        self.root = os.path.abspath(root)
        code_path = os.path.join(self.root, codedir) if codedir else self.root

        # The repo's modules (utils.py, features.py, generatefeatures.py)
        # use paths relative to the repo, so chdir + sys.path both matter.
        self._prev_cwd = os.getcwd()
        os.chdir(self.root)
        for p in (code_path, self.root):
            if p not in sys.path:
                sys.path.insert(0, p)

        # Shim: the repo does `import liblinearutil`; modern pip package
        # `liblinear-official` ships it as `liblinear.liblinearutil`.
        try:
            import liblinearutil as ll
        except ImportError:
            from liblinear import liblinearutil as ll
            sys.modules["liblinearutil"] = ll
        self._ll = ll

        import utils
        from features import Space
        from generatefeatures import ModelNewText
        self._ModelNewText = ModelNewText

        m = os.path.join(self.root, "cotraining_models")

        # --- one-time heavy loading (mirrors module level of speciteller.py)
        self.brnclst = utils.readMetaOptimizeBrownCluster()
        self.embeddings = utils.readMetaOptimizeEmbeddings()   # ~190 MB .gz
        self.brnspace = Space(101)
        self.brnspace.loadFromFile(os.path.join(m, "brnclst1gram.space"))

        self.scales_shallow = self._read_scales(os.path.join(m, "shallow.scale"))
        self.scales_neuralbrn = self._read_scales(os.path.join(m, "neuralbrn.scale"))
        self.model_shallow = ll.load_model(os.path.join(m, "shallow.model"))
        self.model_neuralbrn = ll.load_model(os.path.join(m, "neuralbrn.model"))

        os.chdir(self._prev_cwd)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_scales(scalefile):
        scales = {}
        with open(scalefile) as f:
            for line in f:
                k, v = line.strip().split("\t")
                scales[int(k)] = float(v)
        return scales

    @staticmethod
    def _simple_scale(x, maxes):
        newx = []
        for itemd in x:
            newd = {}
            for k, v in itemd.items():
                if k in maxes and maxes[k] != 0:
                    newd[k] = (v + 0.0) / maxes[k]
                else:
                    newd[k] = 0.0
            newx.append(newd)
        return newx

    @staticmethod
    def _score(p_label, p_val):
        ret = []
        for l, prob in zip(p_label, p_val):
            m = max(prob)
            ret.append(1 - m if l == 1 else m)
        return ret

    # ------------------------------------------------------------------ #
    def predict(self, sentences, identifier="nb", return_all=False):
        """
        sentences : list of WORD-TOKENIZED sentence strings
                    (tokens separated by single spaces; the README warns
                    accuracy drops ~4% on untokenized input).
        Returns a list of specificity scores in [0, 1]
        (0 = most general, 1 = most specific), or a dict with the
        combined / shallow / word-representation scores if return_all.
        """
        prev = os.getcwd()
        os.chdir(self.root)  # feature code may read resources relatively
        try:
            aligner = self._ModelNewText(self.brnspace, self.brnclst,
                                         self.embeddings)
            aligner.loadSentences(identifier, sentences)
            aligner.fShallow()
            aligner.fNeuralVec()
            aligner.fBrownCluster()
            y, xs = aligner.transformShallow()
            _, xw = aligner.transformWordRep()

            xs = self._simple_scale(xs, self.scales_shallow)
            xw = self._simple_scale(xw, self.scales_neuralbrn)

            ll = self._ll
            pl, _, pv = ll.predict(y, xs, self.model_shallow, "-q -b 1")
            ls_s = self._score(pl, pv)
            pl, _, pv = ll.predict(y, xw, self.model_neuralbrn, "-q -b 1")
            ls_w = self._score(pl, pv)
            comb = [(a + b) / 2 for a, b in zip(ls_s, ls_w)]
        finally:
            os.chdir(prev)

        if return_all:
            return {"combined": comb, "shallow": ls_s, "wordrep": ls_w}
        return comb

    # Convenience: tokenize with NLTK before scoring
    def predict_raw(self, sentences, **kw):
        from nltk.tokenize import word_tokenize
        tok = [" ".join(word_tokenize(s)) for s in sentences]
        return self.predict(tok, **kw)
