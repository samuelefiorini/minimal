# minimal
**MINIMAL** (MatrIx regularizatioN In MAchine Learning) is a Python package that learns matrices solving regularized problems of the form

                    min Loss(Y, XW) + tau * Penalty(W)
                     W

where Y is [n x T], X is [n x d] and W is [d x T]. The solutions are required by means of proximal gradient methods (aka forward-backward splitting).

**MINIMAL** is free software. It is licensed under the Free BSD licence.
