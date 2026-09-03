# The journey

Four days, one result — and a much longer list of things that turned out not
to be true, including half of that result's own published conclusions.

These files are the *narrative* record, written to be turned into a talk. The
numbers and methods live elsewhere; this is what actually happened, in order,
including the parts that did not work. That is deliberate. Of the four
improvements this project set out to make, **three were rejected by measuring
them** and the fourth ships without meeting its own acceptance criterion. The
one unambiguous win sat untouched in the backlog for three days, was built on
the last night — and on day 4, **four of its eight published conclusions were
withdrawn**, two of them by a reviewer within hours of the first outside read.

| file | what it is | talk section |
|---|---|---|
| [`01-the-arc.md`](01-the-arc.md) | what happened, in order | the spine |
| [`02-wrong-turns.md`](02-wrong-turns.md) | twenty-four things we got wrong, and how each surfaced | the heart |
| [`03-what-it-taught-us.md`](03-what-it-taught-us.md) | the working method that emerged from all that | the close |

**Every claim here is checkable.** Commit hashes, file paths and measured
figures are given throughout, and where a number is an inference rather than a
measurement it says so.

## The one-slide version

> We spent two days discovering our instruments were broken, one day
> discovering our planned fixes were wrong, one evening on a backlog item that
> worked, and one day discovering how much of that was wrong too. The score we
> shipped predicts the multi-beam spatial filter's verdict from stamp pixels
> alone at 0.9899 ROC-AUC — and injections showed it would *discard* a bright
> non-drifting sky signal 91% of the time. The most useful things we found were
> both about someone else's code: the classical filter's thresholds discard
> nine tenths of the information in their inputs, and one delivered file's beam
> counts are artefacts of how sparsely it was sampled.
