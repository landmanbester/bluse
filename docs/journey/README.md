# The journey

Three days, 78 commits, one result — and a much longer list of things that
turned out not to be true.

These files are the *narrative* record, written to be turned into a talk. The
numbers and methods live elsewhere; this is what actually happened, in order,
including the parts that did not work. That is deliberate. Of the four
improvements this project set out to make, **three were rejected by measuring
them** and the fourth ships without meeting its own acceptance criterion. The
one unambiguous win sat untouched in the backlog for the entire three days and
was built on the last night.

| file | what it is | talk section |
|---|---|---|
| [`01-the-arc.md`](01-the-arc.md) | what happened, in order | the spine |
| [`02-wrong-turns.md`](02-wrong-turns.md) | seventeen things we got wrong, and how each surfaced | the heart |
| [`03-what-it-taught-us.md`](03-what-it-taught-us.md) | the working method that emerged from all that | the close |

**Every claim here is checkable.** Commit hashes, file paths and measured
figures are given throughout, and where a number is an inference rather than a
measurement it says so.

## The one-slide version

> We spent two days discovering our instruments were broken, one day
> discovering our planned fixes were wrong, and one evening on a backlog item
> that worked. The score we shipped predicts the multi-beam spatial filter's
> verdict from stamp pixels alone at 0.9899 ROC-AUC. The most useful thing we
> found was not that — it was that the classical filter's own thresholds are
> discarding nine tenths of the information in their inputs.
