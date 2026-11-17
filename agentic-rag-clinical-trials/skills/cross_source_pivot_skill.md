---
name: cross-source-pivot
description: Pivoting between AACT (structured) and the ingested research papers (unstructured) when a question needs both sources tied together.
allowed-tools: [linkage_lookup, extract_entities, retrieve_chunks, sql_exec_tool]
---

## Pivoting between AACT and the papers (cross-source pivot skill)

- If a question names a specific NCT ID, call `linkage_lookup` first - it
  pulls the AACT record and searches the papers for related content in one
  step.
- If a question names a study by a paper's own informal/colloquial name
  (e.g. "the Kisumu Combined Ring Study") instead of an NCT ID, do NOT go
  looking for that name in AACT's `studies` table - a paper's nickname for
  a trial usually doesn't appear anywhere in AACT's `brief_title` or
  `official_title` (those are the trial's registered clinicaltrials.gov
  title, which can look completely different). Searching for it there with
  `sql_gen_tool`/`sql_exec_tool` will just burn calls on zero-row results.
  Instead: `retrieve_chunks` on that name first - trial papers usually
  state their own NCT ID directly in the text (often near "ClinicalTrials.gov
  Identifier"). Pass the ACTUAL retrieved chunk text (not just the study
  name again) into `extract_entities` - it regex-extracts any NCT ID
  pattern in the text you give it, so make sure that's really the chunk
  content. Then `linkage_lookup` that ID to confirm and get the full AACT
  record.
- If a paper chunk mentions a trial but no NCT ID turns up anywhere, run
  `extract_entities` on that chunk to pull out condition/intervention names
  instead, then query AACT's `conditions`/`interventions` tables for
  matching studies.
- Don't assume a match just because both sources returned something for the
  same topic - `linkage_lookup`'s paper results are a best-effort topical
  match, not a confirmed reference to that trial. Say so if you're not sure.
