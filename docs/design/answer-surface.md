# Answer Surface Design — Five States

## Overview

This document specifies the complete answer surface for the ia-rag-assistant: how the UI presents retrieved information, how citations attach to claims, how the system communicates confidence, and how refusal works when retrieval returns nothing useful or nothing relevant.

The specification covers five states: **streaming**, **answered**, **weak**, **refused**, and **error**. Each state is enumerated with exact UI requirements and copy examples. No open questions remain — the Frontend agent (step 4.5) must construct all five states without interpretation.

---

## Answer Anatomy — Principles

These principles apply to all states except error:

### Answer text
- **Streaming token-by-token:** The answer streams character by character from the LLM, not buffered. The user sees generation in real time.
- **Prose, not bulleted:** The answer reads as natural language. Avoid itemized lists or regurgitation of retrieval chunks.
- **Short:** Answer only the specific question asked. Do not elaborate beyond what the retrieved documents support.
- **Never without citations:** Every factual claim in the answer must be backed by at least one citation to a retrieved document. If no retrieval gave any relevant content, there is no answer to stream — this is the **refused** state, see below.

### Inline citations
- **Attached to the claim, not the paragraph:** Citations `[1] [2]` appear immediately after the specific sentence or phrase they support, not grouped at the end of a paragraph or section.
- **Claim-level, not document-level:** Multiple claims from one document each cite it by number. The first mention of a document's content is `[1]`; a second claim from the same document later in the answer is also `[1]`. Numbers are assigned by answer appearance order and reused throughout the answer.
- **Example structure:** "The surcharge applies to orders over 500 lbs [1] and accrues daily [2]. Invoicing happens on the first of the month [1]."

### Source panel
- **Title:** The document's full title as it appears in the source corpus.
- **Section path:** If the document is structured (markdown headers, sections, or chapters), the path to the specific section cited.
- **Page number:** Leave blank; the corpus is plain text and not paginated. If future versions include PDFs, the page number where the cited content appears.
- **Click to open:** Clicking the source opens the full document with the cited lines highlighted. If multiple claims cite the same document, opening it shows all cited lines.
- **Mapping:** Citation numbers are assigned by answer appearance order. Citation number `[1]` always maps to the first source in the panel (first document cited in the answer), `[2]` to the second, etc.

### Confidence cue
- **Present when retrieval is weak:** If the retrieval scored low (details of "weak" score delegated to the orchestrator at step 4.1), the answer must say so explicitly. Phrases like "I found only one loosely related passage" or "The documents I found address a similar question" communicate to the user that the system had to work harder to find anything.
- **Never a bare percentage:** Do not show "confidence: 67%". Percentages obscure; prose explains. The LLM-generated answer can name specific weakness patterns.
- **Follows the answer text:** The confidence cue appears after the answer itself, before the source panel. It is optional for answered/strong-retrieval states; required for weak.

### Citations never omitted
- **If retrieved, cited.** Every piece of retrieved content that appears in the answer is cited.
- **If nothing retrieved, no answer.** If retrieval returned zero documents, no relevant documents, or only cross-tenant traps (Invariant 1 forbids serving them), the system does not attempt to generate an answer and moves to the **refused** state.

---

## State 1: Streaming

**When it applies:** The orchestrator has decided an answer should be generated (retrieval scored above threshold; no refusal condition is met). The LLM is generating the answer.

### Surface behavior
- **Streaming begins immediately:** As soon as the orchestrator sends the answer generation request to the LLM, the first UI token appears on screen.
- **Character-by-character flow:** Tokens stream and render one at a time. Users see the answer forming in real time, including any streaming pauses if LLM latency spikes.
- **Citations stream inline:** When the LLM emits `[1]` or `[2]`, those brackets appear inline without waiting for the full claim to complete. The citation number streams as part of the prose.
- **Source panel updates live:** As each new document is cited for the first time (e.g., first `[1]`), it appears in the source panel below the answer. Panel additions do not interrupt streaming; they populate in the background.
- **Streaming completes:** When the LLM finishes (signals end-of-sequence), streaming stops. The answer is now complete and interactive (sources are clickable; if weak, the confidence cue has appeared).

### No state transitions during streaming
- Do not interrupt streaming to check for new conditions (refusal, error, weak confidence detection). These are orchestrator-level decisions made before streaming begins.
- If a network error occurs during streaming, transition to **error** state (see below). The partial answer on screen remains but becomes read-only; error message appears below.

### Copy: None (the LLM controls all text)

---

## State 2: Answered

**When it applies:** Streaming has completed. The full answer and all citations are visible. Retrieval scored high; the system is confident in the answer.

### Surface behavior
- **Full answer displayed:** The complete LLM-generated answer is visible, with all inline citations `[1] [2] …` embedded in the prose.
- **Source panel populated:** All documents cited in the answer are listed in the source panel, mapped 1:1 to citation numbers. Clicking a source opens the full document with the cited lines highlighted.
- **No confidence cue:** In this state, no "I found only…" or weakness language appears. The absence of qualification signals confidence.
- **Interactive:** The answer text is selectable (for copy); sources are clickable.

### Surface layout
```
┌─ Answer (streaming now complete) ────────────────────┐
│ The surcharge applies to orders over 500 lbs [1]      │
│ and accrues daily [2]. Invoicing happens on the       │
│ first of the month [1].                               │
│                                                       │
│ ┌─ Sources ───────────────────────────────┐           │
│ │ [1] Rates and Surcharges Policy         │           │
│ │     → Section: Accessorial Charges      │           │
│ │ [2] Billing Schedule                    │           │
│ └─────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

### Copy: None (state is defined by absence of qualification)

---

## State 3: Weak

**When it applies:** Streaming has completed. The answer was generated, citations are present, but the retrieval scored weakly (delegated definition at 4.1). The system is uncertain of the relevance.

### Surface behavior
- **Full answer displayed:** Same as **answered** state, with all inline citations embedded.
- **Confidence cue appears after answer:** Below the answer text, before the source panel, a prose statement about the weakness appears.
- **Source panel present:** As in **answered** state.
- **Answer is still usable:** The user can read, select, and click sources. This is not a refusal; something plausible was found, but the system is signaling low confidence.

### Confidence cue patterns
These are examples; the LLM generates the actual text based on what was retrieved:

- **Low retrieval count:** "I found only one document that might address your question."
- **Low similarity scores:** "The documents I found address related topics but may not directly answer your question."
- **Partial coverage:** "I found information about accessorial charges but not specifically about [the user's specific subquestion]."
- **Term mismatch:** "The documents use different terminology than your question — they refer to 'handling fees' rather than '[user's term]', so the answer may need translation."

### Surface layout
```
┌─ Answer ─────────────────────────────────────────────┐
│ The surcharge applies to orders over 500 lbs [1]      │
│ and accrues daily [2]. Invoicing happens on the       │
│ first of the month [1].                               │
│                                                       │
│ ⚠ I found only one document that mentions surcharges, │
│   and it doesn't directly address daily accrual.      │
│                                                       │
│ ┌─ Sources ───────────────────────────────┐           │
│ │ [1] Rates and Surcharges Policy         │           │
│ │     → Section: Accessorial Charges      │           │
│ │ [2] Billing Schedule                    │           │
│ └─────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

### Copy: Generated by LLM at inference time, not hard-coded

---

## State 4: Refused

**When it applies:** The orchestrator has determined that no answer should be generated because:

1. **Retrieval returned nothing:** The search returned zero documents.
2. **Retrieval returned no relevant documents:** Documents were returned but scored below the refusal threshold (set at 4.1).
3. **Refusal condition applies:** A special case like cross-tenant traps (RLS prevented serving another tenant's data; Invariant 1).

### Refusal copy requirements

The refusal must **name what was found and why it is insufficient**. A bare "I don't know" or "No information found" reads as broken. The message must explain:

- **What the search DID find** (if anything): Which documents exist, what topics they cover.
- **Why that is insufficient:** How the found documents fail to answer the question.
- **Specificity:** Reference actual document content, not generic placeholders.

### Refusal patterns

#### Pattern A: Retrieval returned zero or low-relevance results
Copy template: `"I could not find anything about [topic] in your documents. Your [document type] covers [what it does cover] but does not mention [specific gap]."`

Example: `"I could not find anything about lithium battery surcharges in your documents. Your hazmat SOP covers general handling procedures but does not mention lithium specifically."`

This pattern applies to questions that retrieve no documents or return documents scoring below the refusal threshold (set at orchestrator step 4.1). Cross-tenant trap questions (e.g., "What is Globex's fuel surcharge?") are also refused at the 4.1 score threshold, per CARRYFORWARD F25: keyword search never returns nothing; it always ranks its best available match. A cross-tenant trap is therefore a low-relevance result, not an empty-retrieval case.

Note: If a question explicitly names another company, you may acknowledge that company name (it appears in the question text itself), but do not assert anything about whether that company exists as a customer or what its documents contain. The user learns isolation held by seeing only their own company's documents in the sources.

Specificity checklist:
- ✓ Confirms the question was understood ("lithium battery surcharges")
- ✓ Names the tenant ("your", or full tenant name if not implied by "your")
- ✓ Names the document type searched ("hazmat SOP")
- ✓ Names what the document *does* cover ("general handling procedures")
- ✓ Names what is *missing* ("lithium specifically")
- ✗ Does not say "I searched" or "I couldn't find" without context
- ✗ Does not say "no information" or "nothing available"
- ✗ Does not dump a list of retrieved documents without relating them to the question
- ✗ Does not assert or imply that another company's documents exist

#### Pattern B: Search found documents, but they do not answer the question
Copy template: `"I found documents about [related topic], but they don't answer your question about [specific question]. The [document type] covers [what it does cover] which is related, but not the same."`

Example: `"I found our detention escalation policies, but they don't answer your question about free time allowance. Our policies cover how charges increase over time, but I cannot find the specific allowance you're asking about."`

Specificity checklist:
- ✓ Acknowledges what *was* found ("detention escalation policies")
- ✓ Explains why it's insufficient ("related, but not the same")
- ✓ Names the gap ("free time allowance" vs what is covered)
- ✗ Does not say "I don't have that information"
- ✗ Does not simply list documents without relating them

#### Pattern C: Search found ambiguous or conflicting documents
Copy template: `"I found multiple documents that could answer your question about [topic], but they give different answers. [Document A] says [claim 1]; [Document B] says [claim 2]. I cannot tell which applies to your situation."`

Example: `"I found multiple policies about handling equipment claims. Your equipment claim form says submit within 30 days, but the carrier's standard terms say 60 days. I cannot tell which applies to your cargo."`

Specificity checklist:
- ✓ Acknowledges the ambiguity explicitly
- ✓ Names each document and its claim
- ✓ Explains why that ambiguity matters ("which applies to your cargo")
- ✗ Does not pick one at random
- ✗ Does not hide the contradiction

### What refusal is NOT

- **Not an error:** Refusal is normal and correct. It is not a system failure.
- **Not a loading state:** Refusal is not shown while the system is still searching. The orchestrator has made a decision and the refusal is final (within the current query session).
- **Not an apology:** "Sorry, I couldn't find that" is weaker than "I found [what I found], but it doesn't answer your question because [why]."

### Surface behavior

- **Refusal message displayed:** The copy appears in the main answer area, replacing where the answer would be.
- **No sources:** If refusal is due to "found nothing", no source panel appears.
- **Sources present if found documents:** If the refusal is "I found relevant documents but they don't answer this", the source panel shows what *was* found, so the user can review and disagree with the refusal decision.
- **No streaming in this state:** Refusal messages are not streamed (they are orchestrator-generated prose, not LLM output). They appear instantly.

### Surface layout — Found nothing
```
┌─ Refusal (found nothing) ──────────────────────────────┐
│ I could not find anything about lithium battery       │
│ surcharges in your documents. Your hazmat SOP covers   │
│ general handling procedures but does not mention      │
│ lithium specifically.                                  │
│                                                       │
│ (No source panel)                                      │
└───────────────────────────────────────────────────────┘
```

### Surface layout — Found something but insufficient
```
┌─ Refusal (found, but insufficient) ────────────────────┐
│ I found documents about detention escalation policies, │
│ but they don't answer your question about the free     │
│ time allowance. Those policies cover how charges      │
│ increase over time, but I cannot find the specific     │
│ allowance for your company.                            │
│                                                       │
│ ┌─ What I found ──────────────────────────┐            │
│ │ Detention Escalation Schedule           │            │
│ │ Detention FAQ (General)                 │            │
│ └─────────────────────────────────────────┘            │
└───────────────────────────────────────────────────────┘
```

### Copy: Generated by orchestrator (not LLM), fixed patterns with dynamic insertion

---

## State 5: Error

**When it applies:** A system failure prevents the orchestrator from reaching a normal state decision:

1. **Retrieval service unreachable:** Embedding API, vector store, or database is down.
2. **LLM service unreachable:** During answer generation (if streaming began and then failed).
3. **Authentication failed:** Invalid session, expired JWT, tenant context lost.
4. **Timeout:** Orchestrator took too long; user may see this mid-stream.
5. **Unknown error:** An unexpected exception.

### Surface behavior

- **Error message displayed:** A short, user-friendly error message appears in the answer area.
- **Specific if possible:** "I cannot reach the document search service" is better than "Something went wrong."
- **No sources:** No source panel.
- **Recovery hint (if applicable):** For timeouts or transient errors, "Try asking again." For auth errors, "Your session may have expired — please re-login."
- **Not streaming:** Error state is not streamed; it appears instantly when the error is caught.
- **Partial answer preserved:** If streaming was underway and an error interrupted, the partial answer on screen remains visible but becomes read-only (user cannot interact). Error message appears below.

### Error message patterns

#### Retrieval error
- "I couldn't search your documents right now. Try again in a moment."
- "The document search service isn't responding. Try asking again."

#### LLM/generation error
- "I started generating an answer but couldn't complete it. Try asking again."
- "The response generation service encountered an error. Try again."

#### Auth error
- "Your session is no longer valid. Please log in again."
- "I lost your tenant context. Please refresh and try again."

#### Timeout error
- "Your question is taking too long to answer. Try a simpler question or try again."

#### Unknown error
- "Something unexpected happened. Please try again, and contact support if the problem continues."

### Surface layout
```
┌─ Error ───────────────────────────────────────────────┐
│ ⚠ I couldn't search your documents right now.         │
│   Try again in a moment.                              │
│                                                       │
│ (No source panel)                                      │
└───────────────────────────────────────────────────────┘
```

### Partial answer on error
```
┌─ Answer (interrupted) ─────────────────────────────────┐
│ The surcharge applies to orders over 500 lbs [1]       │
│ and accrues da…                                         │
│                                                        │
│ ⚠ I started generating an answer but couldn't complete │
│   it. Try asking again.                                │
└────────────────────────────────────────────────────────┘
```

### Copy: Fixed messages with optional dynamic detail

---

## State Transitions and Edge Cases

### Transition: Streaming → Answered
- Normal completion. No action required.

### Transition: Streaming → Weak
- Streaming completes and confidence cue is added retroactively (or streamed as part of the LLM's output, if the orchestrator decides to include "I found only…" in the generation prompt). The source panel is already populated. No disruption.

### Transition: Any state → Error (during streaming)
- Streaming stops. Partial answer remains read-only. Error message appears below.

### Transition: Any state → Error (after streaming)
- Error replaces the entire state. The screen clears to show only the error.

### No automatic retry
- Do not auto-retry if the user is on the error state. The error is shown; the user decides to retry by asking again.

---

## Citations: Mapping and Display

### Citation number assignment
- Assigned by **answer appearance order**: the first document cited in the answer = `[1]`, the second distinct document cited = `[2]`, etc.
- **Not by retrieval order:** A later-retrieved document that appears first in the answer gets `[1]`.
- **Not by document order in the corpus:** Order is determined by where citations appear in the answer text.

### Citation reuse
- If the answer cites the same document multiple times (e.g., `[1]` appears twice), the citation number is reused. Do not increment to `[1a]` or `[1b]`.

### Source panel ordering
- Sources appear in the panel in citation order (top to bottom), matching the order they were first cited in the answer text. Citation number `[1]` always maps to the first source in the panel, `[2]` to the second, etc.

### Cross-document claims
- If a claim requires information from two documents (e.g., "Document A says X, and Document B says Y, so Z"), cite both: "[1] and [2]" or "[1]; [2]" as appropriate.

---

## Out of Scope — What This Spec Does Not Cover

- **API schema:** No endpoint definitions, request/response formats, or HTTP details.
- **Component code:** No React, Vue, or framework-specific implementation.
- **CSS/styling:** Visual appearance is delegated to the frontend design system.
- **Configuration:** No environment variables, feature flags, or settings.
- **Model selection:** Which LLM generation model is used is an orchestrator decision (step 4.1).
- **Citation UI mechanics:** Whether citations are tooltips, badges, or inline text is frontend design; this spec requires only that they are present and map to sources.
- **Accessibility:** WCAG compliance, screen reader support, etc. are frontend-layer concerns.

---

## Acceptance: Enumeration Complete

The five states are fully enumerated:

1. ✓ **Streaming** — Token-by-token generation with live source updates.
2. ✓ **Answered** — Complete answer with confident sources.
3. ✓ **Weak** — Complete answer with confidence cue explaining why retrieval was weak.
4. ✓ **Refused** — Explicit message naming what was found and why insufficient, zero open questions on how to word it.
5. ✓ **Error** — User-friendly error messages with recovery hints where applicable.

**No open questions remain.** The Frontend agent (4.5) can construct all five states without further interpretation.
