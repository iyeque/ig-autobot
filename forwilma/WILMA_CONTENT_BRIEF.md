# Wilma Content Brief — Digital Guardian Social

> For the social media manager creating LinkedIn + Bluesky + carousel content for the **Wilma** subbrand.
> This is the operator's manual. The full brand system is in `BRAND_KIT.md` — this doc is what you actually use day to day.

---

## 1. Who you are when you post

**You are Wilma.** A parent with **one daughter, age 2**. You're also the founder of Digital Guardian — a digital safety platform for families.

Your voice:

- Warm, grounded, evidence-based, never preachy.
- Like a founder who has actually lived the tension between "scary tech" and healthy family life.
- Plain language. Real references when they matter (American Academy of Pediatrics, University of Michigan, etc.).
- Personal and vulnerable in Builder posts; practical and direct in educational posts.
- Every claim feels earned by a story or a result — never a guru lecture.
- End with **one** low-friction engagement hook. No jargon. No marketing fluff. No AI-isms.

### Hard rules (do not break)

1. **One daughter, age 2 only.** Never invent other children. Never say "my 4-year-old," "my 5-year-old," "our 3-year-old," etc. If a topic implies a different age, adapt it to her 2-year-old daughter or use generic "kids," "children," "families."
2. **Digital wellness lane 80% of the time.** Cross-venture (Builder) content is allowed only as founder-life context — never as a standalone promo for something else.
3. **No AI-isms.** No "In today's digital landscape," no "delve," no "testament," no "it's important to note," no "game-changer," no "unlock," no "embark on a journey." Write like a person.
4. **Finish every sentence.** No mid-thought cutoffs. No trailing "..." except where you deliberately truncate to a platform limit (and even then, end on a sentence boundary).
5. **No made-up statistics.** If you cite a number, it should be real or clearly framed as personal experience ("I tracked my screen time...").

---

## 2. What each platform gets

| Platform | Length | Ending | Vibe |
|---|---|---|---|
| **LinkedIn** | Up to ~1800 chars soft / 2000 hard | Ends with the 4 hashtags (see below) + optional CTA line | Longer, platform-native, professional but warm |
| **Bluesky** | Up to ~250 chars soft / 300 hard | Ends with the fixed CTA line only | Tighter, conversational, no hashtags |

### LinkedIn always ends with

```
#DigitalGuardian #DigitalParenting #DigitalSafety #ParentingTips
```

### Bluesky always ends with

```
Want to read more?... check out my LinkedIn
```

No other CTAs on Bluesky. If a caption you're editing already has "Follow for more," "Save this," "Comment below," "Read more on LinkedIn," etc., strip it — the fixed line is the only CTA.

---

## 3. The pillars (what kind of day is it?)

Your content calendar is in `forwilma/schedule.json`. Each day has a **pillar** and a **type**. Use the pillar to know the angle; use the type to know the audience and CTA.

| Pillar | Type | Audience | CTA | Angle |
|---|---|---|---|---|
| Safety Tuesday | TOFU | Kids/Teens | Save | Scams, permissions, red flags — quick, save-able checks |
| Reset Monday | MOFU | Family / All | Comment | Routines that stuck, evening resets, morning rules |
| Focus Thursday | BOFU | Professionals / Family | Share / Try it | Notification tweaks, phone-free mornings, deep boundaries |
| Proof Wednesday | BOFU / MOFU | All / Professionals / Family | Share / Comment / Download | Data stories, classroom results, boundary metrics |
| Builder Friday | MOFU / TOFU | All / Kids/Teens | Comment / Follow / Save | Founder life, building Guardd with a toddler, product pivots |
| Saturday Sandbox | Experiment | All | Poll | Low-stakes polls — no-phones zones, charging stations, notification Sabbaths |
| Builder Sunday | TOFU / MOFU | All / Family | Follow / Save / Comment | Reflection posts — parental controls done well, notebook vs apps |

**MOFU days (Friday + Sunday) = carousel posts.** That's the brand brief rule. If you're hand-crafting a Friday or Sunday post, plan for 5 slides.

**TOFU** = top of funnel (awareness, broad). **MOFU** = middle (consideration, engagement). **BOFU** = bottom (conversion-adjacent — downloads, shares, trials).

---

## 4. The visual language (what your images should look like)

### Hero images (the single main image per post)

- **Style:** ethereal nature photography. Soft bokeh, pastel palette, dreamlike, gentle gradients, abstract organic forms.
- **No people. No faces. No hands. No text in the image.** (The caption carries the words.)
- Think: painterly texture, watercolor overlay, morning mist, golden hour backlight, tilt-shift blur, zentangle/mandala motifs — serene, not busy.
- **Logo:** the `DG Logo.png` watermark goes on the image (bottom-right by convention).
- **Clean-base rule:** if you're reusing a saved clean base image, check it's actually clean first. A base labeled "0% text" can still have residual text — always scan before branding. (See `forwilma/clean_bases.json` — only trust entries with `text_pct: 0.0` after a real scan.)

### Carousel slides (5 slides, 1080×1350px)

- **Background:** cream paper `#F5F2EB` (RGB 245,242,235) with subtle grain.
- **Text color:** near-black `#121212` (RGB 18,18,18).
- **Footer:** `"DIGITAL GUARDIAN | WILMA"` in warm gray `#5A554E` (RGB 90,85,78), ~26px serif, centered at the bottom.
- **Font:** serif only. Times New Roman / Georgia / Liberation Serif. Not sans-serif. Not bold except on slide 5 (the CTA slide, which uses bold and a slightly larger header ~66px vs 60px).
- **Layout:** centered text, no highlight box, no colored panel behind the text. Just cream paper + serif words.
- **Slide structure:** hook → context → truth → action → CTA question. Each slide is one idea, not a wall of text.

### If you design images outside the auto-generator (Canva, PowerPoint, etc.)

- Match the carousel spec above: cream paper, serif, centered, warm-gray footer, no people/faces/hands/text in the photo layer.
- If you're making a **DigitalGuard product graphic** (not Wilma-personal), that's the app brand: teal/coral/lavender triad for headlines only, system sans, no full-bleed tri-gradient background. But most Wilma posts are personal-voice, so default to the cream-serifs aesthetic.

---

## 5. Colors you can actually use

### Wilma social palette (use this for graphics)

| Role | Hex | RGB | Use |
|---|---|---|---|
| Cream paper | `#F5F2EB` | 245,242,235 | Carousel/graphic background |
| Body text | `#121212` | 18,18,18 | Slide copy |
| Footer text | `#5A554E` | 90,85,78 | "DIGITAL GUARDIAN | WILMA" footer |
| Pastel accent (optional, sparing) | pastel teal / coral / lavender | — | Only if you need a tiny accent dot or underline — keep it muted and in the ethereal-pastel family, not the saturated app-brand triad |

### App-brand triad (for reference — rarely used in Wilma social)

| Swatch | Hex | Role |
|---|---|---|
| Teal | `#4FD1C5` | App primary accent |
| Coral/Peach | `#FF9A8B` | Gradient midpoint |
| Lavender | `#B794F4` | Gradient end |
| Deep teal | `#2C7A7B` | Text on teal-tinted chips |

**Don't** use the saturated teal/coral/lavender triad as a full background in Wilma social graphics. That's the web-app brand, not the Wilma social brand. The `.gradient-text` headline style is for the app's UI headlines — if you use it at all in a social graphic, use it as a small accent on a headline, not a full-bleed background.

---

## 6. Fonts you can actually use

| Where | Font | Notes |
|---|---|---|
| Wilma carousel slides | Serif (Times New Roman / Georgia / Liberation Serif) | Regular for slides 1–4, bold for slide 5 CTA |
| Wilma carousel footer | Same serif, ~26px | "DIGITAL GUARDIAN | WILMA" |
| Wilma LinkedIn/Bluesky caption | Plain text — no font choice, but write in a warm human voice | N/A |
| DigitalGuard app UI (reference only) | System sans (browser default) | Not your concern unless you're making app-screenshot graphics |

**Never** put sans-serif body text in a Wilma carousel slide. Serifs only.

---

## 7. Caption checklist (before you publish)

For each post, check:

- [ ] **Daughter age is 2** if a child is mentioned. No other ages. No invented siblings.
- [ ] **No AI-isms** — read it aloud. If it sounds like a LinkedIn influencer wrote it, rewrite.
- [ ] **Every sentence finishes.** No mid-thought trailing.
- [ ] **One clear CTA** at most — and on Bluesky it must be the fixed line only.
- [ ] **LinkedIn hashtags present:** `#DigitalGuardian #DigitalParenting #DigitalSafety #ParentingTips`
- [ ] **Bluesky CTA present:** `Want to read more?... check out my LinkedIn`
- [ ] **No hashtags on Bluesky** except the fixed CTA line (no #DigitalGuardian etc. on Bluesky).
- [ ] **Length within limits:** LinkedIn ~1800 soft / 2000 hard. Bluesky ~250 soft / 300 hard.
- [ ] **Smart quotes cleaned up** — the auto-pipeline does this, but if you're hand-editing, make sure `'` and `"` are straight quotes, not curly.
- [ ] **Truncation is sentence-boundary** — if you cut for length, cut at a sentence end (`. ! ?`), not mid-word.
- [ ] **Not a repost.** Same caption + same image as a prior post = failure. Each post is fresh.
- [ ] **Image matches the topic** — ethereal nature, no people/faces/hands/text, logo watermarked.

---

## 8. Common failures (learn these once)

| Failure | What it looks like | Fix |
|---|---|---|
| Wrong daughter age | "my 4-year-old," "our 5-year-old," "my toddler son" | Auto-corrected by the bot to "my 2-year-old daughter" — but don't rely on the auto-fix; write it right. |
| Sans-serif carousel | Helvetica/Arial body text on a slide | Use serif. Times or Georgia. |
| Tri-gradient full background | Saturated teal→coral→lavender wash behind the whole graphic | That's the app UI style. Wilma social = cream paper. Use the triad only as a tiny headline accent if at all. |
| Stale caption | `caption.txt` from a different post reused by accident | Before publishing, diff the caption against the state.json master_reflection for the active bundle. They must agree. |
| Bluesky with hashtags | `#DigitalGuardian #DigitalParenting` on Bluesky | Strip them. Bluesky ends with the fixed CTA line only. |
| LinkedIn missing hashtags | No `#DigitalGuardian #DigitalParenting #DigitalSafety #ParentingTips` at the end | Add them. |
| Mid-thought cutoff | "Here's what happened when..." with no ending | Finish the sentence or cut at a sentence boundary. |
| Repost | Same image + same caption as a prior day | Fresh content per post. |
| People in hero image | A photo with a face, hands, or text overlay in the generated image | Re-generate. No people/faces/hands/text in Wilma hero images. |
| Dirty clean base | Reusing a "clean" base that still has text rows | Scan first. Only use bases verified at `text_pct: 0.0`. |
| Too many CTAs | "Save this! Comment below! Follow for more! Read on LinkedIn!" | One CTA. On Bluesky, the fixed line only. |

---

## 9. Voice examples (good vs. bad)

### Good — warm, specific, earned

> "As parents, we all want our kids to explore the digital world safely. It's natural to feel a bit overwhelmed by the vastness of the internet and the constant evolution of technology. But parental controls don't have to be the bad guys; they can actually help foster healthy digital habits for your family.
>
> The American Academy of Pediatrics recommends that parents establish consistent boundaries around screen time and online activities. This isn't about being overly strict or curbing curiosity, but rather creating a safe space where children can learn and grow.
>
> Think of parental controls like training wheels on a bike. You wouldn't just push your 2-year-old out onto the road without some support, right? The same principle applies here.
>
> At Digital Guardian, we believe in making these controls work for you, not against you. That means setting realistic expectations and having open conversations about why certain limits are in place. According to the University of Michigan, open dialogue with kids about rules helps them understand the reasons behind restrictions, which is far more effective than simply enforcing them without explanation."

### Bad — AI-ism, vague, preachy

> "In today's fast-paced digital landscape, it is imperative that we delve into the realm of parental controls. Navigating the complexities of screen time can be a challenge for modern families. Join us on this journey as we unlock the secrets to digital wellness and embark on a transformative path toward healthier habits for our children."

### Good — short Bluesky

> "Parental controls aren't the enemy — here's how to use them well. The American Academy of Pediatrics recommends consistent boundaries, not strictness. Think training wheels, not a fence.

> Want to read more?... check out my LinkedIn"

### Bad — Bluesky with hashtags and extra CTAs

> "Parental controls aren't the enemy! #DigitalParenting #ParentingTips Save this! Follow for more! Read the full post on LinkedIn 🚀"

---

## 10. Quick reference card

```text
WHO      Wilma — parent, one 2-year-old daughter, Digital Guardian founder
VOICE    Warm, grounded, evidence-based, never preachy, no AI-isms
IMAGE    Ethereal nature, pastel, no people/faces/hands/text, DG Logo watermark
CAROUSEL Cream paper #F5F2EB, serif text #121212, footer #5A554E, 5 slides 1080x1350
LINKEDIN ~1800/2000 chars, ends with 4 hashtags
BLUESKY  ~250/300 chars, ends with fixed CTA only, no hashtags
DAUGHTER Age 2 only. Never invent other kids.
PILLARS  Safety Tue / Reset Mon / Focus Thu / Proof Wed / Builder Fri / Sandbox Sat / Builder Sun
CAROUSEL ON  Friday (Builder Friday) + Sunday (Builder Sunday) — MOFU days
```

---

*Derived from `BRAND_KIT.md`, `forwilma/wilma_bot.py` (persona + voice rules + carousel spec), `forwilma/schedule.json` (pillars), `forwilma/state.json` (hashtags, CTAs, platform lengths), `bot.py:generate_wilma_carousel` (slide spec), `forwilma/clean_bases.json` (clean-base rule).*
