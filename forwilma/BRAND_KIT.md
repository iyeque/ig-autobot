# DigitalGuard Brand Kit

> Digital wellness / family digital parenting platform.
> Clean, calm, warm-tech aesthetic with a tri-color teal / coral / lavender accent gradient.
> Social media subbrand **Wilma** carries a separate cream-paper / serif / ethereal-naturalism visual language.

---

## 1. Identity

### Product (DigitalGuard web app)

| | |
|---|---|
| **Product name** | DigitalGuard |
| **Tagline** | Digital Wellness Reimagined |
| **One-liner** | Take control of your digital life — screen time, digital wellness, online safety. |
| **Logo asset** | `public/dg-logo.png` (PNG, used as favicon + OG + Twitter image + nav mark) |
| **Wordmark** | `DigitalGuard` set in `font-black tracking-tight uppercase` in the nav |

### Social subbrand (Wilma — `forwilma/`)

| | |
|---|---|
| **Subbrand name** | Wilma (Digital Guardian's social voice) |
| **Persona** | A parent with ONE daughter, age 2. Founder-voice: warm, grounded, evidence-based, never preachy. |
| **Mission** | Bridge the gap between children's online exploration and well-being. Values: healthy digital habits, family bonds, open conversations, proactive safety. |
| **Voice** | Empathetic, authoritative, research-backed, relatable. Plain and warm. Personal/vulnerable in Builder content; practical/direct in educational content. Every claim earned by story or result — no guru lecture. |
| **Platforms** | LinkedIn + Bluesky (primary); Instagram carousel on MOFU days (Friday/Sunday) per brand brief. |
| **Logo asset** | `forwilma/DG Logo.png` — applied as watermark on all Wilma hero images. |
| **Carousel footer** | `DIGITAL GUARDIAN | WILMA` in warm gray serif, centered at bottom of every slide. |

### Wilma hard persona rules (enforced in code)

- One daughter, age 2 only. Never invent other children or ages. `"my 4-year-old"`, `"my 5-year-old"` etc. are auto-corrected to `"my 2-year-old daughter"` by `_enforce_wilma_persona()`.
- Children in content = her 2-year-old daughter OR generic `"kids"`, `"children"`, `"families"`.
- Digital wellness lane 80% of the time. Cross-venture (Builder) content only as founder-life context, never standalone promo.
- No AI-isms, no marketing fluff, no mid-thought cutoffs.

---

## 2. Color System

### 2a. Web app — light mode (`:root` / `src/index.css`)

| Token | HSL | Approx. hex | Role |
|---|---|---|---|
| `--background` | `0 0% 100%` | `#FFFFFF` | Page background |
| `--foreground` | `222.2 47.4% 11.2%` | `#0F172A` | Primary text |
| `--muted` | `210 40% 96.1%` | `#F1F5F9` | Subdued surfaces |
| `--muted-foreground` | `215.4 16.3% 35%` | `#5B6470` | Secondary text |
| `--card` | `0 0% 100%` | `#FFFFFF` | Card background |
| `--card-foreground` | `222.2 47.4% 11.2%` | `#0F172A` | Card text |
| `--popover` | `0 0% 100%` | `#FFFFFF` | Popover bg |
| `--border` | `217.2 32.6% 70%` | `#CBD5E0` | Default borders |
| `--input` | `217.2 32.6% 70%` | `#CBD5E0` | Input borders |
| `--ring` | `222.2 47.4% 11.2%` | `#0F172A` | Focus ring |
| `--primary` | `222.2 47.4% 11.2%` | `#0F172A` | Primary actions/text |
| `--primary-foreground` | `210 40% 98%` | `#F8FAFC` | Text on primary |
| `--secondary` | `210 40% 96.1%` | `#F1F5F9` | Secondary surfaces |
| `--secondary-foreground` | `222.2 47.4% 11.2%` | `#0F172A` | Text on secondary |
| `--accent` | `210 40% 96.1%` | `#F1F5F9` | Accent surfaces |
| `--accent-foreground` | `222.2 47.4% 11.2%` | `#0F172A` | Text on accent |
| `--destructive` | `0 100% 50%` | `#EF4444` | Destructive / error |
| `--destructive-foreground` | `210 40% 98%` | `#F8FAFC` | Text on destructive |

### 2b. Web app — dark mode (`.dark`)

| Token | HSL | Approx. hex | Role |
|---|---|---|---|
| `--background` | `222.2 84% 4.9%` | `#0F172A` | Page background |
| `--foreground` | `210 40% 98%` | `#F8FAFC` | Primary text |
| `--muted` | `217.2 32.6% 17.5%` | `#1E293B` | Subdued surfaces |
| `--muted-foreground` | `215 20.2% 75%` | `#A1A5B0` | Secondary text |
| `--card` | `222.2 84% 4.9%` | `#0F172A` | Card background |
| `--card-foreground` | `210 40% 98%` | `#F8FAFC` | Card text |
| `--primary` | `210 40% 98%` | `#F8FAFC` | Primary actions/text |
| `--primary-foreground` | `222.2 47.4% 11.2%` | `#0F172A` | Text on primary |
| `--secondary` | `217.2 32.6% 17.5%` | `#1E293B` | Secondary surfaces |
| `--border` | `217.2 32.6% 25%` | `#334155` | Borders |
| `--ring` | `212.7 26.8% 83.9%` | `#CBD5E1` | Focus ring |
| `--destructive` | `0 62.8% 30.6%` | `#B91C1C` | Destructive / error |

### 2c. Brand accent triad (gradient core — used in `.gradient-text`, ambient blobs, hover states)

| Swatch | Hex | Role |
|---|---|---|
| **Teal** | `#4FD1C5` | Primary interactive accent — buttons, progress, focus rings, links, badge tints (`bg-[#4FD1C5]/10` for chips) |
| **Coral / Peach** | `#FF9A8B` | Warm secondary — gradient midpoint, ambient blob, rose-family hover |
| **Lavender** | `#B794F4` | Cool tertiary — gradient end, ambient blob |
| **Deep teal** | `#2C7A7B` | Darker teal for text on teal-tinted chips (`text-[#2C7A7B]`) |

**`.gradient-text`** (branded headline gradient):
```css
background: linear-gradient(135deg, #4FD1C5 0%, #FF9A8B 50%, #B794F4 100%);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
background-clip: text;
```

### 2d. Ambient background gradients

**Light mode ambient** (`UnifiedBackground` base layer):
`from-[#FFF5F5] via-[#F0FFF4] to-[#E6FFFA]` — warm pink → mint → pale teal.

**Dark mode ambient**:
`from-[#0f172a] via-[#1e293b] to-[#0f172a]` — slate navy.

### 2e. Animated ambient blobs (lazy-parallax, fine-pointer only)

| Blob | Color | Opacity | Blur |
|---|---|---|---|
| Coral/Peach | `#FF9A8B` | 30% light / 15% dark | 60px |
| Mint/Teal | `#4FD1C5` | 25% light / 10% dark | 50px |
| Lavender | `#B794F4` | 25% light / 15% dark | 60px |

### 2f. Wilma carousel palette (social subbrand — `bot.py:generate_wilma_carousel`)

| Role | RGB | Hex | Usage |
|---|---|---|---|
| Paper background | `(245, 242, 235)` | `#F5F2EB` | Full-bleed cream paper, all 5 slides |
| Body text | `(18, 18, 18)` | `#121212` | Centered serif slide copy |
| Footer text | `(90, 85, 78)` | `#5A554E` | `"DIGITAL GUARDIAN | WILMA"` footer |
| Paper grain | ±12 shade noise | — | `GaussianBlur(1.2)` at 18% blend, slides 1–4 |

**Wilma image generation prompts** (`WILMA_BRAND_BASE` + `WILMA_BRAND_SUFFIX`):
- Base: `ethereal nature photography, soft bokeh, pastel color palette, dreamlike atmosphere, gentle gradients, abstract organic forms, no people, no figures, no faces, no hands, no text`
- Suffix: `fine art print, painterly texture, studio ghibli inspired, watercolor overlay, serene mood, zentangle patterns, mandala motifs, tilt-shift blur, macro lens, morning mist, golden hour backlight`

---

## 3. Typography

### 3a. Web app — system font stack (no custom web font loaded)

The app uses the **browser/system font stack** (Tailwind default sans). Visual punch comes from weight and tracking, not a custom typeface.

| Element | Classes | Weight / style |
|---|---|---|
| Wordmark / logo text | `text-2xl font-black tracking-tight uppercase` | 900, tight tracking, all-caps |
| H1 / hero headlines | `text-3xl md:text-6xl font-bold` (or `font-black` for impact) | 700–900 |
| H2 | `text-2xl md:text-4xl font-bold` | 700 |
| H3 | `text-xl font-bold` | 700 |
| Section labels / eyebrows | `text-xs font-black uppercase tracking-widest` | 900, uppercase, wide tracking |
| Body | `text-sm font-medium` or default | 400–500 |
| Button labels | `font-bold` or `font-medium` | 600–700 |
| Gradient text | `.gradient-text` utility | any — clipped to tri-gradient |

**Gap / recommendation:** No custom typeface is loaded. If a brand font is wanted, a single humanist sans (Inter, Manrope, or Plus Jakarta Sans) loaded in `index.html` and set in `tailwind.config.ts` under `theme.extend.fontFamily.sans` would sharpen identity without changing component code.

### 3b. Wilma carousel — serif (social subbrand)

| Role | Font | Fallback chain (from `bot.py`) |
|---|---|---|
| Slide body (regular) | Serif, ~60px | `times.ttf` → `georgia.ttf` → `LiberationSerif-Regular` → `DejaVuSerif` → default |
| Slide body (bold / CTA slide) | Serif bold, ~60px | `timesbd.ttf` → `georgiab.ttf` → `LiberationSerif-Bold` → `DejaVuSerif-Bold` → default |
| Footer | Serif, ~26px | Same regular chain |

**Wilma carousel spec:** 1080×1350px, 5 slides, centered text, cream paper (#F5F2EB), no highlight box, footnote footer. CTA slide (slide 5) uses bold and a slightly larger header (66px vs 60px).

### 3c. Wilma caption voice

Plain, warm, direct. No jargon. No AI-isms. Sentence-boundary truncation only (never mid-word). Platform-specific:
- **LinkedIn:** longer, platform-native, ends with `#DigitalGuardian #DigitalParenting #DigitalSafety #ParentingTips`
- **Bluesky:** tighter, conversational, ends with fixed CTA: `Want to read more?... check out my LinkedIn`

---

## 4. Spatial System & Shapes

| Token | Value | Usage |
|---|---|---|
| `--radius` | `0.5rem` (8px) | Base radius, default for cards/inputs |
| `rounded-xl` | 12px | Common card/container |
| `rounded-2xl` | 16px | Larger cards, feature blocks |
| `rounded-3xl` | 24px | Hero cards, workshop completion |
| `rounded-[2rem]` | 32px | Signature/commitment cards |
| `rounded-[2.5rem]` | 40px | Avatars / circular badge frames |
| `rounded-full` | full | Chips, pills, avatar initials |

**Chips / pills pattern:** `rounded-full` pill with `bg-[#4FD1C5]/10 text-[#2C7A7B]` for teal-tinted info badges; `bg-white/60 dark:bg-black/20 border-border hover:border-primary/30` for rule cards.

---

## 5. Motion & Atmosphere

### Background layer stack (`Layout.tsx`, `-z-20`, `pointer-events-none`)

1. **`UnifiedBackground`** — base ambient gradient (light or dark)
2. **3 animated blobs** (coral, mint, lavender) with `animate-drift-1/2/3` (18–22s loops) + mouse parallax on fine-pointer devices only
3. **SVG noise overlay** at `opacity-[0.03]` (light) / `0.05` (dark)
4. **`MoodColorShift`** — subtle mood-based radial-gradient overlay
5. **`OrganicTexture`** — additional texture layer
6. **`WellnessParticles`** — particle ambiance
7. **`GentleTrail`** — cursor trail (auto-disabled on mobile)
8. **`WellnessCompanion`** — AI companion bubble (bottom-right)

### Key animations (`src/index.css`)

| Animation | Duration | Usage |
|---|---|---|
| `animate-drift-1/2/3` | 18–22s ease-in-out infinite | UnifiedBackground blobs |
| `animate-breathe` | **12s** ease-in-out infinite | Breathing/relaxation UI rhythm (scale 1 → 1.05, opacity 1 → 0.8) |
| `accordion-down / accordion-up` | 0.2s ease-out | Accordion expand/collapse |

### Motion philosophy

- **Calm, slow, ambient.** Blob drifts and breathing loops are slow (12–25s) and subtle — background atmosphere, not attention-grabbing.
- **Breathing at 12s** anchors the wellness/relaxation pacing.
- **`prefers-reduced-motion`** is respected: all animations + transitions collapse to `0.01ms` with `animation-iteration-count: 1`.
- **Interaction feedback** is snappy: `transition-all duration-300` on buttons, `scale-105` / `scale-[1.02]` hover, `active:scale-[0.98]`.

---

## 6. Component Patterns

### Buttons

| Variant | Classes | When to use |
|---|---|---|
| **Primary CTA** | `bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white font-black shadow-xl shadow-green-500/20 rounded-xl transition-all` | Main action buttons |
| **Secondary / ghost** | `bg-white dark:bg-slate-800 border-2 border-slate-200 dark:border-slate-700 font-medium hover:border-[#4FD1C5] transition-colors` | Secondary actions, teal accent on hover |
| **Teal pill** | `bg-[#4FD1C5] text-white font-bold shadow-lg` | In-flow actions (e.g. course "Start" buttons) |

### Cards

| Variant | Classes | When to use |
|---|---|---|
| **Default** | `bg-white dark:bg-slate-800 border-2 border-slate-200 dark:border-slate-700 rounded-xl` | Standard content cards |
| **Glass / translucent** | `bg-white/60 dark:bg-white/5 backdrop-blur-xl border border-border rounded-[2rem]` | Commitments, feature blocks |
| **Tinted gradient card** | `bg-gradient-to-r from-[#4FD1C5]/10 to-[#FF9A8B]/10 border border-[#4FD1C5]/20 rounded-2xl` | Featured/hero content |

### Navigation

- Fixed top nav: `bg-white/10 dark:bg-black/10 backdrop-blur-md border-b border-border transition-all duration-300`
- Logo: `dg-logo.png` (h-8 w-8 rounded-md) + `DigitalGuard` wordmark `text-2xl font-black tracking-tight uppercase`
- Mobile: hamburger menu with `Menu` / `X` icons

### Wilma carousel slides (social subbrand)

- 5 slides, 1080×1350px, cream paper (#F5F2EB) with subtle grain
- Centered serif text, no highlight box
- Footer: `DIGITAL GUARDIAN | WILMA` in warm gray (#5A554E), 26px serif
- Slide 5 (CTA) uses bold + 66px header
- No people/faces/hands/text in the hero image — only abstract organic forms

### Footer

- Fixed, see-through: `Footer.tsx` — glass-style bottom bar.

---

## 7. Imagery & Iconography

### Icons

`lucide-react` — consistent 1.5px stroke throughout (ArrowLeft, Sun/Moon, Check, Upload, AlertTriangle, Loader2, Download, ChevronRight, GraduationCap, etc.).

### Logo mark

- **Web app:** `public/dg-logo.png` — placed in nav, OG, Twitter card, favicon.
- **Wilma social:** `forwilma/DG Logo.png` — applied as watermark on all Wilma hero images via `apply_logo_watermark()`.

### Wilma hero image style

AI Horde prompt = `WILMA_BRAND_BASE + visual_metaphor + WILMA_BRAND_SUFFIX`. The visual metaphor is the topic itself (passed through `_generate_wilma_visual_prompt()`). No people, faces, hands, or text in the generated image. Clean-base fallback images are tracked in `forwilma/clean_bases.json` — only bases with `text_pct: 0.0` are truly clean; always verify before branding.

### Blog/hero imagery (web app)

Inline Tailwind gradient backgrounds via `bg-gradient-to-br from-X-to-Y` (defined per-resource in `public/blog-resources.json`) — blue, green, purple, yellow, red, cyan, slate variants.

---

## 8. Content Pillars (Wilma schedule — `forwilma/schedule.json`)

| Pillar | Type | Audience | CTA | Example topic |
|---|---|---|---|---|
| Safety Tuesday | TOFU | Kids/Teens | Save | Scam checks, app permissions, text scams |
| Reset Monday | MOFU | Family / All | Comment | Phone-free routines, evening resets, morning rules |
| Focus Thursday | BOFU | Professionals / Family | Share / Try it | Phone-free mornings, notification tweaks, deep boundaries |
| Proof Wednesday | BOFU / MOFU | All / Professionals / Family | Share / Comment / Download | Classroom bans, screen-time data, boundary metrics |
| Builder Friday | MOFU / TOFU | All / Kids/Teens | Comment / Follow / Save | Founder life, building Guardd with a toddler, product pivots |
| Saturday Sandbox | Experiment | All / Kids/Teens / Professionals / Family | Poll | No-phones zones, charging stations, notification Sabbaths |
| Builder Sunday | TOFU / MOFU | All / Family | Follow / Save / Comment | Parental controls done well, notebook vs apps, user reviews |

**MOFU days (Friday/Sunday)** enforce carousel posts per the Digital Guardian brand brief.

---

## 9. Do's and Don'ts

### Do

- Use **teal (#4FD1C5)** as the primary interactive accent — buttons, links, focus, progress, chip tints.
- Use the **tri-gradient (#4FD1C5 → #FF9A8B → #B794F4)** sparingly, for branded headlines (`.gradient-text`) and featured card tints only.
- Keep the web app background ambient — slow blobs, breathing at 12s, noise overlay — and let content sit on calm surfaces.
- Respect `prefers-reduced-motion`.
- Pair dark text on light surfaces; off-white (#F8FAFC) on dark surfaces — always check contrast.
- For Wilma carousels: cream paper (#F5F2EB), centered serif, warm gray footer, no highlight box.
- For Wilma hero images: ethereal nature, pastel, no people/faces/hands/text.
- Keep Wilma's daughter age at exactly 2 in all content. Never invent other children.
- End LinkedIn posts with `#DigitalGuardian #DigitalParenting #DigitalSafety #ParentingTips`.
- End Bluesky posts with `Want to read more?... check out my LinkedIn`.

### Don't

- Don't use the tri-gradient as a full-bleed background in content areas — it's for accents and headlines only.
- Don't override `--radius` ad-hoc — use the named Tailwind radius scale (`rounded-xl/2xl/3xl/[2rem]`).
- Don't introduce a fourth accent color without checking it against the triadic teal/coral/lavender palette.
- Don't drop the noise overlay or ambient blobs in a "minimal" variant — they are the brand atmosphere.
- Don't put people, faces, hands, or text in Wilma hero images.
- Don't use sans-serif in Wilma carousels — serif only (Times/Georgia/LiberationSerif).
- Don't reuse a caption+image pair from a prior post — each post must be fresh.
- Don't let `caption.txt`, `output.jpg`, and `state.json` master_reflection drift out of sync on an active bundle.

---

## 10. Quick Reference (copy-paste tokens)

```css
/* Brand gradient (headlines) */
.gradient-text {
  background: linear-gradient(135deg, #4FD1C5 0%, #FF9A8B 50%, #B794F4 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* Core accent tokens */
--accent-teal: #4FD1C5;
--accent-teal-deep: #2C7A7B;
--accent-coral: #FF9A8B;
--accent-lavender: #B794F4;

/* Light ambient bg */
background: linear-gradient(to bottom right, #FFF5F5, #F0FFF4, #E6FFFA);

/* Dark ambient bg */
background: linear-gradient(to bottom right, #0f172a, #1e293b, #0f172a);

/* Primary button */
bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700
text-white font-black shadow-xl shadow-green-500/20 rounded-xl transition-all

/* Secondary button (teal-accent on hover) */
bg-white dark:bg-slate-800 border-2 border-slate-200 dark:border-slate-700 font-medium
hover:border-[#4FD1C5] transition-colors

/* Teal pill */
bg-[#4FD1C5] text-white font-bold shadow-lg

/* Teal-tinted info chip */
rounded-full bg-[#4FD1C5]/10 text-[#2C7A7B]

/* Wordmark */
text-2xl font-black tracking-tight uppercase

/* Breathing animation */
animate-breathe  /* 12s ease-in-out infinite, scale 1→1.05, opacity 1→0.8 */
```

```python
# Wilma carousel constants (bot.py:generate_wilma_carousel)
BG = (245, 242, 235)        # #F5F2EB cream paper
TEXT_COLOR = (18, 18, 18)   # #121212 near-black
FOOTER_COLOR = (90, 85, 78) # #5A554E warm gray
FOOTER_TEXT = "DIGITAL GUARDIAN | WILMA"
SLIDE_SIZE = (1080, 1350)
```

```python
# Wilma image prompt parts (wilma_bot.py)
WILMA_BRAND_BASE = (
    "ethereal nature photography, soft bokeh, pastel color palette, "
    "dreamlike atmosphere, gentle gradients, abstract organic forms, "
    "no people, no figures, no faces, no hands, no text"
)
WILMA_BRAND_SUFFIX = (
    "fine art print, painterly texture, studio ghibli inspired, "
    "watercolor overlay, serene mood, zentangle patterns, mandala motifs, "
    "tilt-shift blur, macro lens, morning mist, golden hour backlight"
)
```

```text
# Wilma LinkedIn hashtags
#DigitalGuardian #DigitalParenting #DigitalSafety #ParentingTips

# Wilma Bluesky fixed CTA
Want to read more?... check out my LinkedIn
```

---

*Sources: `src/index.css` (web app tokens), `tailwind.config.ts`, `bot.py:generate_wilma_carousel` (carousel spec), `forwilma/wilma_bot.py` (persona, brand prompts, voice rules), `forwilma/schedule.json` (content pillars), `forwilma/state.json` (platforms, hashtags, CTAs), `README.md` (architecture overview).*
