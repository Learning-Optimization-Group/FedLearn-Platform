# C5 — Visual Design, UX & Communication (Web + Desktop + Mobile)

**Date:** 2026-05-29
**Scope:** All three UI surfaces — `frontend/` (React 19 + Tailwind v4), `fedlearn-desktop/src/renderer/` (Electron + plain CSS), `origin/fed-mobile:mobile_client/src/` (React Native).
**Target calibration:** Production-grade startup. Recommendations weigh cost, maintainability, accessibility (WCAG 2.2 AA), and — the emphasized ask — **performance observability of the FL runs users create**.
**Builds on:** `docs/audit/2026-05-27/02-frontend-desktop.md` (cited inline; not duplicated). That report covered security/perf/test-coverage of the web+desktop code. **This report is the design/UX/communication layer** — the cross-surface visual system, IA, FL data-visualization, core flows, and brand. Where 02-frontend-desktop already nailed something (CSP, `any`-typing, modals-not-`<dialog>`), I cite and extend rather than restate.

---

## 0. TL;DR

The platform has **three UIs that look like three different products from three different companies.** Each was built in isolation with its own color system, its own typography, its own component vocabulary, and even its own product name (FedLearn / FedLearn Desktop / **FedMob**). There is **no shared design system, no shared tokens, no shared component library** — not even a shared accent color. The web app is genuinely well-crafted (a coherent OKLCH "Instrument" token system, a striking `FederationOrrery`); the desktop renderer is a 1,296-line hand-rolled CSS file with **two conflicting `:root` token blocks**; the mobile app is unstyled Bootstrap-era boilerplate (inline `StyleSheet`, hardcoded `#007bff`/`#28a745`, emoji tab icons).

The single highest-leverage v2 move is a **shared design-token + component system spanning all three surfaces**: shadcn/ui (Radix/Base-UI on Tailwind v4) for web+desktop, [react-native-reusables](https://reactnativereusables.com/) (NativeWind v4) for mobile, with **one OKLCH token source of truth** published as a tiny package. Second-highest: the FL data-visualization story is thin — a portfolio convergence chart and a beautiful-but-decorative orrery — and does not yet deliver the per-client / per-round / cost observability that is the platform's reason to exist. Benchmark target is W&B + Grafana for the run-observability surface, Linear/Vercel for the shell.

---

## 1. Cross-surface visual consistency — the core problem

### 1.1 Three brands, three palettes, three names

| Surface | Product name | Accent | Theme | Typography | Styling tech |
|---|---|---|---|---|---|
| Web `frontend` | **FedLearn** | Orange `oklch(0.62 0.24 48)` ("Federation Pulse") | Light + dark, OKLCH tokens | IBM Plex Sans / Mono / Instrument Serif | Tailwind v4 + `@theme` tokens |
| `fedlearn-desktop` | **"FedLearn Desktop"** | **Conflicting: orange `#e55c26` AND indigo `#6366f1`** | Dark only, glassmorphism | Inter / JetBrains Mono | Hand-rolled CSS (1,296 lines) |
| `mobile_client` | **FedMob** | Blue `#007bff` + green `#28a745` (Bootstrap defaults) | Light only | System default | Inline RN `StyleSheet` |

**Evidence — the desktop file contradicts itself.** `fedlearn-desktop/src/renderer/styles.css` declares **two separate `:root` blocks**:
- Line 10–50: a "premium" vocabulary — `--color-accent: #e55c26` (orange, line 25), `--glass-bg`, `--color-text-primary: #e4e4ef`.
- Line 932–957: a *second* block whose comment literally says *"Design tokens — mirrors the web app's theme.css"* but sets `--accent-primary: #6366f1` (indigo, line 942) and `--bg-card: #1a1e25`.

So the desktop app ships **two accent colors that don't match each other and neither matches the web's orange** (`frontend/src/styles/theme.css:26`, `oklch(0.62 0.24 48)` ≈ `#d24a17`). The comment claiming parity with the web theme is false. Components reference both token sets (`var(--color-accent)` and `var(--accent-primary)` both appear), so the rendered accent depends on which selector wins — a coin-flip design system.

**Evidence — mobile is unthemed.** Every `mobile_client/src/screens/*.jsx` hardcodes Bootstrap hex literals inline: `backgroundColor: '#007bff'` (FL button), `'#28a745'` (connect/test), `'#dc3545'` (stop/delete), `'#f8f9fa'` (card bg), repeated across `TrainingScreen.jsx`, `ModelLibraryScreen.jsx`, `InferenceScreen.jsx`. Tab icons are **emoji** (`🤖 📚 🔍` in `AppNavigator.jsx:55,64,73`). The verified screenshot (`phone_screen.png`, 1080×2316) shows a flat light card layout titled **"FedMob"** with green/blue buttons — visually a different product from the web's warm orange Instrument aesthetic.

**Verdict: the cross-surface visual system is `rebuild`.** There is nothing to salvage at the system level because there is no system — only three local conventions. (Individual *web* tokens are salvageable as the seed; see §6.)

### 1.2 The web token system is the one genuinely good asset

`frontend/src/styles/theme.css` is real design-system work: OKLCH color space (perceptually uniform, future-proof for wide-gamut), semantic token layering (`--accent-primary` → `--primary` → `@theme inline --color-primary`), light/dark parity, chart tokens (`--chart-1..5`), sidebar tokens, and a `:focus-visible` ring on all native controls (lines 255–261 — a real a11y win). This is the seed for the v2 token package. **Verdict: `salvage` the token file; promote it to the shared source of truth.**

**Caveat (consistency erosion inside the web app itself):** despite the token system, the codebase still scatters raw hex. `TrainingInsightsView.tsx:130` hardcodes `stroke="#ef4444"` for the loss line and `#ef4444` error styling (lines 76); `LogViewer.tsx:315,343` hardcode `#f43f5e` / `#22c55e` chart strokes; the 02-frontend-desktop report already flagged "hex color literals scattered" (Low). The chart tokens `--chart-1..5` exist precisely for this and are unused in the charts. So even the good surface is leaking. **Extends 02's Low item with a concrete remediation: route every chart series through `--chart-*`.**

---

## 2. Information architecture & navigation

### 2.1 Each surface has a different IA — and none models the org/project hierarchy

The V5 identity model (per `CLAUDE.md`) is three nested layers: **Organisation → Project → (members/clients)**. **No surface's navigation reflects this.**

- **Web** (`Sidebar.tsx:22–34`): a flat 8-item list — Overview, Node Network, Models, Training, Datasets, Discover, My Requests, Settings — plus an admin section (Manage Users, All Projects). There is **no org switcher**, no org context, no project-scoped navigation. A user with two orgs has no way to see or switch tenant context. For a multi-tenant startup this is a structural gap, not cosmetic.
- **Desktop** (`Sidebar.tsx:11–16`): a different, shorter 5-item list — My Projects, Discover, My Requests, Models, Settings — using **Unicode glyph icons** (`▦ ◎ ↻ ◇ ⚙`) instead of the web's lucide icons. Same conceptual nouns ("Discover", "My Requests", "Models"), different labels, different icons, different order.
- **Mobile** (`AppNavigator.jsx`): a 3-tab bottom bar — Training / Library / Inference — a completely different mental model (it's a *client* app, not the management console). That's defensible given mobile's role, but the labels collide confusingly: the tab is named `"Training"` but titled `"Federated Learning"`, `"Library"` → `"Model Library"`, `"Inference"` → `"Model Testing"`. Tab name ≠ header name on every screen.

**Consequence:** a user moving web→desktop re-learns the navigation; "Models" means three different things (web = registry view, desktop = downloadable client models, mobile = on-device saved `.pt` files). There is no shared mental model of "where am I and what can I do here."

**Recommendation:** define **one canonical IA** in the v2 design phase — an org switcher at the top of the shell (Linear/Vercel pattern: workspace selector → projects → project detail tabs), with surface-appropriate subsets. Mobile keeps its 3-tab client model but adopts the shared labels and a "Connected to: {org/project}" context header.

**Verdict: IA is `refactor`** on web (add org context + project-scoped routes; the route table is sound), **`rebuild`** on desktop nav to converge on the canonical IA, **`refactor`** on mobile (keep structure, fix naming + context).

---

## 3. Data visualization for FL runs — the platform's reason to exist

This is the dimension the brainstorm (`00-DESIGN.md` §4, C5 row) and B3-observability both single out: **observability of the FL runs users create.** Today the viz is partial and partly decorative.

### 3.1 What exists

| Component | What it shows | Assessment |
|---|---|---|
| `FederationOrrery.tsx` | Animated SVG: clients orbiting a core on 3 rings, ring assigned by `contribution` (>0.7 / >0.4 / else), uploading clients draw a line to core, round counter HUD | **Beautiful but mostly decorative.** Ring placement encodes contribution into a hard-to-read radial position; there's no axis, no legend, no actual values. It answers "is something happening?" not "which client contributed what, and is convergence healthy?" |
| `TrainingInsightsView.tsx` | 5 stat tiles (active projects, total rounds, best acc, latest loss, best loss) + one dual-axis loss/accuracy LineChart over last 30 rounds | **The most useful viz.** But it's *portfolio-level* (flattens `Object.values(resultsMap).flat()`, line 40) — it mixes rounds across all projects into one timeline, which is statistically meaningless (round 5 of project A next to round 5 of project B). |
| `LogViewer.tsx` telemetry pane | Two 100px sparklines (loss step-after, accuracy monotone) + latest value, fed from STOMP `RoundResultDto` parsing (lines 115–129) | Good live-feel; correct that it's a terminal aesthetic deliberately off-token (documented at lines 5–12). But sparklines are tiny, axis-less, and capped at 30 points (`prev.slice(-30)`, line 119). |
| `NodeNetwork.tsx` | (separate "Node Network" page) | Not deeply audited here; another network-topology viz — overlaps conceptually with the orrery. Two different node visualizations is itself an inconsistency. |
| Mobile | A `progress%` bar (`TrainingScreen.jsx`), a `StatusRow` list, raw text logs; inference draws a hand-rolled 28×28 MNIST pixel grid with `<View>` per pixel (`InferenceScreen.jsx renderMNISTGrid`) | Functional but primitive. 784 absolutely-positioned `<View>`s per image is a perf/jank liability. Confidence bars are a **fake** proxy (`exp(-loss)` spread uniformly with a boost at the true label — `runInference`, lines ~190), which is honest in the note text but misleading as a "Per-Class Score" chart. |

### 3.2 The gaps that matter for a production FL platform

1. **Per-client contribution over time is not charted anywhere.** The orrery encodes a single instantaneous `contribution` scalar as ring position; there is no time series, no per-client loss, no "client 3 went stale at round 18", no staleness/heartbeat timeline. For DeComFL specifically (the paper this platform implements), the interesting telemetry is **per-client ZO gradient-scalar magnitude / seed participation per round** — none of it is surfaced. (Cross-ref B3-observability: the `RoundResult` telemetry pipeline is noted as under-wired server-side; the *UI to render it* is equally absent.)
2. **No convergence diagnostics.** No confidence band, no smoothing, no "rounds to target accuracy", no divergence/spike alerting. `TrainingInsightsView` even tells the user "Monitor sudden upward spikes" (line 147) as prose instead of detecting them.
3. **No communication-cost visualization** — which is *the* DeComFL selling point (communication O(K·P), independent of model dimension). A startup whose wedge is communication efficiency should chart bytes-on-wire per round vs. a FedAvg baseline. This is the most differentiated chart the product could ship and it doesn't exist.
4. **No round-progress / per-round timeline with phase breakdown** (fit vs. aggregate vs. eval wall-clock). Round progress is a single `%` integer.
5. **No cost/resource view** of the spawned FL server process (CPU/mem/duration) — directly relevant to the FL-server-per-project economics flagged in B6-scale-cost.

### 3.3 Recommendation

Build a **dedicated "Run Observability" surface** (one per project run, not portfolio-flattened) benchmarked against [Weights & Biases](https://wandb.ai/) run pages and [Grafana](https://grafana.com/) panels:
- Per-run convergence chart (loss + accuracy, with smoothing toggle and target line) — **scoped to a single project's rounds**, fixing the `.flat()` bug.
- Per-client small-multiples: one sparkline strip per client (contribution, local loss, last-seen). This is the W&B "system metrics per worker" pattern.
- A **communication-cost panel** (bytes/round, cumulative, vs. dense-FedAvg theoretical) — the DeComFL differentiator.
- Round timeline (Gantt-style phase bars) and a live round-progress ring (the orrery's core pulse is the right *feel*; give it real data).
- Keep `FederationOrrery` as the **hero "live" widget** but bind ring position to a *legible* metric and add a legend/tooltip, or demote it to a status glance and put a real per-client table beside it.

**Verdict on viz:** `FederationOrrery` → **`refactor`** (great craft, weak information design; keep as hero, make it data-honest). `TrainingInsightsView` → **`refactor`** (fix portfolio-flatten bug, add per-run scope). `LogViewer` telemetry pane → **`salvage`** (good pattern, expand). Communication-cost & per-client-timeline viz → **`rebuild`** (net-new; the highest-value missing surface). Mobile inference "confidence" chart → **`kill`** (it visualizes a fake proxy; either run real inference and show real softmax, or remove the chart and show only the honest loss).

---

## 4. Core UX flows

### 4.1 Signup / onboarding — **the biggest flow gap**

- **No onboarding exists.** `RegisterPage.tsx` → on success sets a message and `setTimeout(navigate('/login'), 2000)` (lines 79–86). The user registers, waits 2s, lands on login, logs in, and arrives at a dashboard that says **"No projects found. Create one to start federated training."** (`DashboardV2.tsx:497`). There is no welcome, no guided first-project, no org-creation step, no "invite a client" walkthrough, no sample/demo run. For a product whose core loop (spin up server → enroll clients → watch rounds) is genuinely unfamiliar to most users, a zero-onboarding cold start is a conversion killer.
- **Org creation is invisible in the UI.** The V5 model requires every project to belong to an org (`projects.org_id NOT NULL`), and the backend bootstraps a Platform org — but there is **no org-creation or org-switch UI** on any surface (confirmed: no org-related strings in `DashboardV2.tsx`, no org item in either `Sidebar`). New non-bootstrap users have no modeled path to create their tenant.
- **Client enrollment is a raw string field.** Mobile `TrainingScreen.jsx` asks the user to type a gRPC address (`localhost:50063`) and a `client_id` into bare `TextInput`s. The desktop has a `HardwareSelector` and dataset-path picker (better), but there's no unified "join this project" flow with a project code / QR / deep link. Enrollment is the make-or-break moment for an FL platform and it's currently a developer-grade form.

**Recommendation:** a v2 first-run flow — (1) create org, (2) create first project from a template, (3) "add a client" with a copyable join command + QR for mobile, (4) a one-click demo run on bundled MNIST so the user sees the orrery move within 60s. This is the Vercel/Linear "time-to-first-value" discipline.

**Verdict: onboarding = `rebuild` (net-new).** Auth pages themselves (`LoginPage`, `RegisterPage`) are clean and on-token → **`salvage`**.

### 4.2 Project creation, run monitoring, results

- **Create project** (`CreateProjectModal.tsx`): solid — typed model/optimizer option maps, resets on close. But `modelOptions` is hardcoded client-side (`CNN`/`Transformer` lists) and will drift from the framework's actual capabilities; should be backend-driven. 02-frontend-desktop already flagged the `any`-typed `createProject` payload (H3) — that's the type half; this is the design half (config UI should be schema-driven).
- **Run monitoring** (`LogViewer.tsx`): the live terminal + telemetry split is the strongest UX on the platform. Documented intentional off-token terminal palette (lines 5–12) is the *correct* call. Connection state is well-communicated (animated ping dot, "Live Streaming / Paused / Connecting…"). Keep it. (02-frontend-desktop's H4 history/stream race and H5 StrictMode leak still apply — those are correctness, not design.)
- **Results** (`ResultsModal`): present but thin; results are also `<div>` overlay modals (a11y issue, §5).

**Verdict:** create/monitor/results flows → **`refactor`** (good bones; make config schema-driven, fix modal a11y).

### 4.3 Modal pattern is a shared liability

All modals across web are `<div>` overlays with manual `if (!isOpen) return null` (e.g. `CreateProjectModal.tsx:58`, `LogViewer.tsx:158`), no `<dialog>`, no focus trap, no `aria-modal`, inconsistent Esc-close. **02-frontend-desktop M5 already flagged this** — I extend it: this is the single most reused interactive pattern in the app, so fixing it via a shared Radix `Dialog` primitive (focus trap + Esc + `aria-modal` for free) is both the a11y fix *and* the proof-of-value for adopting the component library (§6).

---

## 5. Communication & information design

### 5.1 Empty / loading / error states are inconsistent and under-designed

- **Empty:** web dashboard empty state is one line of text (`DashboardV2.tsx:497`); mobile model library has a slightly better two-line empty with a "train a model" hint (`ModelLibraryScreen.jsx`). No illustrations, no primary CTA, no "here's what to do next." Empty states are onboarding surfaces in disguise and are being wasted.
- **Loading:** a grab-bag — web `TrainingInsightsView` shows plain text "Loading insights..." (line 82); there's a bespoke `DiskLoader` component with its own CSS (02-frontend-desktop Low: "DiskLoader ships its own CSS while everything else is Tailwind"); mobile uses `ActivityIndicator`. No skeleton screens anywhere. Skeletons are the W&B/Linear standard and materially improve perceived performance.
- **Error:** web uses ad-hoc inline error divs with `color-mix(in srgb, #ef4444 …)` hardcoded (`TrainingInsightsView.tsx:76`); mobile uses native `Alert.alert(...)` modal popups extensively (`TrainingScreen`, `InferenceScreen`, `ModelLibraryScreen`) — jarring, non-dismissible-in-context, and inconsistent with web toasts. `AuthContext` swallowing `/auth/me` errors (02's H6) means network failure is communicated as "logged out" — a *communication* bug, not just a logic one.

### 5.2 Toasts — recently added, web-only, not yet a system

`ToastContext.tsx` is clean (typed levels, per-level auto-dismiss timings, `useToast` guard). **But:** it's web-only (desktop and mobile have no equivalent), and 02-frontend-desktop's H9 notes the *NotificationContext* (the server-push bell) doesn't clear on logout. The two feedback channels (ephemeral toasts vs. persistent bell) are a good conceptual split (documented in `ToastContext.tsx`), but they're web-only and the distinction isn't mirrored anywhere else. **Verdict: `salvage` and promote to shared package** (toast + notification primitives belong in the cross-surface UI lib).

### 5.3 Microcopy

Mostly competent on web ("Welcome back to the platform", "Increase aggregate accuracy while holding loss volatility low"). But: jargon leaks to end users (mobile "Start ZO-FL (FedAvg + ZO)", "DeComFL: Decomposed FL with ZO gradient scalars. Byzantine-robust" — `TrainingScreen.jsx TRAINING_MODES`); two of three modes are `disabled: true` with no explanation of *why* or *when* they'll unlock; the landing page claims **"V2 Platform Live"** (`LandingPage.tsx:39`) which is aspirational/false for a POC and a trust risk. **Recommendation:** a microcopy pass with a glossary, "coming soon" affordances on disabled modes, and honest status badges.

**Verdict on communication design:** empty/loading/error → **`rebuild`** as a shared set of primitives (EmptyState, Skeleton, ErrorState, Toast) — this is exactly what a design system is for. Microcopy → **`refactor`**.

---

## 6. Accessibility (WCAG 2.2 AA)

Extends 02-frontend-desktop M4 ("no `eslint-plugin-jsx-a11y`") and M5 (modals). Findings specific to design/UX:

- **No focus management in modals** (no trap, no return-focus, no `aria-modal`) — fails WCAG 2.4.3 Focus Order / 2.1.2 No Keyboard Trap-avoidance. Web-wide.
- **Color-only status encoding.** FL state is communicated purely by hue: orrery client color (`InferenceScreen`/orrery: blue=training, orange=uploading, gray=offline) and the `StatusRow color` props (`#28a745`/`#dc3545`) carry meaning with no text/icon redundancy — fails 1.4.1 Use of Color. The LogViewer's connection dot is the *good* pattern (dot **plus** "Live Streaming" text); generalize it.
- **Contrast unverified.** `--text-secondary` light mode is `oklch(0.36 0.014 40)` on `oklch(0.987 …)` (likely passes), but the orrery node labels at `fontSize: 9` in `--text-secondary` over an animated SVG (`FederationOrrery.tsx:179–189`) almost certainly fail 1.4.3 (small text, low-contrast, moving background). Mobile's `#6c757d` placeholder/secondary text on `#f8f9fa` is borderline.
- **Emoji as the only icon** (mobile tabs `🤖📚🔍`) — screen readers announce inconsistent emoji names; not a reliable label. Tab `name` doubles as both visible icon and a11y label.
- **Motion.** Orrery (`requestAnimationFrame` spin, observed in prior session note), pulse-ring animations, `framer-motion` page transitions — **no `prefers-reduced-motion` handling anywhere** (confirmed: no such media query in `theme.css` or components). Fails 2.3.3 Animation from Interactions and is a vestibular-safety issue.
- **No semantic landmarks / skip-link** audited on the SPA shell.

**Verdict: accessibility = `rebuild` posture** — not because the web is hostile (the `:focus-visible` ring is a genuine win), but because a11y must be *systemic*, and there's no system. Adopting Radix/Base-UI primitives (which ship focus-trap, `aria-*`, and roving-tabindex correctly) plus `eslint-plugin-jsx-a11y` + `@axe-core/react` (both already recommended in 02) closes most of this *by construction*.

---

## 7. v2 recommendation — a shared design system across web + desktop + mobile

### 7.1 The stack

| Layer | Recommendation | Why |
|---|---|---|
| **Token source of truth** | A tiny published package (`@fedlearn/tokens`) holding the OKLCH tokens from `frontend/src/styles/theme.css`, emitted as **CSS vars** (web/desktop) **and** a JS object (NativeWind theme). One file, three consumers. | Kills the three-palette problem at the root. The desktop's two-conflicting-`:root` situation becomes impossible. |
| **Web + Desktop components** | **shadcn/ui** (copy-in components on **Radix** primitives, optionally **Base UI** as of the Feb 2026 shadcn release) on **Tailwind v4**. | Industry-default for new React apps in 2026 ([Vercel Academy](https://vercel.com/academy/shadcn-ui), [shadcn/ui](https://ui.shadcn.com/)); you *own* the source (no runtime dep lock-in); Radix gives WCAG-correct dialogs/menus/tooltips for free — directly fixing §5/§6. Web already uses `class-variance-authority` + `tailwind-merge` + `clsx` (`package.json`), which **is the shadcn substrate** — so adoption is incremental, not a rewrite. |
| **Mobile components** | **[react-native-reusables](https://reactnativereusables.com/)** (shadcn-for-RN on **NativeWind v4**). | Same token model, same component vocabulary, same accessibility posture, on mobile. Covers ~36 of 51 shadcn components. Lets the mobile app drop its inline `StyleSheet` hex soup and consume `@fedlearn/tokens`. |
| **Desktop styling migration** | Replace the 1,296-line hand CSS with Tailwind v4 + shadcn (same as web). | Eliminates the duplicate token blocks and the web/desktop component drift; one team can now ship a component once and reuse it on both. (Note: B5-desktop-strategy is separately evaluating Electron-vs-Tauri-vs-native; **whatever shell wins, if it renders web tech it should consume the same shadcn+token layer.** If B5 recommends native/thin-shell-over-C++-core, the *token package still applies* to whatever chrome remains.) |
| **Charts** | Standardize on one library across web+desktop. `recharts` (already in web) is fine for run charts; route all series through `--chart-*` tokens. For dense/real-time FL telemetry consider `visx`/`uPlot` (uPlot for the live high-frequency loss stream — far lighter than recharts for streaming). | Today: recharts on web, hand-rolled SVG mobile, none on desktop. One charting story. |
| **Icons** | One icon set. Web already uses **lucide-react**; ship `lucide-react-native` on mobile (drop emoji), and lucide on desktop (drop Unicode glyphs). 02-frontend-desktop Low: web ships *both* `react-icons` and `lucide-react` — pick lucide, drop react-icons. | Visual consistency + a11y (real `aria-label`s). |
| **Cross-surface primitives to extract first** | `Dialog/Modal`, `Toast`, `EmptyState`, `Skeleton`, `ErrorState`, `StatusBadge` (text+icon+color), `MetricTile`, `ConvergenceChart`, `ClientList`. | These are the exact pain points in §3–§6; building them once in the shared lib is the proof-of-value and the fastest a11y/consistency win. |

### 7.2 What this is NOT

- It is **not** a heavyweight component framework (MUI/Chakra/Ant) — those would fight the existing Tailwind v4 + CVA setup and add runtime weight (02-frontend-desktop M2 already flags bundle bloat from eager `framer-motion`+`recharts`). shadcn is copy-in source, zero runtime lock-in.
- It does **not** require touching the FL invariants (no `flwr`, cookie auth, Flyway, gRPC contract, chunking/heartbeat). This is pure presentation layer.

### 7.3 Sequencing (cost-aware for a startup)

1. **Extract `@fedlearn/tokens`** from the web `theme.css` (½ day). Immediately point desktop at it; delete the duplicate `:root` blocks. This alone unifies the accent color across web+desktop.
2. **Adopt shadcn on web** (it's already substrate-compatible) — migrate `Dialog`, `Toast`, `Button`, form fields first. Fixes modal a11y (§4.3, §6) and the `<dialog>` gap in one move.
3. **Stand up react-native-reusables on mobile** + consume tokens; replace inline hex and emoji icons.
4. **Migrate desktop renderer CSS → Tailwind+shadcn** (largest single chunk; gated on B5's shell decision).
5. **Build the Run Observability surface** (§3.3) with the shared chart primitives — the differentiated, highest-value UI.

---

## 8. Brand & visual identity direction

Three product names (FedLearn / FedLearn Desktop / FedMob) and three palettes signal "research prototype," not "startup." A startup needs **one name, one mark, one palette, one voice.**

- **Keep the web "Federation Pulse" orange (`oklch(0.62 0.24 48)`) as the brand accent.** It's distinctive (most ML/dev tools are blue/purple — W&B yellow-and-blue, Grafana orange-ish, Linear purple, Vercel black/white), warm, and already the most-developed surface. Orange + near-black + IBM Plex is a credible, ownable identity in a sea of indigo SaaS.
- **Retire "FedMob" and "FedLearn Desktop"** as distinct brands; they're one product (FedLearn) with web/desktop/mobile clients. Use a consistent wordmark + the `Network`/pulse mark already on `LandingPage.tsx:16`.
- **Voice:** the web copy is on-brand ("Unlock insights from distributed data. Securely."). Codify it: precise, systems-engineering-confident, no hype. **Drop the false "V2 Platform Live" badge** until it's true.
- **The orrery is a brand asset.** A live, orbiting federation visual is a genuinely ownable hero motif (cf. how Vercel's deploy viz and W&B's run sparklines became identity). Make it data-honest (§3) and it doubles as marketing and product.
- Benchmark posture: **shell** = Linear/Vercel (calm, dense, keyboard-fast); **run observability** = W&B + Grafana (charts-first, comparison-native); **dev-tool credibility** = Hugging Face (model/dataset cards). Cite these as the north stars in the v2 brainstorm.

---

## 9. Decision table

| Module / subsystem | Surface | Verdict | One-line rationale |
|---|---|---|---|
| Cross-surface visual system (tokens) | all | **rebuild** | No shared system exists; three palettes (incl. desktop's two-conflicting-`:root`). Seed from web `theme.css`. |
| Web `theme.css` token file | web | **salvage** | Genuine OKLCH design-system work; promote to shared source of truth. |
| Web shell / IA (Sidebar, routes) | web | **refactor** | Sound routes; add org switcher + project-scoped nav for multi-tenancy. |
| Desktop renderer styling (`styles.css`) | desktop | **rebuild** | 1,296-line hand CSS with duplicate/conflicting token blocks; migrate to shared Tailwind+shadcn. |
| Desktop nav (glyph icons, divergent labels) | desktop | **rebuild** | Converge on canonical IA + lucide icons. |
| Mobile screens styling (inline hex, emoji tabs) | mobile | **rebuild** | Unthemed Bootstrap-era boilerplate; adopt react-native-reusables + shared tokens. |
| Mobile IA / screen structure | mobile | **refactor** | 3-tab client model is fine; fix tab-vs-header naming + add project context. |
| `FederationOrrery` | web | **refactor** | Great craft, weak info-design; keep as hero, bind to legible metric, add legend/reduced-motion. |
| `TrainingInsightsView` | web | **refactor** | Useful chart, but portfolio-flatten bug; scope to per-run, add diagnostics. |
| `LogViewer` (terminal + telemetry) | web | **salvage** | Strongest UX on the platform; deliberate off-token terminal is correct; expand telemetry. |
| Per-client / communication-cost viz | all | **rebuild** | Net-new; the highest-value missing surface and the DeComFL differentiator. |
| Mobile inference "confidence" chart | mobile | **kill** | Visualizes a fake `exp(-loss)` proxy as per-class score; remove or replace with real softmax. |
| Onboarding / first-run / org creation | all | **rebuild** | Does not exist; cold-start dead-ends at "No projects found." Conversion-critical. |
| Auth pages (Login/Register) | web | **salvage** | Clean, on-token; only fix post-register redirect → onboarding. |
| Create/Monitor/Results flows | web | **refactor** | Good bones; make config schema-driven, fix modal a11y. |
| Modal pattern (`<div>` overlays) | web | **rebuild** | No `<dialog>`/focus-trap/`aria-modal`; replace with Radix Dialog primitive (a11y + consistency). |
| Empty / loading / error states | all | **rebuild** | Inconsistent, no skeletons, mobile uses `Alert.alert`; build shared primitives. |
| `ToastContext` | web | **salvage** | Clean; promote to shared package (desktop+mobile have none). |
| Microcopy / glossary | all | **refactor** | Competent on web; jargon leaks on mobile, false "V2 Live" badge, no disabled-mode affordances. |
| Accessibility posture | all | **rebuild** | No focus mgmt, color-only status, no `prefers-reduced-motion`, emoji-only icons; fix systemically via Radix + a11y lint. |
| Component library / design system | all | **rebuild** | None exists; adopt shadcn (web+desktop) + react-native-reusables (mobile) on one token package. |

---

## 10. Prioritized recommendations

**P0 — unblock the rest (days):**
1. Extract `@fedlearn/tokens` from web `theme.css`; delete desktop's duplicate `:root` block; unify accent → one orange everywhere.
2. Adopt shadcn on web (substrate already present); migrate Dialog + Toast first → fixes modal a11y (§4.3/§6) and the `<dialog>` gap.

**P1 — the product's reason to exist (weeks):**
3. Build the **per-run Observability surface** (per-run convergence with diagnostics, per-client small-multiples, communication-cost panel) — benchmarked to W&B/Grafana. Fix the `TrainingInsightsView` portfolio-flatten bug.
4. Make `FederationOrrery` data-honest (legend, legible metric, reduced-motion) and keep it as the hero.

**P2 — conversion + reach (weeks):**
5. Build first-run onboarding + org-creation/switch UI + client-enrollment flow (join code/QR/deep-link) + one-click demo run.
6. Stand up react-native-reusables on mobile; consume shared tokens; drop inline hex + emoji icons + the fake confidence chart.

**P3 — systemic quality:**
7. Migrate desktop renderer to Tailwind+shadcn (gated on B5 shell decision).
8. Shared empty/loading(skeleton)/error/status primitives; `eslint-plugin-jsx-a11y` + `@axe-core/react` (per 02); microcopy pass + glossary; drop "V2 Platform Live".

---

## 11. Uncertainty & cross-references

- **B5-desktop-strategy may recommend a non-Electron shell** (Tauri / native / thin-shell-over-C++-core). My desktop-CSS-migration recommendation is contingent: *if* the desktop renders web tech, share the shadcn+token layer; *regardless*, share the token package for any chrome. Flagged, not assumed.
- **B3-observability owns the server-side telemetry pipeline** (the under-wired `RoundResult`). §3's viz recommendations assume that data becomes available; the *UI* is my scope, the *pipeline* is B3's. The two must be planned together — there's no point building per-client charts without per-client telemetry emission.
- **Per-client DeComFL telemetry** (ZO gradient-scalar magnitude, seed participation) as a chartable signal is my inference about what *would* differentiate the product; whether the framework currently emits it is a B1/B3 question. Flagged as a recommendation, not a claim about current data.
- **`phone_screen.png` verified** (1080×2316 PNG at branch root, not under `mobile_client/`) — confirms the mobile visual state described in §1.1.
- I did not deeply audit `NodeNetwork.tsx`, `DatasetsView.tsx`, `ModelsView.tsx`, `SettingsView.tsx`, or the desktop view components line-by-line; the cross-surface conclusions hold from the token/component evidence, but a per-component design pass is a v2-brainstorm task.

---

### Sources (market/benchmark claims)
- shadcn/ui — <https://ui.shadcn.com/> ; Vercel Academy — <https://vercel.com/academy/shadcn-ui> ; 2026 stack guidance — <https://jishulabs.com/blog/shadcn-ui-component-library-guide-2026>
- react-native-reusables — <https://reactnativereusables.com/>
- Weights & Biases (run-observability benchmark) — <https://wandb.ai/>
- Grafana (telemetry-panel benchmark) — <https://grafana.com/>
- WCAG 2.2 — <https://www.w3.org/TR/WCAG22/>
</content>
</invoke>
