# How steep are real streets? Standards, limits, records and distributions

Every number below comes from a source that was opened. A few come from search snippets only, and are marked **n.v.** Translations and derived numbers are labelled as mine. Grade means rise/run × 100%; 1:12 ≈ 8.33%.

All URLs were opened on 2026-11-25. Sources were saved locally for page lookup and are not committed.

---

## 1. Road-design maximum grades

### 1.1 United States (AASHTO-based), via the New York State DOT Highway Design Manual

AASHTO's *A Policy on Geometric Design of Highways and Streets* (the Green Book) is not free. NYSDOT HDM Chapter 2 (Revision 104) adopts its criteria, stating "Classifications are based on AASHTO's A Policy on Geometric Design of Highways and Streets, 2018." (Exhibit 2-1 note 2, p. 2-25). Whether every NYSDOT grade cell equals the Green Book cell is **n.v.**

- URL: https://www.dot.ny.gov/divisions/engineering/design/dqab/hdm/hdm-repository/chapt_02.pdf

| Street class (urban / suburban / urban core) | Design speed | Max grade, level / rolling / mountainous terrain | Source |
|---|---|---|---|
| **Local streets, residential** | 20–30 mph | **15%** | §2.7.4.3 G (p. 2-75); Exhibit 2-8 (p. 2-76) |
| **Local streets, commercial / industrial** | 20–30 mph | **8%** (desirable) | same |
| Collectors (Exhibit 2-6, p. 2-64) | 25 mph | 9 / 12 / 13% | |
| | 30 mph | 9 / 11 / 12% | |
| | 35–40 mph | 9 / 10 / 12% | |
| | 45 mph | 8 / 9 / 11% | |
| | 50 mph | 7 / 8 / 10% | |
| Arterials, non-NHS (Exhibit 2-4, p. 2-51) | 25 mph | 7 / 10 / 12% | |
| | 30 mph | 7 / 9 / 11% | |
| | 35–40 mph | 7 / 8 / 10% | |
| | 45–50 mph | 6 / 7 / 9% | |
| | 55 mph | 5 / 6 / 8% | |

Verbatim:
- §2.7.4.3 G (p. 2-75): "Grades for local streets = 15% maximum. An 8% maximum or flatter grade is desirable in commercial and industrial areas."
- Exhibit 2-6 note 5 (p. 2-64): "Maximum grades of short length (< 490 ft.) and on one-way downgrades may be 2% steeper."

### 1.2 Japan: Road Structure Ordinance (道路構造令), Cabinet Order No. 320 of 1970, Article 20

- URL: https://laws.e-gov.go.jp/api/1/lawdata/345CO0000000320 (e-Gov law XML)
- Article 3 defines **Type 4 (第四種)** as non-expressway roads in **urban areas (都市部)**.
- Article 20, verbatim (ruby readings removed): "車道の縦断勾配は、道路の区分及び道路の設計速度に応じ、次の表の縦断勾配の欄の上欄に掲げる値以下とするものとする。ただし、地形の状況その他の特別の理由によりやむを得ない場合においては、同表の縦断勾配の欄の下欄に掲げる値以下とすることができる。"
  - My translation: the longitudinal grade of the carriageway shall not exceed the upper value for the road class and design speed. Where terrain or other special reasons make it unavoidable, it may be up to the lower value.

| Class | Design speed (km/h) | Standard max | Exceptional max (terrain etc.) |
|---|---|---|---|
| **Type 4 (urban), ordinary roads** | 60 | 5% | 7% |
| | 50 | 6% | 8% |
| | 40 | 7% | 9% |
| | 30 | 8% | 10% |
| | 20 | **9%** | **11%** |
| Type 4, small-vehicle roads (小型道路; one value only) | 60 / 50 / 40 / 30 / 20 | 8 / 9 / 10 / 11 / **12%** | n/a |
| Types 1–3 (rural and expressways), ordinary roads | 120 / 100 / 80 / 60 / 50 / 40 / 30 / 20 | 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9% | 5 / 6 / 7 / 8 / 9 / 10 / 11 / 12% |

### 1.3 Spain: Norma 3.1-IC Trazado (Orden FOM/273/2016), BOE no. 55, 4 Mar 2016

- URL: https://www.boe.es/boe/dias/2016/03/04/pdfs/BOE-A-2016-2217.pdf
- This is the state *road* instruction (interurban), not an urban-street standard.

**Tabla 5.2** (p. 17715), conventional and multilane roads:

| Design speed (km/h) | Maximum | Exceptional |
|---|---|---|
| 100 | 4% | 5% |
| 90–80 | 5% | 7% |
| 70–60 | 6% | 8% |
| 50–40 | **7%** | **10%** |

Ice rule, verbatim (p. 17716): "En los tramos de posible existencia de hielo en la calzada se procurará que la inclinación de la línea de máxima pendiente en cualquier punto de la plataforma no sea superior al diez por ciento ( 10 %)."
- My translation: where ice is possible, the line of steepest slope anywhere on the platform should not exceed 10%.

### 1.4 United Kingdom

- **DMRB CD 109 Rev 1** (Highway link design, Mar 2020). Copy opened: https://www.milton-keynes.gov.uk/sites/default/files/2022-04/CD13j%20Highway%20link%20design.PDF.
  - It covers trunk roads only: "shall be used for both new and improved motorway and all-purpose trunk roads" (§1.1).
  - **Table 5.1** (p. 23): desirable maximum / permitted relaxation are motorways **3% / 4%**, all-purpose dual carriageways **4% / 8%**, all-purpose single carriageways **6% / 8%**.
- **Manual for Streets** (DfT 2007): https://assets.publishing.service.gov.uk/media/5a7e0035ed915d74e6223743/pdfmanforstreets.pdf. It sets **no carriageway maximum** in the passages found.
  - Pedestrians, §6.3.27 (p. 69): "Longitudinal gradients should ideally be no more than 5%, although topography or other circumstances may make this difficult to achieve".
  - Cyclists, §6.4.11 (p. 72): "The maximum gradients should generally be no more than 3%, or 5% maximum over a distance of 100 m or less, and 7% maximum over a distance of 30 m or less. However, topography may dictate the gradients, particularly if the route is in the carriageway."
  - An example street specification (p. 34) lists "Gradients (footways) 1:15 Maximum, footway to follow carriageway".
- **Local highway-authority residential guides** (search snippets only, **n.v.**; the PDFs refused download):
  - Nottinghamshire: residential carriageway max 1:20 (5%);
  - Leicestershire: 1:20;
  - Herefordshire: 1:20 desirable, up to 1:10 considered.

### 1.5 Summary of design maxima for urban streets

- **Through streets** (arterials and collectors): about **5–9%** on level/standard terrain, up to **10–13%** in mountainous terrain or short sections.
- **Local streets:** **9–12%** in Japan, **15%** residential in the US.
- These are design conventions, set for operating speed, safety, drainage, winter conditions and trucks. They are not physical limits.

---

## 2. Fire access and accessibility reference points

- **Fire apparatus access roads, International Fire Code Appendix D §D103.2.**
  - Model text as adopted in the **2021 Seattle Fire Code**, where it is renumbered D103.1 with "((D103.2))" marking the IFC number: https://seattle.gov/documents/Departments/SDCI/Codes/SeattleFireCode/2021SFCAppendixD.pdf (p. D-2).
  - Verbatim: "Fire apparatus access roads shall not exceed 10 percent in grade. Exception: Grades steeper than 10 percent as approved by the fire code official."
  - Local variants from search snippets, **n.v.**: Hillsboro OR allows up to 15% with sprinklers; Mercer Island WA caps at 20%.
- **US accessibility, 2010 ADA Standards** (US Access Board): https://www.access-board.gov/ada/
  - 403.3: "The running slope of walking surfaces shall not be steeper than 1:20. The cross slope of walking surfaces shall not be steeper than 1:48." That is **5%**, and 2.08% cross slope.
  - 405.2: "Ramp runs shall have a running slope not steeper than 1:12." That is **8.33%**.
- **Passenger cars, UN Regulation No. 13-H** (braking), Annex 3 §2.3.1.
  - Read in GS 4013H:2019, a Ghana Standard stating it is an identical adoption of UNECE R13-H: https://gsa.gov.gh/wp-content/uploads/2019/07/GS-4013H-2019.pdf
  - Verbatim: "The parking braking system must be capable of holding the laden vehicle stationary on a 20 per cent up or down gradient."
  - §2.3.2: 12% for a car with a trailer.
- **Heavy goods vehicles, EU Regulation 1230/2012**, Annex I Part C §4.1, for N2/N3 vehicles designed to tow, when laden: they "shall be capable of starting five times within five minutes at an up-hill gradient of at least 12 %."
  - Read via WebFetch of https://www.legislation.gov.uk/eur/2012/1230/annex/I/part/C/adopted?view=plain. A model extracted the text, so exact wording is **n.v.**; EUR-Lex returned a bot challenge.
- **Transit buses, Georgia DOAS 2022 Technical and Performance Specifications for Transit Buses** ("BASED ON APTA SPECIFICATION").
  - URL: https://doas.ga.gov/sites/default/files/assets/State%20Purchasing/2022%20Mass%20Transit%20Vehicles%20Specifications/Cat%202%20Transit%20Bus%20Specification.pdf
  - Verbatim: "The propulsion system shall enable the bus to start from a full stop and achieve and maintain a speed of 40 mph on a 2.5 percent ascending grade continuous, 10 mph on a 10 percent ascending grade continuous, and 7 mph on a 16 percent ascending grade for a minimum of 60 seconds."
  - Also: door interlocks must be "capable of holding a fully loaded bus on a 6 percent grade".
  - The APTA White Book itself returned 403.

---

## 3. The steepest public streets

| Street | Grade | How measured | Source (access) |
|---|---|---|---|
| **Baldwin Street, Dunedin (NZ)** | **34.8%** | centreline; current Guinness record holder | Guinness World Records news, 8 Apr 2020: https://www.guinnessworldrecords.com/news/2020/4/baldwin-street-in-new-zealand-reinstated-as-the-worlds-steepest-street-614287 (FT) |
| **Ffordd Pen Llech, Harlech (Wales)** | **37.45%** (1:2.67) in 2019, re-measured at **28.6%** on the centreline in 2020 | 2019: steepest 10 m section (inside of a curve); 2020: centreline | Guinness 2019 news: https://www.guinnessworldrecords.com/news/2019/7/welsh-town-claims-title-for-worlds-steepest-street-582452 (FT), plus the 2020 article |
| **Canton Avenue, Pittsburgh** | **37%** | city engineering records; the 37% section is reported as about 6.5 m long (Wikipedia, secondary) | Pittsburgh Post-Gazette, 30 Jan 2005, archived: https://web.archive.org/web/20070817112347/http://www.post-gazette.com/pg/05030/448976.stm (FT). Length from https://en.wikipedia.org/wiki/Baldwin_Street |
| Filbert St and 22nd St, San Francisco | 31–31.5% for 60–70 m | city map elevations | Wikipedia (secondary, primary not opened): https://en.wikipedia.org/wiki/Baldwin_Street, https://en.wikipedia.org/wiki/List_of_steepest_roads_and_streets |
| Eldred St (LA) 33%; 28th St, San Pedro 33.3%; Baxter St 32% | as stated | LA Times 2003, cited by Wikipedia | Wikipedia list (secondary) |
| Bradford St, San Francisco | 39–40% over a 9 m section | n.v. | Wikipedia (secondary) |

Verbatim:
- Guinness 2019: "The record is measured based on the steepest (highest gradient) section over a 10 m distance."
- Guinness 2020: "The new results confirmed Baldwin Street has the steeper gradient of 34.8%, compared to Ffordd Pen Llech's gradient of 28.6%."
- Guinness 2020: "the best practice for the gradient is to take the measurement from the centreline of the street."
- Post-Gazette 2005: "It's the steepest street in this hilly town and, probably, the region, with a grade of 37 percent -- that is, rising 37 feet per 100 feet of run. So confirm records from the city Department of Engineering and Construction."
- Guinness 2019 (resident's account): "my car slid straight down with all four tires locked"

**Takeaway.** The steepest public streets top out at about **31–37% over short (≤ 10–70 m) sections**. Any street steeper than about 35% sustained over more than a few tens of metres has, as far as the sources found show, no real-world precedent.

---

## 4. What a vehicle can physically climb

### 4.1 Traction (friction) limit: my derivation, not from a cited text

Consider a rigid two-axle vehicle standing on grade θ, ignoring rolling and air resistance:
- wheelbase L, centre-of-gravity height h;
- a and b are the distances from the CG to the front and rear axles (a + b = L);
- peak tyre–road friction μ.

Axle loads: N_f = W(b cosθ − h sinθ)/L and N_r = W(a cosθ + h sinθ)/L. The climb is possible while W sinθ ≤ μ N_driven:

- **4WD/AWD:** tan θ ≤ μ.
- **Front-wheel drive:** tan θ ≤ μ (b/L) / (1 + μ h/L). Uphill load transfer *unloads* the driven axle.
- **Rear-wheel drive:** tan θ ≤ μ (a/L) / (1 − μ h/L).

Friction values: https://hpwizard.com/tire-friction-coefficient.html, a non-official page reproducing a table it attributes to J.Y. Wong, *Theory of Ground Vehicles*, 2nd ed., 1993, p. 26 (Wong not opened). Peak values:
- dry asphalt/concrete **0.80–0.90**;
- wet asphalt **0.50–0.70**;
- hard-packed snow **0.20**;
- ice **0.10**.

Illustrative results, with **assumed** geometry h/L = 0.2, front-heavy FWD b/L = 0.6 and RWD a/L = 0.5 (my numbers):

| Surface (μ) | AWD | FWD | RWD |
|---|---|---|---|
| dry (0.8) | 80% | 41% | 48% |
| wet (0.5) | 50% | 27% | 28% |
| snow (0.2) | 20% | 12% | 10% |
| ice (0.1) | 10% | 6% | 5% |

Rolling resistance lowers each value by about 1 percentage point. Engine torque and gearing are not the binding limit for cars at walking speed. That is my statement and **n.v.** against a source.

### 4.2 Consistency with observations

- Cars do drive the 31–35% streets of §3, which matches the dry 2WD bound of about 40–48%.
- The Harlech resident's car sliding "with all four tires locked" matches the friction limit when braking downhill.
- Canton Avenue is not driven or ploughed in ice and snow (Post-Gazette), which matches the ice bound of about 5–6%.

### 4.3 Certified minimum capabilities, not maxima

- Cars must *hold* on a 20% grade with the parking brake (UN R13-H).
- Towing trucks must *restart* on 12% (EU 1230/2012).
- Procured transit buses must keep 10 mph on 10% and 7 mph on 16% for 60 s (GA/APTA-based specification).

---

## 5. Empirical distributions of real street grades

Sources: Boeing 2022 and Dataverse v3.1. Details and caveats are in [boeing2022_global.md](boeing2022_global.md).

- **Grade definition there:** node-to-node per simplified edge, (Δ elevation)/(edge length), on ASTER (about 30 m) or SRTM (about 90 m). Values are smoothed and segment-averaged relative to 10 m sub-spans.
- **Paper, Table 2:** population-weighted mean of urban-area grade_mean is **2.0–4.5%** by world subregion. Highest are Australia/NZ 4.5% and Southern Europe 4.1%; lowest are South-Central Asia 2.0% and Eastern Asia 2.1%.
- **Open data v3.1, 10,351 urban areas (computed by us):**
  - median urban-area mean |grade| **2.7%**, p95 **5.8%**, maximum **18.8%** (Taxco);
  - only **0.9%** of urban areas average above 8%, and **0.2%** above 10%.
- **Share of street length steeper than 15%, per edge (computed by us):**
  - Manizales 18.3%, Funchal 17.1%, Medellín 12.7%, La Paz 8.2%, Valparaíso 7.1%, Quito 6.9%, Caracas 6.8%;
  - Wellington 4.7%, Genoa 4.4%, Lisbon 3.9%, Dunedin 3.8%, Busan 3.5%, Pittsburgh 2.2%, Seoul 2.0%;
  - San Francisco urban centre 0.9%; flat Chicago 0.2% and Sacramento 0.1%, which is the noise floor.
- **Share steeper than 10%:** from 37% (Funchal) and 34% (Manizales) down to 5.6% (Seoul) and 0.2–0.4% (flat cities).
- **Share steeper than 20%:** 9.6% at most (Manizales); 1–2% in Lisbon, Wellington and Genoa.
- **Two biases, both toward lower values than a 10 m measurement:** within-edge maxima are invisible, and DEMs smooth.

No published study was found that reports the **ratio of street grade to underlying terrain slope** (contour following) for real cities. See [_index_measurement.md](_index_measurement.md) §3.

---

## 6. Proposed grade thresholds for UrbanMorph

Grades should be measured on **10 m sub-spans** of ground (non-bridge, non-tunnel) edges. That matches the Guinness 10 m convention and our TAS pieces. "Sustained" means a 50 m running mean along an edge; the 50 m figure is my proposal, a common-sense floor below the 150 m short-length allowance in NYSDOT note 5.

| Level | Threshold | Meaning | Sources | Hard limit or convention? |
|---|---|---|---|---|
| **Comfortable / accessible** | **≤ 5%** | walkable without ramps; bus-friendly | ADA 403.3 (1:20); MfS §6.3.27; UK local residential guides (1:20, n.v.); Japan Type 4 at 60 km/h (5%) | **Convention**, legally binding for US pedestrian routes, not for vehicles |
| **Through-street / emergency standard** | **≤ 10%** | max for arterials, collectors and fire access; continuous bus operation | IFC D103.2 (10%); Spain 3.1-IC exceptional at 40–50 km/h (10%) and ice rule (10%); NYSDOT collectors, mountainous (10–13%); GA/APTA bus 10 mph on 10% | **Convention** (codes allow local exceptions) |
| **Local-street design maximum** | **≤ 15%** | steepest grade a modern code permits for residential streets | NYSDOT/AASHTO local residential (15%); Japan Type 4 small roads at 20 km/h (12%), ordinary exceptional (11%); bus 16% for 60 s only | **Convention** |
| **Physical / precedent limit** | **> 30% implausible; > 35% unprecedented** | beyond almost all public streets; 2WD traction margin gone on wet pavement (about 27%) | Guinness: Baldwin 34.8% (centreline), Ffordd Pen Llech 28.6%; Canton 37% over about 6.5 m; traction derivation §4.1 | **Physics-based but condition-dependent**: dry about 40–48% for 2WD, wet about 27%, ice about 5% (derived) |

**Suggested use:**
1. Keep TAS's G at 15%; it is defensible as the AASHTO/NYSDOT local maximum.
2. Report the length shares above 5%, 10% and 15% against the **same-pipeline** real tiles.
3. Add a hard-violation count: length above 30% on 10 m sub-spans, expected to be about 0 for real cities.
4. For **grade-limited routing success**, use g_max = **10%** (all vehicles, fire and buses) as primary and **15%** (cars on local streets) as secondary. Report 20% only as a sensitivity bound (car parking-brake certification).

The only hard engineering limits in this list are the traction bounds. Everything from 5% to 15% is convention, though grounded in vehicle and human capability; vehicle certification values are minimum capabilities, not maxima.
