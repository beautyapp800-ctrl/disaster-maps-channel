# Channel brief: U.S. natural disasters, mapped (step 1)

Status 2026-11-22: step 1 is ready for the owner's choice. Automation is OFF.
Repo: github.com/beautyapp800-ctrl/disaster-maps-channel (public). Work happens on `dev`, and `main` gets it only through smoke.
The engine was copied from America Over Time (map-timelapse). map-timelapse itself was not changed.

## 1. Names (handles checked on youtube.com 2026-11-22: 404 = free)

| # | Name | Handle | Why |
|---|------|--------|-----|
| A | ~~Every Disaster Mapped~~ → **Storm Atlas USA** (channel created 23.11) | @StormAtlasUSA | Matches the "Every X Mapped" title format. It is also the most common search phrase ("every tornado", "every hurricane"). |
| B | **Natural Fury Mapped** | @NaturalFuryMapped | More emotional and easy to remember. It is less tied to the title format. |
| C | **Disasters Mapped** | @DisastersMapped | Short, and still readable on a 40 px avatar. |

Taken: @DisasterAtlas, @AtlasOfDisasters, @HazardAtlas, @StormAtlas.
Also free: @EveryStormMapped, @MappedDisasters, @HazardMapped, @DisasterMapsUSA.

## 2. Avatars and banners (branding/options/)

All of them use real data only: 71,912 NOAA SPC tornado tracks, NOAA NHC HURDAT2 tracks and Census state borders.
`*_preview.png` shows each avatar at its real 98 px and 40 px sizes. Each banner preview shows the desktop crop and the phone crop.

- **Avatar A** `avatar_A_tornado_x`: a big yellow "X" over the F3+ tracks of Tornado Alley. It reads best at 40 px.
- **Avatar B** `avatar_B_hurricane_eye`: an orange/yellow hurricane spiral with real Cat 3+ tracks behind it.
- **Avatar C** `avatar_C_usa_edm`: the U.S. silhouette filled with every tornado 1950-2025, shock-wave rings and an "EDM" badge.
- **Banner A** `banner_A_every_tornado`: the whole U.S. with every tornado 1950-2025, coloured by F/EF scale.
- **Banner B** `banner_B_major_hurricanes`: every Atlantic hurricane that reached Cat 3+, 1851-2025, coloured by wind speed.
- **Banner C** `banner_C_hazard_mix`: F3+ tornado tracks plus earthquake rings at the Northridge 1994 epicentre.

The banner text uses name A. I will redraw it for the name you choose.

## 3. Description and keywords

**Description (About):**
Every tornado, hurricane, earthquake and wildfire in U.S. history, mapped from real government data. No narration: just the map, the numbers and the events, year by year. Data: NOAA Storm Prediction Center, NOAA National Hurricane Center, USGS Earthquake Hazards Program, NOAA NCEI, NIFC. Two new maps every week.

**Keywords:** every tornado mapped, tornado map, tornado history, hurricane map, every hurricane, hurricane landfall, earthquake map, california earthquakes, wildfire map, tornado alley, natural disasters usa, disaster map animation, NOAA data, USGS earthquakes, map timelapse, data visualization

**Video settings:** category 27 (Education), madeForKids=false, containsSyntheticMedia=false.
**Description rules:** sources as plain text without https://, no < >, under 5000 characters, 3-5 hashtags, and the marker [ref: disastermaps/<slug>].

## 4. Topics (22, ordered by search demand × data strength)

Search demand comes from YouTube autocomplete (2026-11-22). Every topic lists its data source; see data/sources.json for the licences.
A topic whose source still has `licence_confirmed=false` is not built until the licence is confirmed on the source site.

| # | Working title (≤70 chars) | Data | Size of the real data | Licence |
|---|---|---|---|---|
| 1 | Every Tornado in US History Mapped (1950–2025) | NOAA SPC | 73,458 tornadoes, 60 F5/EF5 | ✅ PD |
| 2 | Every Hurricane That Hit America, Mapped (1851–2025) | NOAA HURDAT2 + GridSat-B1 clouds | ~300 U.S. hurricane landfalls (rough count, checked before the build) | ✅ PD / CDR open |
| 3 | Every Earthquake in California Since 1900, Mapped | USGS ComCat | 4,362 quakes of M4+ in the CA box | ✅ PD |
| 4 | Every F5 and EF5 Tornado in History, Mapped | NOAA SPC | 60 | ✅ |
| 5 | Every Tornado in Texas History, Mapped | NOAA SPC | autocomplete "every tornado in texas" | ✅ |
| 6 | Every Tornado in Oklahoma History, Mapped | NOAA SPC | autocomplete "every tornado in oklahoma" | ✅ |
| 7 | Tornado Alley Is Shifting: 75 Years of Tornadoes Mapped | NOAA SPC | autocomplete "tornado alley shifting" | ✅ |
| 8 | Every Major Hurricane (Cat 3+) to Hit the US, Mapped | HURDAT2 | ~98 (rough count) | ✅ |
| 9 | Every Hurricane to Hit Florida, Mapped | HURDAT2 | ~107 (rough count) | ✅ |
| 10 | Every Earthquake in Alaska Since 1900, Mapped | USGS ComCat | 3,376 quakes of M5+ | ✅ |
| 11 | Every Big Earthquake in the Lower 48 Since 1900 | USGS ComCat | 1,323 quakes of M5+ | ✅ |
| 12 | Why Oklahoma Started Shaking: Every Earthquake Mapped | USGS ComCat | 3,087 quakes of M3+ | ✅ |
| 13 | Every Wildfire in the US Since 1984, Mapped | MTBS / NIFC perimeters | 98,168 perimeters (NIFC) | ⏳ confirm |
| 14 | Every California Wildfire Ever Recorded, Mapped | NIFC perimeter history | | ⏳ confirm |
| 15 | Every Tsunami That Hit the United States, Mapped | NCEI tsunami DB + ETOPO relief | 277 events | ⏳ confirm |
| 16 | Every Volcanic Eruption in the US, Mapped | NCEI + USGS | 50 significant eruptions | ⏳ confirm |
| 17 | Every Giant Hailstorm in US History, Mapped | NOAA SPC hail 1955–2024 | | ✅ |
| 18 | Every Damaging Windstorm and Derecho, Mapped | NOAA SPC wind 1955–2024 | | ✅ |
| 19 | Every Flash Flood in America Since 1996, Mapped | NCEI Storm Events | | ✅ |
| 20 | Every Blizzard in America Since 1996, Mapped | NCEI Storm Events | | ✅ |
| 21 | Every Hurricane That Hit Hawaii and the Pacific Coast | HURDAT2 NE/Central Pacific 1949–2025 | | ✅ |
| 22 | The Worst Tornado Outbreaks in US History, Mapped (1974, 2011…) | NOAA SPC | | ✅ |

Weak topics, kept for later: fireballs/meteors (only 31 JPL fireballs over the lower 48, too few for 10 minutes).
Excluded sources: OpenFEMA (its terms limit use to statistics/reporting), U.S. Drought Monitor (copyright NDMC, no permission found).
Real satellite clouds for hurricanes come from GridSat-B1 (1980 onward). Earlier storms get no satellite clouds; their spiral is drawn only from the track and the wind, and the on-screen caption says so.

## 5. Competitors (YouTube search, 2026-11-22)

- **Small "Every X Mapped" channels**: Terra Tempus (Every Earthquake Since 1900, Mapped: 96 views), Disaster Odds (hurricane landfalls 31 views, wildfires 16), Japan Mapped, PlotPoint, World History Visualization (24 views). They are new, use a dark static style and get almost no views. This is a free niche for a bright, animated format.
- **Strong single hits**: Michael Ferragamo "I Mapped Every Single U.S. Hurricane in Recorded History" got **159K** (41 min, hand-made). Esri "California Wildfires 1910-2019" got **451K**. Insider "all earthquakes of the past 15 years" got **450K**. PacificTWC "120 Years of Earthquakes and Tsunamis" got **1.08M**. Map Pack "Why 2025 is the year Tornado Alley changed" got **341K**.
- **Big tornado channels, footage and explainers rather than maps**: Swegle Studios "Worst Tornado of Every Year" **1.27M**, TornadoTRX "History of F5 Tornadoes" **2.8M**, Disasters Explained **193K–3.5M**.
- **Conclusion**: the demand for tornado/hurricane history is in the millions, while full data maps with 3D and effects barely exist. Our edge is "every event from the real database" plus motion every 2–4 s.

## 6. Render benchmark 1080p vs 4K (measured on GitHub, run 45018929049)

Runner: ubuntu-latest, 4 CPUs. 3D uses OpenGL on the CPU (Mesa llvmpipe) with a 1.3-million-triangle relief (NOAA ETOPO1) and MSAA ×4. x264 medium CRF 20 encoding ran at the same time as the rendering.

| per frame | 1080p | 4K |
|---|---|---|
| 2D map (tracks accumulating, glow, counter) + encoding | 0.07 s | 0.44 s |
| 3D relief (tilted camera, vortex) + encoding | 0.48 s | 1.09 s |
| bitrate 2D / 3D | 6 / 97 MB per min | 13 / 239 MB per min |

Projection for a 12-minute video at 24 fps (17,280 frames). The mix is assumed at 60% 2D and 40% 3D, with ×1.5 added for effects (clouds, fire, particles, cards):
- **1080p**: ~1.7 CPU-hours. 12 segments take ~10 min each. The file is ~0.5 GB.
- **4K**: ~5 CPU-hours. 24 segments take ~13–15 min each, ~45–60 min wall clock at max-parallel 8. The file is ~1.2 GB, which fits under the 2 GB release limit.
- **Verdict**: 4K is realistic, far below the 4-hour job limit. Even 1080p viewers get better quality because YouTube encodes 4K uploads with VP9/AV1. It costs about 3× more CPU and twice the file size.
- **Recommendation**: 4K, with a fallback to 1080p if a segment runs past 45 min. I will re-measure on the pilot with all the effects.
- Samples: docs/bench/sample3d_1080.jpg, sample3d_2160.jpg. The relief exaggeration in the benchmark (×35) is too high; the pilot will use ×8–12 with smoothing.

## 7. Schedule proposal (not enabled)

- Publishing: Wed + Sat 18:00 Kyiv (11:00 ET). This does not overlap America Over Time (Mon/Fri).
- Rendering: Mon + Thu 05:00 UTC, max-parallel 8. This avoids the beautyapp800-ctrl jobs: Stoic 01:17/10:43, kids 02:27/10:53/13:47, weekly Mon 06:43 (light), Germany manual for now.
- Reserve: 2 ready videos plus 1–2 emergency videos. The watchdog puts an emergency video in place if a slot is less than 24 h away and has no video. There is no retry on 429 and no mail.

## 8. Google Cloud: recommendation

| Project | Already on it | Load |
|---|---|---|
| yt-factory-510411 | Stoic (daily), kids (daily), Huge To Scale (daily) | the busiest: 3 uploads a day |
| hypothetica-511509 | What If, America Over Time, Germany (chosen 22.11) | 3 channels, low load |
| gen-lang-client-0367233741 | archive, prehistoric, Closed Tabs + Gemini | 3 channels, Gemini uses a separate quota |

**Recommendation: gen-lang-client-0367233741**, to spread the risk. hypothetica already has 3 channels, including AOT with the same engine; if that project has a problem, both map channels would stop together.
The upload code (youtube_auth.py, publish.py) works with any client. Before connecting I will check the project's daily upload quota.
If you choose hypothetica-511509, the token setup follows the AOT path, which is already tested.

## 9. What I need from you

1. Name (A/B/C or your own), avatar (A/B/C) and banner (A/B/C).
2. Google Cloud project.
3. 4K or 1080p.
4. Whether the Wed/Sat 18:00 Kyiv schedule is OK.

After that: step 2, the pilot "Every Tornado in US History Mapped (1950–2025)" plus 3 covers.
