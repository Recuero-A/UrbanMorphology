# M16 real style references (m16-style-refs-1)

36 real 512 m street windows (9 per style: 5 `calib`, 4 `sort`), labelled from documented typology (source per window in `configs/m16_style_refs/manifest.csv`). Held-out metros (Lleida, Alicante, Seoul, Daegu, Lima, Trujillo) and the USA are excluded. Street definition = training graphs (STREET_VALUES, build_graph, highway classes, M12 levels). Radial metrics use the declared focus (= tile centre), others the tile centre.
OSM data (c) OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright); fetched 2026-11-20T16:26:22+00:00 to 2026-11-20T16:52:28+00:00.

## Calibration subset: mean (SD) per style, n = 5 each

| metric | grid | organic | radial | culdesac |
|---|---|---|---|---|
| block_rect_share | 0.689 (0.21) | 0.1 (0.11) | 0.129 (0.23) | 0.125 (0.25) n=4 |
| block_phi_med | 0.504 (0.11) | 0.419 (0.06) | 0.514 (0.041) | 0.336 (0.13) n=4 |
| block_logA_med | 8.61 (1.1) | 6.76 (0.83) | 8.5 (1.1) | 6.2 (1.9) n=4 |
| block_logA_sd | 1.09 (0.63) | 1.66 (0.31) | 1.02 (0.59) | 1.56 (1.1) n=2 |
| jangle_right_share | 0.847 (0.13) | 0.39 (0.09) | 0.628 (0.15) | 0.536 (0.096) |
| through_share | 0.971 (0.064) | 0.895 (0.025) | 0.924 (0.079) | 0.956 (0.045) |
| stroke_xjunc_lw | 4.38 (0.63) | 2.26 (0.95) | 3.31 (1.4) | 1.93 (1) |
| tree_len_share | 0.014 (0.024) | 0.115 (0.068) | 0.0393 (0.034) | 0.45 (0.18) |
| term_len_med | 34.3 (16) n=2 | 34.1 (8) | 41.9 (21) | 78.1 (36) |
| deadend_per_km | 0.306 (0.45) | 3.24 (1.5) | 0.99 (0.53) | 6.76 (3.5) |
| loops_per_km2 | 76.3 (44) | 172 (48) | 163 (2.2e+02) | 7.63 (6.6) |
| share_4way | 0.806 (0.16) | 0.247 (0.028) | 0.396 (0.15) | 0.0101 (0.014) |
| spokes_long | 1.6 (1.5) | 1.8 (1.3) | 9.4 (3.4) | 1.4 (1.5) |
| ring_cover_max | 0.322 (0.065) | 0.421 (0.032) | 0.793 (0.22) | 0.241 (0.1) |
| phi_boeing | 0.963 (0.04) | 0.179 (0.11) | 0.17 (0.069) | 0.549 (0.1) |
| orth_order | 0.952 (0.044) | 0.349 (0.18) | 0.13 (0.036) | 0.651 (0.096) |
| street_km | 5.7 (1.4) | 8.28 (1) | 7.27 (4.4) | 3.53 (0.71) |

NaN values are skipped (n shown when < 5): a window with no interior block or no interior dead end has no block / terminal statistics.

Notes: closed pedestrian ways (area=yes squares) are kept, as in the training graphs (osmium/OSMnx path), and appear as small loops. Radburn 'turning loops' (Kitimat C07, C09; Crestwood C03) are part of the documented cul-de-sac typology ('Living streets mostly take the form of culs-de-sac and turning loops') but have few dead ends. Raw data: OSM API 0.6 map call (live database; data timestamp = fetch time).

## Windows

| id | city | area | style | subset | source |
|---|---|---|---|---|---|
| G01 | Barcelona | Dreta de l'Eixample | grid | calib | https://en.wikipedia.org/wiki/Eixample |
| G02 | Mannheim | Quadrate (centre) | grid | calib | https://en.wikipedia.org/wiki/Mannheim |
| G03 | Melbourne | Hoddle Grid (CBD) | grid | calib | https://en.wikipedia.org/wiki/Hoddle_Grid |
| G04 | Buenos Aires | Balvanera | grid | calib | https://en.wikipedia.org/wiki/Balvanera |
| G05 | Kyoto | Nakagyo-ku (Rakuchu) | grid | calib | https://en.wikipedia.org/wiki/Kyoto |
| G06 | Adelaide | City centre (Light's plan), east of Victoria Square | grid | sort | https://en.wikipedia.org/wiki/Adelaide_city_centre |
| G07 | Sapporo | Chuo-ku (south of Odori) | grid | sort | https://en.wikipedia.org/wiki/Sapporo |
| G08 | Valencia | Ensanche (Pla del Remei / Gran Via) | grid | sort | https://es.wikipedia.org/wiki/Ensanche_(Valencia) |
| G09 | Turin | San Salvario | grid | sort | https://en.wikipedia.org/wiki/Turin |
| O01 | Toledo | Casco historico | organic | calib | https://www.britannica.com/place/Toledo-Spain |
| O02 | Cordoba | Historic centre (Juderia) | organic | calib | https://en.wikipedia.org/wiki/Historic_centre_of_C%C3%B3rdoba |
| O03 | Tunis | Medina | organic | calib | https://en.wikipedia.org/wiki/Medina_of_Tunis |
| O04 | Granada | Albaicin | organic | calib | https://en.wikipedia.org/wiki/Albaic%C3%ADn |
| O05 | Barcelona | Barri Gotic | organic | calib | https://en.wikipedia.org/wiki/Gothic_Quarter,_Barcelona |
| O06 | Marrakesh | Medina | organic | sort | https://en.wikipedia.org/wiki/Marrakesh |
| O07 | Kairouan | Medina | organic | sort | https://whc.unesco.org/en/list/499 |
| O08 | Lisbon | Alfama | organic | sort | https://en.wikipedia.org/wiki/Alfama |
| O09 | Seville | Barrio de Santa Cruz | organic | sort | https://es.wikipedia.org/wiki/Barrio_de_Santa_Cruz_(Sevilla) |
| R01 | Palmanova | Piazza Grande (star fortress town) | radial | calib | https://en.wikipedia.org/wiki/Palmanova |
| R02 | Paris | Place Charles de Gaulle (Etoile) | radial | calib | https://en.wikipedia.org/wiki/Place_Charles_de_Gaulle |
| R03 | Coevorden | Markt (old fortress town) | radial | calib | https://en.wikipedia.org/wiki/Coevorden |
| R04 | Grammichele | Piazza Carafa (hexagonal plan) | radial | calib | https://it.wikipedia.org/wiki/Grammichele |
| R05 | New Delhi | Connaught Place | radial | calib | https://en.wikipedia.org/wiki/Connaught_Place,_New_Delhi |
| R06 | Paris | Place de la Nation | radial | sort | https://fr.wikipedia.org/wiki/Place_de_la_Nation_(Paris) |
| R07 | Canberra | City Hill (Griffin plan) | radial | sort | https://en.wikipedia.org/wiki/City_Hill,_Canberra |
| R08 | Hamina | Raatihuoneentori (circular plan) | radial | sort | https://en.wikipedia.org/wiki/Hamina |
| R09 | Rome | Piazza del Popolo (Tridente) | radial | sort | https://en.wikipedia.org/wiki/Piazza_del_Popolo |
| C01 | Livingston | Craigshill | culdesac | calib | https://en.wikipedia.org/wiki/Radburn_design_housing |
| C02 | Canberra | Charnwood | culdesac | calib | https://en.wikipedia.org/wiki/Radburn_design_housing |
| C03 | Perth | Crestwood Estate, Thornlie | culdesac | calib | https://en.wikipedia.org/wiki/Radburn_design_housing |
| C04 | Bracknell | Great Hollands | culdesac | calib | https://en.wikipedia.org/wiki/Radburn_design_housing |
| C05 | Kingston upon Hull | Bransholme | culdesac | calib | https://en.wikipedia.org/wiki/Radburn_design_housing |
| C06 | Canberra | Rivett | culdesac | sort | https://en.wikipedia.org/wiki/Radburn_design_housing |
| C07 | Kitimat | Kildala | culdesac | sort | https://en.wikipedia.org/wiki/Radburn_design_housing |
| C08 | Yate | Stanshawe Estate | culdesac | sort | https://en.wikipedia.org/wiki/Radburn_design_housing |
| C09 | Kitimat | Nechako | culdesac | sort | https://en.wikipedia.org/wiki/Radburn_design_housing |

## Failures / exclusions

- none among the 36

Candidates dropped before the final list:

- Fez, Fes el-Bali: medina lanes are mapped highway=footway/path: almost no streets under STREET_VALUES (fetched, preview)
- Genoa, centro storico: caruggi mostly footway / pedestrian areas: sparse, polygon-dominated graph (fetched, preview)
- Karlsruhe, Schloss: the documented fan starts ~250 m south of the palace tower; radials in the palace garden are footways, so a window centred on the focus shows no fan (fetched, preview)
- Siena, Toledo (Wikipedia): Wikipedia pages do not describe the street layout; Toledo kept with Britannica instead, Siena dropped
- Madrid, Barrio de Salamanca: source pages name the ensanche but do not describe a grid
- Versailles, Place d'Armes: Wikipedia describes a 'rational and symmetrical grid', not the trident
- Bourtange / Nordlingen: Wikipedia pages do not describe radial streets
- La Plata: grid crossed by diagonals: mixed grid/radial, not a clean grid label
- Winnipeg, Wildwood Park: its bays are mapped highway=service: no streets inside the window (fetched, preview)
- Kitimat downtown / Minto (Sydney) / Hull Orchard Park: window fell on a commercial centre or arterials with few culs-de-sac (Orchard Park also tried 400 m north); replaced by Kitimat Kildala and Nechako and Hull Bransholme
- Spanish urbanizaciones (Las Rozas, Pozuelo, ...): no source describing a cul-de-sac / Radburn layout for a specific area

Figure: `outputs/m16/figures/style_refs.png` (all 36 windows, ids only).
