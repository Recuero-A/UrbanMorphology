// Headless runner for t-mw/citygen (MIT; third_party/citygen at commit aec49d2e) in the M18 site benchmark.
// No browser, no pixi: it imports the generation core (segment, math, config, heatmap) and the P&M-constrained copy
// of generate.ts (generate_pm.ts).
//
//   node runner.cjs --demo --seed S --out f.json
//       upstream generateCity (default settings, no terrain) AND generateCityPM with terrain off; writes both segment
//       lists in citygen units and whether they are identical (sanity check).
//   node runner.cjs --params p.json --seed S --out f.json
//       one benchmark city: the generator runs in citygen units; metres = centroid + scale * units (x east, y north).
//
// Built by scripts/m18_external.py (esbuild, bundled to scripts/m18_citygen/build/runner.cjs).
import { readFileSync, writeFileSync } from "node:fs";

import config from "../../third_party/citygen/src/config";
import { generateCity } from "../../third_party/citygen/src/city-gen/generate";
import { populationHeatmap } from "../../third_party/citygen/src/city-gen/heatmap";
import { type Point } from "../../third_party/citygen/src/math";

import { generateCityPM, type Terrain } from "./generate_pm";

function arg(name: string): string | null {
    const i = process.argv.indexOf(name);
    return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : null;
}

interface Params {
    n: number; // grid cells per side (row 0 = north)
    res: number; // m
    margin: number; // m
    L: number; // domain side (m)
    elev_path: string; // float32 little-endian, n * n, row-major
    legal_path: string; // uint8 n * n: 1 = inside the city limits and not water
    centroid_m: [number, number]; // city-limits centroid, domain metres
    scale_m_per_unit: number;
    target_length_m: number;
    max_grade: number; // a DEM step with |rise| / run >= max_grade is illegal
    sample_step_m: number;
    prune_min_fraction: number;
    rotate_max_deg: number;
    rotate_step_deg: number;
    segment_budget: number;
    max_proposals: number;
    axiom_dir_step_deg: number;
}

const seed = Number(arg("--seed") ?? "1");
const out = arg("--out");
if (out == null) {
    throw new Error("--out is required");
}

const segOut = (segments: { r: { start: Point; end: Point }; q: { highway?: boolean } }[]) =>
    segments.map((s) => [s.r.start.x, s.r.start.y, s.r.end.x, s.r.end.y, s.q.highway ? 1 : 0]);

if (process.argv.includes("--demo")) {
    const t0 = Date.now();
    const up = segOut(generateCity(seed).segments);
    const t1 = Date.now();
    const pm = segOut(generateCityPM(seed).segments);
    const identical =
        up.length === pm.length && up.every((r, i) => r.every((v, j) => v === pm[i][j]));
    writeFileSync(
        out,
        JSON.stringify({ seed, upstream: up, pm_terrain_off: pm, identical, upstream_ms: t1 - t0 }),
    );
    console.error(`demo seed ${seed}: ${up.length} segments, identical to upstream: ${identical}`);
    process.exit(0);
}

const P: Params = JSON.parse(readFileSync(arg("--params") as string, "utf8"));
const elev = new Float32Array(readFileSync(P.elev_path).buffer.slice(0));
const legal = new Uint8Array(readFileSync(P.legal_path));
const { n, res, margin, L } = P;
const [cx, cy] = P.centroid_m;
const s = P.scale_m_per_unit;

const toM = (p: Point): Point => ({ x: cx + s * p.x, y: cy + s * p.y });
const toU = (p: Point): Point => ({ x: (p.x - cx) / s, y: (p.y - cy) / s });

function cellLegal(x: number, y: number): boolean {
    const c = Math.floor((x + margin) / res);
    const r = Math.floor((L + margin - y) / res);
    if (r < 0 || c < 0 || r >= n || c >= n) {
        return false;
    }
    return legal[r * n + c] === 1;
}

function elevAt(x: number, y: number): number {
    // bilinear between cell centres
    let fc = (x + margin) / res - 0.5;
    let fr = (L + margin - y) / res - 0.5;
    fc = Math.min(Math.max(fc, 0), n - 1.000001);
    fr = Math.min(Math.max(fr, 0), n - 1.000001);
    const c0 = Math.floor(fc);
    const r0 = Math.floor(fr);
    const dc = fc - c0;
    const dr = fr - r0;
    const z00 = elev[r0 * n + c0];
    const z01 = elev[r0 * n + c0 + 1];
    const z10 = elev[(r0 + 1) * n + c0];
    const z11 = elev[(r0 + 1) * n + c0 + 1];
    return (
        z00 * (1 - dr) * (1 - dc) + z01 * (1 - dr) * dc + z10 * dr * (1 - dc) + z11 * dr * dc
    );
}

// P&M illegal area: water, outside the city limits, or a DEM step (<= sample_step_m) steeper than max_grade
const terrain: Terrain = {
    legalUntil(a: Point, b: Point): number {
        const pa = toM(a);
        const pb = toM(b);
        const len = Math.hypot(pb.x - pa.x, pb.y - pa.y);
        if (!cellLegal(pa.x, pa.y)) {
            return 0;
        }
        const k = Math.max(1, Math.ceil(len / P.sample_step_m));
        const d = len / k;
        let zPrev = elevAt(pa.x, pa.y);
        for (let i = 1; i <= k; i++) {
            const t = i / k;
            const x = pa.x + t * (pb.x - pa.x);
            const y = pa.y + t * (pb.y - pa.y);
            if (!cellLegal(x, y)) {
                return (i - 1) / k;
            }
            const z = elevAt(x, y);
            if (d > 0 && Math.abs(z - zPrev) / d >= P.max_grade) {
                return (i - 1) / k;
            }
            zPrev = z;
        }
        return 1;
    },
};

// Axiom (P&M: "This initial segment needs to be located inside a legal area"): the city-limits centroid and the
// upstream direction (due east) when that highway segment is legal; otherwise the first legal (start, direction)
// over cells ordered by distance from the centroid and directions 90, 95, ... (mod 180, as the axiom is a pair of
// opposite segments).
function findAxiom(): { start: Point; dir: number; offset_m: number } | null {
    const hl = config.mapGeneration.HIGHWAY_SEGMENT_LENGTH;
    const dirs: number[] = [];
    for (let a = 0; a < 180; a += P.axiom_dir_step_deg) {
        dirs.push(90 + a);
    }
    const tryAt = (pU: Point): number | null => {
        for (const d of dirs) {
            const e = {
                x: pU.x + hl * Math.sin((d * Math.PI) / 180),
                y: pU.y + hl * Math.cos((d * Math.PI) / 180),
            };
            if (terrain.legalUntil(pU, e) >= 1) {
                return d;
            }
        }
        return null;
    };
    const cells: [number, number, number][] = [];
    for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
            if (legal[r * n + c] === 1) {
                const x = -margin + (c + 0.5) * res;
                const y = L + margin - (r + 0.5) * res;
                cells.push([Math.hypot(x - cx, y - cy), x, y]);
            }
        }
    }
    cells.sort((p, q) => p[0] - q[0]);
    const c0 = { x: 0, y: 0 };
    const d0 = tryAt(c0);
    if (d0 != null) {
        return { start: c0, dir: d0, offset_m: 0 };
    }
    for (const [dist, x, y] of cells) {
        const pU = toU({ x, y });
        const d = tryAt(pU);
        if (d != null) {
            return { start: pU, dir: d, offset_m: dist };
        }
    }
    return null;
}

const t0 = Date.now();
const ax = findAxiom();
let segments: number[][] = [];
let stats: object = { stopReason: "no legal axiom" };
if (ax != null) {
    const res2 = generateCityPM(seed, {
        terrain,
        pm: {
            pruneMinFraction: P.prune_min_fraction,
            rotateMaxDeg: P.rotate_max_deg,
            rotateStepDeg: P.rotate_step_deg,
        },
        axiomStart: ax.start,
        axiomDir: ax.dir,
        stopLength: P.target_length_m / s,
        segmentCountLimit: P.segment_budget,
        maxProposals: P.max_proposals,
    });
    stats = res2.stats;
    segments = res2.segments.map((q) => {
        const a = toM(q.r.start);
        const b = toM(q.r.end);
        return [a.x, a.y, b.x, b.y, q.q.highway ? 1 : 0];
    });
}
// population diagnostics (citygen's own seeded heatmap, unchanged): value at the axiom, share of the city above the
// street-branching threshold
let above = 0;
let tot = 0;
for (let r = 0; r < n; r += 4) {
    for (let c = 0; c < n; c += 4) {
        if (legal[r * n + c] === 1) {
            const u = toU({ x: -margin + (c + 0.5) * res, y: L + margin - (r + 0.5) * res });
            tot += 1;
            above += populationHeatmap.populationAt(u.x, u.y) >
                config.mapGeneration.NORMAL_BRANCH_POPULATION_THRESHOLD
                ? 1
                : 0;
        }
    }
}
const lenM = segments.reduce((acc, q) => acc + Math.hypot(q[2] - q[0], q[3] - q[1]), 0);
writeFileSync(
    out,
    JSON.stringify({
        seed,
        segments,
        stats,
        axiom: ax == null ? null : { start_m: toM(ax.start), dir_deg: ax.dir, offset_from_centroid_m: ax.offset_m },
        length_m: lenM,
        population_at_axiom: ax == null ? null : populationHeatmap.populationAt(ax.start.x, ax.start.y),
        population_share_above_threshold: tot ? above / tot : null,
        ms: Date.now() - t0,
    }),
);
console.error(`seed ${seed}: ${segments.length} segments, ${(lenM / 1000).toFixed(2)} km, ${JSON.stringify(stats)}`);
