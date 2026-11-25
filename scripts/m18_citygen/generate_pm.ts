// Headless Parish & Mueller 2001 street growth for the M18 site benchmark.
//
// Derived from t-mw/citygen src/city-gen/generate.ts at commit aec49d2efaa59a1eebffaacc322459c5219ab9a6
// (MIT licence, (c) 2015 Tobias Mansfield-Williams; full text in third_party/citygen/LICENSE). The upstream file is
// copied here, not edited in place, so third_party/citygen stays pristine. Every change is marked "M18":
//
//   1. localConstraints() gets the P&M 2001 section 3.3 first step ("check if the road segment ends inside or crosses
//      an illegal area") with P&M's own readjustments: prune the segment length up to a factor, rotate it within a
//      maximal angle, otherwise FAILED. Illegal = water, outside the city limits, or a DEM step steeper than the max
//      grade (P&M list "elevation" among the local constraints, section 3.2.1, without a number).
//      The P&M highway bridge rule is not used (the benchmark asks for segments entering water to be rejected or
//      shortened). An extension made by the upstream snap / extend-to-road step must be legal too.
//   2. The axiom (root highway segment and its opposite) can start at any point and direction (upstream: (0, 0),
//      due east); the defaults reproduce upstream.
//   3. Growth stops when the summed segment length reaches a target, or at the segment budget (upstream limit).
//   4. Buildings are not generated (streets only).
// With terrain = null and the default axiom the segments are identical to upstream generateCity (checked by the
// runner's --demo mode).
import Quadtree from "@timohausmann/quadtree-js";
import seedrandom from "seedrandom";

import config from "../../third_party/citygen/src/config";
import math, { type Point } from "../../third_party/citygen/src/math";

import {
    Segment,
    cloneSegment,
    createSegmentUsingDirection,
    insertSegment,
    type Road,
    type SegmentMeta,
    type SegmentQuadtree,
} from "../../third_party/citygen/src/city-gen/segment";
import {
    populationHeatmap,
    seedPopulationHeatmap,
} from "../../third_party/citygen/src/city-gen/heatmap";

// M18: the legal-area oracle (in citygen units). legalUntil returns the fraction t in [0, 1] of the segment
// start -> end that is legal from the start (1 = the whole segment is legal).
export interface Terrain {
    legalUntil(start: Point, end: Point): number;
}

// M18: P&M readjustment parameters (the paper gives no numbers; ours, see outputs/m18/external/README.md)
export interface PMParams {
    pruneMinFraction: number; // prune "up to a certain factor": keep >= this fraction of the proposed length
    rotateMaxDeg: number; // rotate "within a maximal angle"
    rotateStepDeg: number;
}

export interface PMStats {
    proposals: number;
    accepted: number;
    legal: number;
    pruned: number;
    rotated: number;
    failed: number;
    snapRejected: number;
    stopReason: string;
}

interface DebugData {
    intersections?: Point[];
    snaps?: Point[];
    intersectionsRadius?: Point[];
}

type RoadIntersection = false | { x: number; y: number; t: number };

function minDegreeDifference(d1: number, d2: number) {
    const diff = Math.abs(d1 - d2) % 180;
    return Math.min(diff, Math.abs(diff - 180));
}

function doRoadSegmentsIntersect(r1: Road, r2: Road): RoadIntersection {
    return math.doLineSegmentsIntersect(
        r1.start,
        r1.end,
        r2.start,
        r2.end,
        true,
    ) as RoadIntersection;
}

// M18: P&M 2001 section 3.3, first step. "If the first check determines that the road end is inside water, a park
// or another illegal area, the system tries to readjust the values for the road segment in the following ways.
// Prune the segment length up to a certain factor so that it fits inside the legal area of the starting point.
// Rotate the segment within a maximal angle until it is completely inside the legal area."
function legalAreaConstraint(
    segment: Segment,
    terrain: Terrain,
    pm: PMParams,
    stats: PMStats,
): boolean {
    const start = segment.r.start;
    const end = segment.r.end;
    const t = terrain.legalUntil(start, end);
    if (t >= 1) {
        stats.legal += 1;
        return true;
    }
    if (t >= pm.pruneMinFraction) {
        segment.r.setEnd(math.fractionBetween(start, end, t));
        stats.pruned += 1;
        return true;
    }
    const len = segment.length();
    const dir = segment.dir();
    const firstSign = Math.random() < 0.5 ? 1 : -1; // seeded (global seedrandom); only drawn when terrain is on
    for (
        let a = pm.rotateStepDeg;
        a <= pm.rotateMaxDeg + 1e-9;
        a += pm.rotateStepDeg
    ) {
        for (const sgn of [firstSign, -firstSign]) {
            const d = dir + sgn * a;
            const e = {
                x: start.x + len * math.sinDegrees(d),
                y: start.y + len * math.cosDegrees(d),
            };
            if (terrain.legalUntil(start, e) >= 1) {
                segment.r.setEnd(e);
                stats.rotated += 1;
                return true;
            }
        }
    }
    stats.failed += 1;
    return false;
}

function localConstraints(
    segment: Segment,
    segments: Segment[],
    qTree: SegmentQuadtree,
    debugData: DebugData,
    terrain: Terrain | null,
    pm: PMParams,
    stats: PMStats,
) {
    // M18: step 1 of P&M (illegal areas) before the upstream step 2 (crossings)
    if (terrain != null && !legalAreaConstraint(segment, terrain, pm, stats)) {
        return false;
    }
    // M18: an end moved by the snap / extend checks below must stay in the legal area
    const legalTo = (point: Point) => {
        if (terrain == null || terrain.legalUntil(segment.r.start, point) >= 1) {
            return true;
        }
        stats.snapRejected += 1;
        return false;
    };

    const action = {
        priority: 0,
        func: null as (() => boolean) | null,
    };
    let closestIntersectionT: number | null = null;

    const snapDistance = config.mapGeneration.ROAD_SNAP_DISTANCE;
    const snapDistance2 = snapDistance * snapDistance;
    const matches = qTree.retrieve(segment.collider.limits());

    for (const match of matches) {
        const other = match.o ?? null;
        if (other == null) {
            continue;
        }

        // intersection check
        if (action.priority <= 4) {
            const intersection = doRoadSegmentsIntersect(segment.r, other.r);
            if (
                intersection &&
                (closestIntersectionT == null ||
                    intersection.t < closestIntersectionT)
            ) {
                closestIntersectionT = intersection.t;
                const currentOther = other;
                const currentIntersection = intersection;

                action.priority = 4;
                action.func = () => {
                    // if intersecting lines are too similar don't continue
                    if (
                        minDegreeDifference(currentOther.dir(), segment.dir()) <
                        config.mapGeneration.MINIMUM_INTERSECTION_DEVIATION
                    ) {
                        return false;
                    }

                    currentOther.split(
                        currentIntersection,
                        segment,
                        segments,
                        qTree,
                    );
                    segment.r.setEnd(currentIntersection);
                    segment.q.severed = true;
                    debugData.intersections ??= [];
                    debugData.intersections.push({
                        x: currentIntersection.x,
                        y: currentIntersection.y,
                    });

                    return true;
                };
            }
        }

        // snap to crossing within radius check
        if (action.priority <= 3) {
            // current segment's start must have been checked to have been created.
            // other segment's start must have a corresponding end.
            if (math.length(segment.r.end, other.r.end) <= snapDistance) {
                const currentOther = other;
                const point = other.r.end;

                action.priority = 3;
                action.func = () => {
                    if (!legalTo(point)) {
                        return false; // M18
                    }
                    segment.r.setEnd(point);
                    segment.q.severed = true;

                    // update links of otherSegment corresponding to other.r.end
                    const links = currentOther.startIsBackwards()
                        ? currentOther.links.f
                        : currentOther.links.b;
                    // check for duplicate lines, don't add if it exists
                    // this should be done before links are setup, to avoid having to undo that step
                    if (
                        links.some((link) => {
                            return (
                                (math.equalV(link.r.start, segment.r.end) &&
                                    math.equalV(link.r.end, segment.r.start)) ||
                                (math.equalV(link.r.start, segment.r.start) &&
                                    math.equalV(link.r.end, segment.r.end))
                            );
                        })
                    ) {
                        return false;
                    }

                    for (const link of links) {
                        // pick links of remaining segments at junction corresponding to other.r.end
                        const endLinks =
                            link.linksForEndContaining(currentOther);
                        if (endLinks == null) {
                            throw new Error("Expected segment end links");
                        }

                        endLinks.push(segment);
                        // add junction segments to snapped segment
                        segment.links.f.push(link);
                    }

                    links.push(segment);
                    segment.links.f.push(currentOther);
                    debugData.snaps ??= [];
                    debugData.snaps.push({
                        x: point.x,
                        y: point.y,
                    });

                    return true;
                };
            }
        }

        // intersection within radius check
        if (action.priority <= 2) {
            const { distance2, pointOnLine, lineProj2, length2 } =
                math.distanceToLine(segment.r.end, other.r.start, other.r.end);

            if (
                distance2 < snapDistance2 &&
                lineProj2 >= 0 &&
                lineProj2 <= length2
            ) {
                const currentOther = other;
                const point = pointOnLine;

                action.priority = 2;
                action.func = () => {
                    if (!legalTo(point)) {
                        return false; // M18
                    }
                    segment.r.setEnd(point);
                    segment.q.severed = true;

                    // if intersecting lines are too similar don't continue
                    if (
                        minDegreeDifference(currentOther.dir(), segment.dir()) <
                        config.mapGeneration.MINIMUM_INTERSECTION_DEVIATION
                    ) {
                        return false;
                    }

                    currentOther.split(point, segment, segments, qTree);
                    debugData.intersectionsRadius ??= [];
                    debugData.intersectionsRadius.push({
                        x: point.x,
                        y: point.y,
                    });

                    return true;
                };
            }
        }
    }

    if (action.func != null) {
        return action.func();
    }

    return true;
}

function createBranchTemplate(previousSegment: Segment) {
    const template = (
        direction: number,
        length: number,
        t: number,
        q: SegmentMeta,
    ) => {
        return createSegmentUsingDirection(
            previousSegment.r.end,
            direction,
            length,
            t,
            q,
        );
    };

    return {
        // used for highways or going straight on a normal branch
        continue(direction: number) {
            return template(
                direction,
                previousSegment.length(),
                0,
                previousSegment.q,
            );
        },

        // not using q, i.e. not highways
        branch(direction: number) {
            return template(
                direction,
                config.mapGeneration.DEFAULT_SEGMENT_LENGTH,
                previousSegment.q.highway
                    ? config.mapGeneration.NORMAL_BRANCH_TIME_DELAY_FROM_HIGHWAY
                    : 0,
                {},
            );
        },
    };
}

function generateBranches(previousSegment: Segment) {
    const newBranches: Segment[] = [];

    if (!previousSegment.q.severed) {
        const template = createBranchTemplate(previousSegment);
        const continueStraight = template.continue(previousSegment.dir());
        const straightPop = populationHeatmap.popOnRoad(continueStraight.r);

        if (previousSegment.q.highway) {
            const randomStraight = template.continue(
                previousSegment.dir() +
                    config.mapGeneration.RANDOM_STRAIGHT_ANGLE(),
            );
            const randomPop = populationHeatmap.popOnRoad(randomStraight.r);
            let roadPop: number;

            if (randomPop > straightPop) {
                newBranches.push(randomStraight);
                roadPop = randomPop;
            } else {
                newBranches.push(continueStraight);
                roadPop = straightPop;
            }

            if (
                roadPop >
                config.mapGeneration.HIGHWAY_BRANCH_POPULATION_THRESHOLD
            ) {
                const leftRoll = Math.random();
                if (
                    leftRoll < config.mapGeneration.HIGHWAY_BRANCH_PROBABILITY
                ) {
                    const leftHighwayBranch = template.continue(
                        previousSegment.dir() -
                            90 +
                            config.mapGeneration.RANDOM_BRANCH_ANGLE(),
                    );
                    newBranches.push(leftHighwayBranch);
                } else {
                    const rightRoll = Math.random();
                    if (
                        rightRoll <
                        config.mapGeneration.HIGHWAY_BRANCH_PROBABILITY
                    ) {
                        const rightHighwayBranch = template.continue(
                            previousSegment.dir() +
                                90 +
                                config.mapGeneration.RANDOM_BRANCH_ANGLE(),
                        );
                        newBranches.push(rightHighwayBranch);
                    }
                }
            }
        } else if (
            straightPop >
            config.mapGeneration.NORMAL_BRANCH_POPULATION_THRESHOLD
        ) {
            newBranches.push(continueStraight);
        }

        if (
            straightPop >
            config.mapGeneration.NORMAL_BRANCH_POPULATION_THRESHOLD
        ) {
            const leftRoll = Math.random();
            if (leftRoll < config.mapGeneration.DEFAULT_BRANCH_PROBABILITY) {
                const leftBranch = template.branch(
                    previousSegment.dir() -
                        90 +
                        config.mapGeneration.RANDOM_BRANCH_ANGLE(),
                );
                newBranches.push(leftBranch);
            } else {
                const rightRoll = Math.random();
                if (
                    rightRoll < config.mapGeneration.DEFAULT_BRANCH_PROBABILITY
                ) {
                    const rightBranch = template.branch(
                        previousSegment.dir() +
                            90 +
                            config.mapGeneration.RANDOM_BRANCH_ANGLE(),
                    );
                    newBranches.push(rightBranch);
                }
            }
        }
    }

    for (const branch of newBranches) {
        branch.previousSegmentToLink = previousSegment;
    }

    return newBranches;
}

export interface PMOptions {
    segmentCountLimit?: number;
    terrain?: Terrain | null; // M18
    pm?: PMParams; // M18
    axiomStart?: Point; // M18 (upstream: (0, 0))
    axiomDir?: number; // M18 (upstream: 90 = due east)
    stopLength?: number; // M18: stop when the summed segment length (units) reaches this
    maxProposals?: number; // M18: safety cap on popped proposals
}

export function generateCityPM(seed: number, options: PMOptions = {}) {
    const debugData: DebugData = {};
    const segmentCountLimit =
        options.segmentCountLimit ?? config.mapGeneration.SEGMENT_COUNT_LIMIT;
    const terrain = options.terrain ?? null;
    const pm = options.pm ?? {
        pruneMinFraction: 0.5,
        rotateMaxDeg: 90,
        rotateStepDeg: 5,
    };
    const stopLength = options.stopLength ?? Infinity;
    const maxProposals = options.maxProposals ?? Infinity;
    const stats: PMStats = {
        proposals: 0,
        accepted: 0,
        legal: 0,
        pruned: 0,
        rotated: 0,
        failed: 0,
        snapRejected: 0,
        stopReason: "queue empty",
    };

    seedrandom(String(seed), { global: true });
    // this perlin noise library only supports 65536 different seeds
    seedPopulationHeatmap(Math.random());

    const priorityQ: Segment[] = [];

    // setup first segments in queue (M18: start point and direction are options; defaults = upstream)
    const upstreamAxiom =
        options.axiomStart == null && options.axiomDir == null;
    const rootSegment = upstreamAxiom
        ? new Segment(
              { x: 0, y: 0 },
              { x: config.mapGeneration.HIGHWAY_SEGMENT_LENGTH, y: 0 },
              0,
              {
                  highway: true,
              },
          )
        : createSegmentUsingDirection(
              options.axiomStart ?? { x: 0, y: 0 },
              options.axiomDir ?? 90,
              config.mapGeneration.HIGHWAY_SEGMENT_LENGTH,
              0,
              { highway: true },
          );
    const oppositeDirection = cloneSegment(rootSegment);
    const newEnd = upstreamAxiom
        ? {
              x:
                  rootSegment.r.start.x -
                  config.mapGeneration.HIGHWAY_SEGMENT_LENGTH,
              y: oppositeDirection.r.end.y,
          }
        : {
              x: 2 * rootSegment.r.start.x - rootSegment.r.end.x,
              y: 2 * rootSegment.r.start.y - rootSegment.r.end.y,
          };
    oppositeDirection.r.setEnd(newEnd);
    oppositeDirection.links.b.push(rootSegment);
    rootSegment.links.b.push(oppositeDirection);
    priorityQ.push(rootSegment);
    priorityQ.push(oppositeDirection);

    const segments: Segment[] = [];
    const qTree = new Quadtree(
        config.mapGeneration.QUADTREE_PARAMS,
        config.mapGeneration.QUADTREE_MAX_OBJECTS,
        config.mapGeneration.QUADTREE_MAX_LEVELS,
    ) as SegmentQuadtree;

    let totalLength = 0;
    while (priorityQ.length > 0 && segments.length < segmentCountLimit) {
        if (stats.proposals >= maxProposals) {
            stats.stopReason = "proposal cap";
            break;
        }
        // pop smallest r(ti, ri, qi) from Q (i.e., smallest 't')
        let minT: number | null = null;
        let minTIndex = 0;

        for (const [index, segment] of priorityQ.entries()) {
            if (minT == null || segment.t < minT) {
                minT = segment.t;
                minTIndex = index;
            }
        }

        const minSegment = priorityQ.splice(minTIndex, 1)[0];
        stats.proposals += 1;
        const accepted = localConstraints(
            minSegment,
            segments,
            qTree,
            debugData,
            terrain,
            pm,
            stats,
        );

        if (accepted) {
            minSegment.setupBranchLinks();

            insertSegment(minSegment, segments, qTree);
            stats.accepted += 1;

            for (const newSegment of generateBranches(minSegment)) {
                newSegment.t = minSegment.t + 1 + newSegment.t;
                priorityQ.push(newSegment);
            }

            // M18: density target (splits partition a segment, so the sum is recomputed)
            if (stopLength !== Infinity) {
                totalLength = 0;
                for (const s of segments) {
                    totalLength += s.length();
                }
                if (totalLength >= stopLength) {
                    stats.stopReason = "density target";
                    break;
                }
            }
        }
    }
    if (segments.length >= segmentCountLimit) {
        stats.stopReason = "segment budget";
    }

    return { segments, debugData, stats, heatmap: populationHeatmap };
}
