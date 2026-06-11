/**
 * Board rendering for the Catan 3D visualizer.
 * Generates hexagonal tiles, vertices, and edges.
 */

import * as THREE from "three";
import {
  ResourceType,
  RESOURCE_COLORS,
  NUM_TILES,
  NUM_VERTICES,
  TILE_ROWS,
  TILE_VERTICES,
} from "./types";
import { getConfig, BoardConfig } from "./config";

// Hexagon geometry constants
export const HEX_RADIUS = 1.0;
export const HEX_HEIGHT = 0.2; // Base extrusion depth
const BEVEL_THICKNESS = 0.02;
export const TILE_TOP_HEIGHT = HEX_HEIGHT + BEVEL_THICKNESS; // Actual top surface height

// Default gap values (can be overridden by config)
export const DEFAULT_TILE_GAP_HORIZ = 0.5;
export const DEFAULT_TILE_GAP_VERT = 0.01;

/**
 * Compute spacing values from config.
 */
function getSpacing(config?: BoardConfig): {
  horizSpacing: number;
  vertSpacing: number;
} {
  const cfg = config || getConfig();
  const HEX_WIDTH = HEX_RADIUS * 2;
  const horizSpacing = HEX_WIDTH * 0.75 + cfg.tileGapHoriz;
  const vertSpacing = HEX_RADIUS * Math.sqrt(3) + cfg.tileGapVert;
  return { horizSpacing, vertSpacing };
}

export interface TilePosition {
  x: number;
  z: number;
  row: number;
  col: number;
}

export interface VertexPosition {
  x: number;
  z: number;
}

export interface EdgePosition {
  x: number;
  z: number;
  rotation: number; // Y-axis rotation in radians
  length: number; // Distance between the two vertices
}

/**
 * Compute 3D positions for all 19 tiles.
 */
export function computeTilePositions(config?: BoardConfig): TilePosition[] {
  const positions: TilePosition[] = [];
  const { horizSpacing, vertSpacing } = getSpacing(config);

  for (let row = 0; row < TILE_ROWS.length; row++) {
    const tilesInRow = TILE_ROWS[row];
    // Center the row horizontally
    const rowOffset = (tilesInRow - 1) * horizSpacing * 0.5;
    // Vertical offset from center (row 2 is the middle)
    const vertOffset = (row - 2) * vertSpacing;

    for (let col = 0; col < tilesInRow; col++) {
      positions.push({
        x: col * horizSpacing - rowOffset,
        z: vertOffset,
        row,
        col,
      });
    }
  }

  return positions;
}

/**
 * Compute 3D positions for all 54 vertices.
 * Vertices are positioned at the CENTROID of all tile corners that reference them.
 * This places vertices in the center of the gap between adjacent tiles.
 */
export function computeVertexPositions(
  tilePositions: TilePosition[],
  config?: BoardConfig
): VertexPosition[] {
  const cfg = config || getConfig();
  const radiusScale = cfg.vertexRadiusScale;

  // Collect ALL corner positions for each vertex from all tiles
  const vertexContributions = new Map<number, { x: number; z: number }[]>();

  // For each tile, compute its 6 vertex positions at HEX_RADIUS
  for (let tileIdx = 0; tileIdx < NUM_TILES; tileIdx++) {
    const tile = tilePositions[tileIdx];
    const vertices = TILE_VERTICES[tileIdx];

    for (let i = 0; i < 6; i++) {
      const vertexIdx = vertices[i];

      // Flat-top hexagon: vertices at 90°, 30°, -30°, -90°, -150°, 150°
      const angle = Math.PI / 2 - (i * Math.PI) / 3;
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);

      // Direction from tile center to vertex
      const dx = cosA;
      const dz = -sinA;

      // Vertex position at hexagon corner (scaled by radiusScale)
      const x = tile.x + HEX_RADIUS * radiusScale * dx;
      const z = tile.z + HEX_RADIUS * radiusScale * dz;

      // Add this corner position to the vertex's contributions
      if (!vertexContributions.has(vertexIdx)) {
        vertexContributions.set(vertexIdx, []);
      }
      vertexContributions.get(vertexIdx)!.push({ x, z });
    }
  }

  // Compute centroid (average) of all contributions for each vertex
  const vertexPositions = new Map<number, VertexPosition>();
  vertexContributions.forEach((contributions, vertexIdx) => {
    const sumX = contributions.reduce((sum, c) => sum + c.x, 0);
    const sumZ = contributions.reduce((sum, c) => sum + c.z, 0);
    const count = contributions.length;

    // Apply per-vertex override if exists
    const override = cfg.vertexOverrides.find((v) => v.idx === vertexIdx);
    const offsetX = override?.dx || 0;
    const offsetZ = override?.dz || 0;

    vertexPositions.set(vertexIdx, {
      x: sumX / count + offsetX,
      z: sumZ / count + offsetZ,
    });
  });

  // Convert map to array
  const positions: VertexPosition[] = new Array(NUM_VERTICES);
  vertexPositions.forEach((pos, idx) => {
    positions[idx] = pos;
  });

  return positions;
}

/**
 * Compute edge data from vertex connectivity.
 * Returns edges as pairs of vertex indices and their 3D positions.
 */
export function computeEdges(
  vertexPositions: VertexPosition[],
  config?: BoardConfig
): { vertices: [number, number]; position: EdgePosition }[] {
  const cfg = config || getConfig();
  const edgeSet = new Set<string>();
  const edges: { vertices: [number, number]; position: EdgePosition }[] = [];

  let edgeIdx = 0;

  // Build edges from tile vertex adjacency
  for (let tileIdx = 0; tileIdx < NUM_TILES; tileIdx++) {
    const vertices = TILE_VERTICES[tileIdx];
    for (let i = 0; i < 6; i++) {
      const v1 = vertices[i];
      const v2 = vertices[(i + 1) % 6];
      const key = v1 < v2 ? `${v1}-${v2}` : `${v2}-${v1}`;

      if (!edgeSet.has(key)) {
        edgeSet.add(key);

        const p1 = vertexPositions[v1];
        const p2 = vertexPositions[v2];

        // Edge center position
        let x = (p1.x + p2.x) / 2;
        let z = (p1.z + p2.z) / 2;

        // Edge rotation (angle from p1 to p2)
        let rotation = Math.atan2(p2.z - p1.z, p2.x - p1.x);

        // Edge length (distance between vertices, scaled)
        const dx = p2.x - p1.x;
        const dz = p2.z - p1.z;
        const length = Math.sqrt(dx * dx + dz * dz) * cfg.edgeLengthScale;

        // Apply per-edge override if exists
        const override = cfg.edgeOverrides.find((e) => e.idx === edgeIdx);
        if (override) {
          x += override.dx;
          z += override.dz;
          rotation += override.rotation;
        }

        edges.push({
          vertices: [Math.min(v1, v2), Math.max(v1, v2)],
          position: { x, z, rotation, length },
        });

        edgeIdx++;
      }
    }
  }

  return edges;
}

/**
 * Create a hexagonal prism geometry for tiles.
 */
function createHexagonGeometry(): THREE.BufferGeometry {
  const shape = new THREE.Shape();

  // Create hexagon shape (flat-top)
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i + Math.PI / 6;
    const x = HEX_RADIUS * Math.cos(angle);
    const y = HEX_RADIUS * Math.sin(angle);
    if (i === 0) {
      shape.moveTo(x, y);
    } else {
      shape.lineTo(x, y);
    }
  }
  shape.closePath();

  // Extrude to create 3D prism
  const geometry = new THREE.ExtrudeGeometry(shape, {
    depth: HEX_HEIGHT,
    bevelEnabled: true,
    bevelThickness: BEVEL_THICKNESS,
    bevelSize: BEVEL_THICKNESS,
    bevelSegments: 1,
  });

  // Rotate so the flat side is on top (extrusion goes down)
  geometry.rotateX(-Math.PI / 2);
  geometry.translate(0, HEX_HEIGHT, 0);

  return geometry;
}

// Safe radius to avoid overlapping with adjacent tiles
// Hexagons are staggered, so diagonally adjacent tiles are closer
// Using more conservative values to prevent any cross-tile overlap
const MAX_SAFE_RADIUS = 0.45; // Conservative radius for all directions

// Helpers for random placement outside center
function getRandomPosition(
  minRadius: number,
  maxRadius: number,
  existingPositions: { x: number; z: number; r: number }[] = [],
  objectRadius: number = 0.1
): { x: number; z: number } | null {
  const maxAttempts = 20;
  for (let i = 0; i < maxAttempts; i++) {
    const angle = Math.random() * Math.PI * 2;
    const r = minRadius + Math.random() * (maxRadius - minRadius);
    let x = r * Math.cos(angle);
    let z = r * Math.sin(angle);

    // Clamp to safe bounds to prevent overlap with adjacent tiles
    const maxR = MAX_SAFE_RADIUS - objectRadius;
    x = Math.max(-maxR, Math.min(maxR, x));
    z = Math.max(-maxR, Math.min(maxR, z));

    // Check collision with center (approx 0.3 radius for number token)
    const distFromCenter = Math.sqrt(x * x + z * z);
    if (distFromCenter < 0.35 + objectRadius) continue;

    // Check collision with existing objects
    let collision = false;
    for (const pos of existingPositions) {
      const dx = pos.x - x;
      const dz = pos.z - z;
      const dist = Math.sqrt(dx * dx + dz * dz);
      if (dist < pos.r + objectRadius) {
        collision = true;
        break;
      }
    }

    if (!collision) {
      return { x, z };
    }
  }
  return null;
}

/**
 * Create forest (Wood) details - Trees
 */
function createForestDetails(): THREE.Group {
  const group = new THREE.Group();
  const trunkGeo = new THREE.CylinderGeometry(0.05, 0.08, 0.2, 5);
  const leavesGeo = new THREE.ConeGeometry(0.2, 0.5, 5);

  const trunkMat = new THREE.MeshStandardMaterial({ color: 0x5d4037 }); // Brown
  const leavesMat = new THREE.MeshStandardMaterial({ color: 0x2e7d32 }); // Dark green

  const positions: { x: number; z: number; r: number }[] = [];

  // Place 3-5 trees randomly but OUTSIDE center
  const numTrees = 3 + Math.floor(Math.random() * 3);
  for (let i = 0; i < numTrees; i++) {
    // Tree approx radius 0.15
    const pos = getRandomPosition(0.4, 0.8, positions, 0.15);
    if (!pos) continue;

    positions.push({ x: pos.x, z: pos.z, r: 0.15 });

    const tree = new THREE.Group();

    const trunk = new THREE.Mesh(trunkGeo, trunkMat);
    trunk.position.y = 0.1;
    trunk.castShadow = true;
    tree.add(trunk);

    const leaves = new THREE.Mesh(leavesGeo, leavesMat);
    leaves.position.y = 0.4;
    leaves.castShadow = true;
    tree.add(leaves);

    tree.position.set(pos.x, HEX_HEIGHT, pos.z);
    // Random scale variation
    const s = 0.8 + Math.random() * 0.4;
    tree.scale.set(s, s, s);

    group.add(tree);
  }
  return group;
}

/**
 * Create hills (Brick) details - Rounded mounds
 */
function createHillsDetails(): THREE.Group {
  const group = new THREE.Group();
  const hillGeo = new THREE.SphereGeometry(1, 8, 8);
  const hillMat = new THREE.MeshStandardMaterial({
    color: 0xc0392b, // Terracotta
    roughness: 0.9,
  });

  const positions: { x: number; z: number; r: number }[] = [];

  // 2-3 hills
  const numHills = 2 + Math.floor(Math.random() * 2);
  for (let i = 0; i < numHills; i++) {
    const hill = new THREE.Mesh(hillGeo, hillMat);
    // Hill radius ~0.35 when scaled
    const pos = getRandomPosition(0.45, 0.75, positions, 0.3);
    if (!pos) continue;

    positions.push({ x: pos.x, z: pos.z, r: 0.3 });

    // Flatten spheres to look like mounds
    hill.scale.set(0.35, 0.2, 0.35);
    hill.position.set(pos.x, HEX_HEIGHT, pos.z);

    hill.castShadow = true;
    group.add(hill);
  }
  return group;
}

/**
 * Create mountains (Ore) details - Jagged peaks
 */
function createMountainDetails(): THREE.Group {
  const group = new THREE.Group();
  const peakGeo = new THREE.ConeGeometry(0.3, 0.8, 4); // Low poly pyramid
  const peakMat = new THREE.MeshStandardMaterial({
    color: 0x7f8c8d, // Grey
    roughness: 0.6,
    flatShading: true,
  });

  const positions: { x: number; z: number; r: number }[] = [];

  // 2-3 peaks
  const numPeaks = 2 + Math.floor(Math.random() * 2);
  for (let i = 0; i < numPeaks; i++) {
    const peak = new THREE.Mesh(peakGeo, peakMat);
    const pos = getRandomPosition(0.5, 0.8, positions, 0.2);
    if (!pos) continue;

    positions.push({ x: pos.x, z: pos.z, r: 0.2 });

    peak.position.set(pos.x, HEX_HEIGHT + 0.3, pos.z);
    // Random rotation and scale
    peak.rotation.y = Math.random() * Math.PI;
    peak.scale.setScalar(0.8 + Math.random() * 0.5);

    peak.castShadow = true;
    group.add(peak);
  }
  return group;
}

/**
 * Create fields (Wheat) details - Patches
 */
function createFieldDetails(): THREE.Group {
  const group = new THREE.Group();
  // Create patches of wheat using small boxes
  const patchGeo = new THREE.BoxGeometry(0.1, 0.2, 0.1);
  const patchMat = new THREE.MeshStandardMaterial({
    color: 0xf1c40f, // Yellow
  });

  const positions: { x: number; z: number; r: number }[] = [];

  // Many small stalks
  for (let i = 0; i < 15; i++) {
    const stalk = new THREE.Mesh(patchGeo, patchMat);
    const pos = getRandomPosition(0.4, 0.85, positions, 0.08);
    if (!pos) continue;

    positions.push({ x: pos.x, z: pos.z, r: 0.08 });

    stalk.position.set(pos.x, HEX_HEIGHT + 0.1, pos.z);
    stalk.rotation.y = Math.random() * Math.PI;
    stalk.scale.y = 0.5 + Math.random() * 1.0;

    stalk.castShadow = true;
    group.add(stalk);
  }
  return group;
}

/**
 * Create pasture (Sheep) details - Bushes/Sheep
 */
function createPastureDetails(): THREE.Group {
  const group = new THREE.Group();
  // Using white spheres to represent sheep
  const sheepGeo = new THREE.SphereGeometry(0.12, 6, 6);
  const sheepMat = new THREE.MeshStandardMaterial({ color: 0xffffff });
  const headGeo = new THREE.SphereGeometry(0.06, 6, 6);
  const headMat = new THREE.MeshStandardMaterial({ color: 0x333333 }); // Black head

  const positions: { x: number; z: number; r: number }[] = [];

  // 3-4 sheep
  const numSheep = 3 + Math.floor(Math.random() * 2);
  for (let i = 0; i < numSheep; i++) {
    const sheepGroup = new THREE.Group();

    const body = new THREE.Mesh(sheepGeo, sheepMat);
    body.position.y = 0.12;
    body.scale.z = 1.2; // Elongate body
    sheepGroup.add(body);

    const head = new THREE.Mesh(headGeo, headMat);
    head.position.set(0, 0.18, 0.15);
    sheepGroup.add(head);

    const pos = getRandomPosition(0.45, 0.8, positions, 0.2);
    if (!pos) continue;

    positions.push({ x: pos.x, z: pos.z, r: 0.2 });

    sheepGroup.position.set(pos.x, HEX_HEIGHT, pos.z);
    sheepGroup.rotation.y = Math.random() * Math.PI * 2;

    group.add(sheepGroup);
  }
  return group;
}

/**
 * Create desert details - Cactus/Rocks
 */
function createDesertDetails(): THREE.Group {
  const group = new THREE.Group();

  // A cactus
  const cactusMat = new THREE.MeshStandardMaterial({ color: 0x2e7d32 });
  const cactus = new THREE.Group();

  // Main stem
  const stem = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.1, 0.4, 4, 8),
    cactusMat
  );
  stem.position.y = 0.3;
  cactus.add(stem);

  // Arm
  const arm = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.08, 0.2, 4, 8),
    cactusMat
  );
  arm.position.set(0.15, 0.3, 0);
  arm.rotation.z = -Math.PI / 4;
  cactus.add(arm);

  // Desert is usually center (robber start), no number token.
  cactus.position.set(0, HEX_HEIGHT, 0);
  cactus.castShadow = true;
  group.add(cactus);

  return group;
}

/**
 * Create the board group with all tiles.
 */
export function createBoard(
  tileResources: number[],
  tileNumbers: number[],
  config?: BoardConfig
): {
  group: THREE.Group;
  tilePositions: TilePosition[];
  vertexPositions: VertexPosition[];
  edges: { vertices: [number, number]; position: EdgePosition }[];
  tileMeshes: THREE.Mesh[];
} {
  const group = new THREE.Group();
  const tilePositions = computeTilePositions(config);
  const vertexPositions = computeVertexPositions(tilePositions, config);
  const edges = computeEdges(vertexPositions, config);
  const tileMeshes: THREE.Mesh[] = [];

  const hexGeometry = createHexagonGeometry();

  // Create tile meshes
  for (let i = 0; i < NUM_TILES; i++) {
    const pos = tilePositions[i];
    const resourceType = tileResources[i] as ResourceType;
    const color = RESOURCE_COLORS[resourceType];

    const tileGroup = new THREE.Group();
    tileGroup.position.set(pos.x, 0, pos.z);

    // Base Tile
    const material = new THREE.MeshStandardMaterial({
      color,
      roughness: 0.8,
      metalness: 0.1,
    });

    const mesh = new THREE.Mesh(hexGeometry, material);
    mesh.receiveShadow = true;
    mesh.userData = { tileIdx: i, resourceType };
    tileMeshes.push(mesh);
    tileGroup.add(mesh);

    // Add decorations based on resource type
    let decoration: THREE.Group | null = null;
    switch (resourceType) {
      case ResourceType.WOOD:
        decoration = createForestDetails();
        break;
      case ResourceType.BRICK:
        decoration = createHillsDetails();
        break;
      case ResourceType.STONE:
        decoration = createMountainDetails();
        break;
      case ResourceType.WHEAT:
        decoration = createFieldDetails();
        break;
      case ResourceType.SHEEP:
        decoration = createPastureDetails();
        break;
      case ResourceType.DESERT:
        decoration = createDesertDetails();
        break;
    }

    if (decoration) {
      tileGroup.add(decoration);
    }

    // Add number token (except for desert)
    if (resourceType !== ResourceType.DESERT) {
      const numberToken = createNumberToken(tileNumbers[i]);
      // Standardize height, slightly above base
      const tokenHeight = HEX_HEIGHT + 0.1;
      numberToken.position.set(0, tokenHeight, 0);
      tileGroup.add(numberToken);
    }

    group.add(tileGroup);
  }

  // Add ocean/background plane
  const oceanGeometry = new THREE.CircleGeometry(12, 32);
  const oceanMaterial = new THREE.MeshStandardMaterial({
    color: 0x1e3a5f,
    roughness: 0.9,
  });
  const ocean = new THREE.Mesh(oceanGeometry, oceanMaterial);
  ocean.rotation.x = -Math.PI / 2;
  ocean.position.y = -0.05;
  ocean.receiveShadow = true;
  group.add(ocean);

  return { group, tilePositions, vertexPositions, edges, tileMeshes };
}

/**
 * Create a number token for a tile.
 */
function createNumberToken(number: number): THREE.Group {
  const group = new THREE.Group();

  // Token disc
  const discGeometry = new THREE.CylinderGeometry(0.25, 0.25, 0.05, 16);
  const isRedNumber = number === 6 || number === 8;
  const discMaterial = new THREE.MeshStandardMaterial({
    color: 0xf5f5dc, // Beige
    roughness: 0.5,
  });
  const disc = new THREE.Mesh(discGeometry, discMaterial);
  disc.castShadow = true;
  group.add(disc);

  // Number text using canvas texture
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d")!;

  ctx.fillStyle = "#f5f5dc";
  ctx.fillRect(0, 0, 64, 64);

  ctx.fillStyle = isRedNumber ? "#dc2626" : "#1f2937";
  ctx.font = "bold 36px Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(number.toString(), 32, 32);

  const texture = new THREE.CanvasTexture(canvas);
  const textMaterial = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
  });

  const textPlane = new THREE.PlaneGeometry(0.4, 0.4);
  const textMesh = new THREE.Mesh(textPlane, textMaterial);
  textMesh.rotation.x = -Math.PI / 2;
  textMesh.position.y = 0.03;
  group.add(textMesh);

  return group;
}

/**
 * Create vertex position markers (debug helper).
 * Yellow spheres show where settlements/cities would be placed.
 */
export function createVertexMarkers(
  vertexPositions: VertexPosition[]
): THREE.Group {
  const group = new THREE.Group();
  const geometry = new THREE.SphereGeometry(0.12, 12, 12);
  const material = new THREE.MeshStandardMaterial({
    color: 0xffff00, // Bright yellow
    emissive: 0x444400,
  });

  for (let i = 0; i < vertexPositions.length; i++) {
    const pos = vertexPositions[i];
    if (pos) {
      const marker = new THREE.Mesh(geometry, material);
      marker.position.set(pos.x, TILE_TOP_HEIGHT + 0.1, pos.z);
      marker.userData = { vertexIdx: i };
      marker.castShadow = true;
      group.add(marker);
    }
  }

  return group;
}

/**
 * Create edge position markers (debug helper).
 * Cyan capsules show where roads would be placed.
 */
export function createEdgeMarkers(
  edges: { vertices: [number, number]; position: EdgePosition }[]
): THREE.Group {
  const group = new THREE.Group();
  const geometry = new THREE.CapsuleGeometry(0.06, 0.2, 4, 8);
  const material = new THREE.MeshStandardMaterial({
    color: 0x00ffff, // Bright cyan
    emissive: 0x004444,
  });

  for (let i = 0; i < edges.length; i++) {
    const edge = edges[i];
    const marker = new THREE.Mesh(geometry, material);
    marker.position.set(
      edge.position.x,
      TILE_TOP_HEIGHT + 0.1,
      edge.position.z
    );
    // Rotate capsule to lie flat and align with edge direction
    marker.rotation.z = Math.PI / 2;
    marker.rotation.y = -edge.position.rotation;
    marker.userData = { edgeIdx: i, vertices: edge.vertices };
    marker.castShadow = true;
    group.add(marker);
  }

  return group;
}
