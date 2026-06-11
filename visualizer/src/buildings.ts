/**
 * Building meshes for the Catan 3D visualizer.
 * Creates settlements, cities, roads, and the robber.
 */

import * as THREE from "three";
import { PLAYER_COLORS } from "./types";
import { TILE_TOP_HEIGHT } from "./board";
import type { VertexPosition, EdgePosition } from "./board";

/**
 * Create a settlement mesh (small house shape).
 */
export function createSettlement(playerIdx: number): THREE.Group {
  const group = new THREE.Group();
  const color = PLAYER_COLORS[playerIdx];

  // House base (cube)
  const baseGeometry = new THREE.BoxGeometry(0.2, 0.15, 0.2);
  const material = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.6,
    metalness: 0.1,
  });
  const base = new THREE.Mesh(baseGeometry, material);
  base.position.y = 0.075;
  base.castShadow = true;
  group.add(base);

  // Roof (pyramid)
  const roofGeometry = new THREE.ConeGeometry(0.18, 0.12, 4);
  const roofMaterial = new THREE.MeshStandardMaterial({
    color: 0x4a3728, // Brown roof
    roughness: 0.8,
  });
  const roof = new THREE.Mesh(roofGeometry, roofMaterial);
  roof.position.y = 0.21;
  roof.rotation.y = Math.PI / 4;
  roof.castShadow = true;
  group.add(roof);

  return group;
}

/**
 * Create a city mesh (larger building with two towers).
 */
export function createCity(playerIdx: number): THREE.Group {
  const group = new THREE.Group();
  const color = PLAYER_COLORS[playerIdx];

  const material = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.6,
    metalness: 0.1,
  });

  // Main building
  const mainGeometry = new THREE.BoxGeometry(0.25, 0.25, 0.35);
  const main = new THREE.Mesh(mainGeometry, material);
  main.position.y = 0.125;
  main.castShadow = true;
  group.add(main);

  // Left tower
  const towerGeometry = new THREE.BoxGeometry(0.12, 0.35, 0.12);
  const leftTower = new THREE.Mesh(towerGeometry, material);
  leftTower.position.set(-0.08, 0.175, -0.12);
  leftTower.castShadow = true;
  group.add(leftTower);

  // Right tower
  const rightTower = new THREE.Mesh(towerGeometry, material);
  rightTower.position.set(0.08, 0.175, -0.12);
  rightTower.castShadow = true;
  group.add(rightTower);

  // Tower roofs
  const roofGeometry = new THREE.ConeGeometry(0.1, 0.12, 4);
  const roofMaterial = new THREE.MeshStandardMaterial({
    color: 0x4a3728,
    roughness: 0.8,
  });

  const leftRoof = new THREE.Mesh(roofGeometry, roofMaterial);
  leftRoof.position.set(-0.08, 0.41, -0.12);
  leftRoof.rotation.y = Math.PI / 4;
  leftRoof.castShadow = true;
  group.add(leftRoof);

  const rightRoof = new THREE.Mesh(roofGeometry, roofMaterial);
  rightRoof.position.set(0.08, 0.41, -0.12);
  rightRoof.rotation.y = Math.PI / 4;
  rightRoof.castShadow = true;
  group.add(rightRoof);

  return group;
}

/**
 * Create a road mesh (elongated box).
 */
export function createRoad(playerIdx: number): THREE.Mesh {
  const color = PLAYER_COLORS[playerIdx];

  const geometry = new THREE.BoxGeometry(0.5, 0.06, 0.1);
  const material = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.7,
    metalness: 0.1,
  });

  const road = new THREE.Mesh(geometry, material);
  road.castShadow = true;

  return road;
}

/**
 * Create the robber mesh (dark pawn shape).
 */
export function createRobber(): THREE.Group {
  const group = new THREE.Group();

  const material = new THREE.MeshStandardMaterial({
    color: 0x1a1a1a,
    roughness: 0.3,
    metalness: 0.2,
  });

  // Body (cylinder tapering upward)
  const bodyGeometry = new THREE.CylinderGeometry(0.12, 0.18, 0.35, 16);
  const body = new THREE.Mesh(bodyGeometry, material);
  body.position.y = 0.175;
  body.castShadow = true;
  group.add(body);

  // Head (sphere)
  const headGeometry = new THREE.SphereGeometry(0.12, 16, 16);
  const head = new THREE.Mesh(headGeometry, material);
  head.position.y = 0.45;
  head.castShadow = true;
  group.add(head);

  return group;
}

/**
 * Manager class for all buildings on the board.
 */
export class BuildingsManager {
  private group: THREE.Group;
  private settlements: Map<number, THREE.Group> = new Map();
  private cities: Map<number, THREE.Group> = new Map();
  private roads: Map<string, THREE.Mesh> = new Map();
  private robber: THREE.Group;

  private vertexPositions: VertexPosition[];
  private edges: { vertices: [number, number]; position: EdgePosition }[];
  private tilePositions: { x: number; z: number }[];

  constructor(
    vertexPositions: VertexPosition[],
    edges: { vertices: [number, number]; position: EdgePosition }[],
    tilePositions: { x: number; z: number }[]
  ) {
    this.group = new THREE.Group();
    this.vertexPositions = vertexPositions;
    this.edges = edges;
    this.tilePositions = tilePositions;

    // Create robber (starts hidden until placed)
    this.robber = createRobber();
    this.robber.visible = false;
    this.group.add(this.robber);
  }

  getGroup(): THREE.Group {
    return this.group;
  }

  /**
   * Place a settlement at a vertex.
   */
  placeSettlement(vertexIdx: number, playerIdx: number): void {
    // Remove existing building if any
    this.removeBuilding(vertexIdx);

    const settlement = createSettlement(playerIdx);
    const pos = this.vertexPositions[vertexIdx];
    if (pos) {
      settlement.position.set(pos.x, TILE_TOP_HEIGHT, pos.z);
      settlement.userData = { vertexIdx, playerIdx, type: "settlement" };
      this.settlements.set(vertexIdx, settlement);
      this.group.add(settlement);

      // Add pop-in animation
      this.animatePopIn(settlement);
    }
  }

  /**
   * Upgrade a settlement to a city.
   */
  placeCity(vertexIdx: number, playerIdx: number): void {
    // Remove existing settlement
    this.removeBuilding(vertexIdx);

    const city = createCity(playerIdx);
    const pos = this.vertexPositions[vertexIdx];
    if (pos) {
      city.position.set(pos.x, TILE_TOP_HEIGHT, pos.z);
      city.userData = { vertexIdx, playerIdx, type: "city" };
      this.cities.set(vertexIdx, city);
      this.group.add(city);

      // Add pop-in animation
      this.animatePopIn(city);
    }
  }

  /**
   * Place a road between two vertices.
   */
  placeRoad(edgeIdx: number, playerIdx: number): void {
    const edge = this.edges[edgeIdx];
    if (!edge) return;

    const key = `${edge.vertices[0]}-${edge.vertices[1]}`;
    if (this.roads.has(key)) return;

    const road = createRoad(playerIdx);
    road.position.set(edge.position.x, TILE_TOP_HEIGHT + 0.03, edge.position.z);
    road.rotation.y = -edge.position.rotation;

    // Scale road length to match the actual distance between vertices
    // Base geometry is 0.5 units long, subtract padding to avoid overlapping buildings
    const BASE_ROAD_LENGTH = 0.5;
    const ROAD_PADDING = 0.15; // Gap at each end to avoid overlapping settlements
    const targetLength = Math.max(0.1, edge.position.length - ROAD_PADDING);
    road.scale.x = targetLength / BASE_ROAD_LENGTH;

    road.userData = { edgeIdx, playerIdx, type: "road" };

    this.roads.set(key, road);
    this.group.add(road);

    // Add pop-in animation
    this.animatePopIn(road);
  }

  /**
   * Move the robber to a tile.
   */
  moveRobber(tileIdx: number): void {
    const pos = this.tilePositions[tileIdx];
    if (pos) {
      this.robber.position.set(pos.x, TILE_TOP_HEIGHT, pos.z);
      this.robber.visible = true;

      // Add bounce animation
      this.animateBounce(this.robber);
    }
  }

  /**
   * Remove a building from a vertex.
   */
  private removeBuilding(vertexIdx: number): void {
    const settlement = this.settlements.get(vertexIdx);
    if (settlement) {
      this.group.remove(settlement);
      this.settlements.delete(vertexIdx);
    }

    const city = this.cities.get(vertexIdx);
    if (city) {
      this.group.remove(city);
      this.cities.delete(vertexIdx);
    }
  }

  /**
   * Pop-in animation for new buildings.
   */
  private animatePopIn(object: THREE.Object3D): void {
    const targetScale = object.scale.clone();
    object.scale.set(0.01, 0.01, 0.01);

    const startTime = performance.now();
    const duration = 300;

    const animate = () => {
      const elapsed = performance.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Ease out elastic
      const eased =
        progress === 1
          ? 1
          : 1 -
            Math.pow(2, -10 * progress) *
              Math.cos((progress * 10 - 0.75) * ((2 * Math.PI) / 3));

      object.scale.set(
        targetScale.x * eased,
        targetScale.y * eased,
        targetScale.z * eased
      );

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }

  /**
   * Bounce animation for robber movement.
   */
  private animateBounce(object: THREE.Object3D): void {
    const baseY = object.position.y;
    const startTime = performance.now();
    const duration = 400;

    const animate = () => {
      const elapsed = performance.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Bounce effect
      const bounce = Math.sin(progress * Math.PI) * 0.3 * (1 - progress);
      object.position.y = baseY + bounce;

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        object.position.y = baseY;
      }
    };

    requestAnimationFrame(animate);
  }

  /**
   * Clear all buildings (for reset).
   */
  clear(): void {
    // Remove all settlements
    this.settlements.forEach((mesh) => this.group.remove(mesh));
    this.settlements.clear();

    // Remove all cities
    this.cities.forEach((mesh) => this.group.remove(mesh));
    this.cities.clear();

    // Remove all roads
    this.roads.forEach((mesh) => this.group.remove(mesh));
    this.roads.clear();

    // Hide robber
    this.robber.visible = false;
  }

  /**
   * Update from game state arrays.
   */
  updateFromState(
    vertices: number[],
    edges: number[],
    robberTile: number
  ): void {
    // Clear existing
    this.clear();

    // Place buildings from vertices array
    // vertices[i] < 0: empty, 0-3: settlement by player, 4-7: city by player
    for (let i = 0; i < vertices.length; i++) {
      const val = vertices[i];
      if (val >= 0) {
        const playerIdx = val % 4;
        const isCity = val >= 4;
        if (isCity) {
          this.placeCity(i, playerIdx);
        } else {
          this.placeSettlement(i, playerIdx);
        }
      }
    }

    // Place roads from edges array
    // edges[i] < 0: empty, 0-3: road by player
    for (let i = 0; i < edges.length; i++) {
      const val = edges[i];
      if (val >= 0) {
        this.placeRoad(i, val);
      }
    }

    // Move robber
    if (robberTile >= 0) {
      this.moveRobber(robberTile);
    }
  }
}
