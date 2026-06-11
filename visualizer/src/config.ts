/**
 * Configuration module for the Catan 3D visualizer.
 * Manages board layout parameters and per-element overrides.
 */

export interface VertexOverride {
  idx: number;
  dx: number;
  dz: number;
}

export interface EdgeOverride {
  idx: number;
  dx: number;
  dz: number;
  rotation: number; // Additional rotation offset in radians
}

export interface BoardConfig {
  // Global tile gap parameters
  tileGapHoriz: number;
  tileGapVert: number;

  // Vertex positioning
  vertexRadiusScale: number; // Scale factor for vertex distance from tile center

  // Edge positioning
  edgeLengthScale: number; // Scale factor for edge length

  // Per-element overrides
  vertexOverrides: VertexOverride[];
  edgeOverrides: EdgeOverride[];
}

// Default configuration values
export const DEFAULT_CONFIG: BoardConfig = {
  tileGapHoriz: 0.5,
  tileGapVert: 0.01,
  vertexRadiusScale: 1.0,
  edgeLengthScale: 1.0,
  vertexOverrides: [],
  edgeOverrides: [],
};

// Current active configuration
let currentConfig: BoardConfig = { ...DEFAULT_CONFIG };

/**
 * Get the current configuration.
 */
export function getConfig(): BoardConfig {
  return currentConfig;
}

/**
 * Update the current configuration with partial values.
 */
export function updateConfig(partial: Partial<BoardConfig>): BoardConfig {
  currentConfig = { ...currentConfig, ...partial };
  return currentConfig;
}

/**
 * Reset configuration to defaults.
 */
export function resetConfig(): BoardConfig {
  currentConfig = { ...DEFAULT_CONFIG, vertexOverrides: [], edgeOverrides: [] };
  return currentConfig;
}

/**
 * Set a vertex override (adds or updates).
 */
export function setVertexOverride(idx: number, dx: number, dz: number): void {
  const existing = currentConfig.vertexOverrides.findIndex((v) => v.idx === idx);
  if (existing >= 0) {
    currentConfig.vertexOverrides[existing] = { idx, dx, dz };
  } else {
    currentConfig.vertexOverrides.push({ idx, dx, dz });
  }
}

/**
 * Get a vertex override if it exists.
 */
export function getVertexOverride(idx: number): VertexOverride | undefined {
  return currentConfig.vertexOverrides.find((v) => v.idx === idx);
}

/**
 * Set an edge override (adds or updates).
 */
export function setEdgeOverride(
  idx: number,
  dx: number,
  dz: number,
  rotation: number
): void {
  const existing = currentConfig.edgeOverrides.findIndex((e) => e.idx === idx);
  if (existing >= 0) {
    currentConfig.edgeOverrides[existing] = { idx, dx, dz, rotation };
  } else {
    currentConfig.edgeOverrides.push({ idx, dx, dz, rotation });
  }
}

/**
 * Get an edge override if it exists.
 */
export function getEdgeOverride(idx: number): EdgeOverride | undefined {
  return currentConfig.edgeOverrides.find((e) => e.idx === idx);
}

/**
 * Load configuration from JSON file.
 */
export async function loadConfig(): Promise<BoardConfig> {
  try {
    const response = await fetch("/board-config.json");
    if (response.ok) {
      const loaded = await response.json();
      currentConfig = {
        ...DEFAULT_CONFIG,
        ...loaded,
        vertexOverrides: loaded.vertexOverrides || [],
        edgeOverrides: loaded.edgeOverrides || [],
      };
      console.log("Board config loaded:", currentConfig);
    } else {
      console.log("No board config found, using defaults");
      currentConfig = { ...DEFAULT_CONFIG };
    }
  } catch (error) {
    console.log("Could not load board config, using defaults:", error);
    currentConfig = { ...DEFAULT_CONFIG };
  }
  return currentConfig;
}

/**
 * Export configuration as JSON string.
 */
export function exportConfigAsJson(): string {
  return JSON.stringify(currentConfig, null, 2);
}

/**
 * Download configuration as a JSON file.
 */
export function downloadConfig(): void {
  const json = exportConfigAsJson();
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "board-config.json";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Copy configuration to clipboard.
 */
export async function copyConfigToClipboard(): Promise<void> {
  const json = exportConfigAsJson();
  await navigator.clipboard.writeText(json);
  console.log("Config copied to clipboard");
}


