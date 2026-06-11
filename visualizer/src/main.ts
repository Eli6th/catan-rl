/**
 * Main entry point for the Catan 3D Visualizer.
 * Sets up Three.js scene, loads game data, and handles UI.
 */

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { createBoard, createVertexMarkers, createEdgeMarkers } from "./board";
import { BuildingsManager } from "./buildings";
import { ReplayController, ReplayEvent } from "./replay";
import { LogData, PLAYER_COLOR_NAMES, ReplayRecord, ReplaySummary } from "./types";
import { loadConfig, getConfig, BoardConfig } from "./config";
import { EditorController } from "./editor";
import { PlayController, createGame, PlayView } from "./play";

// Debug mode - set to true to show vertex/edge placement markers
const DEBUG_SHOW_MARKERS = true;

// DOM Elements
let container: HTMLElement;
let loadingEl: HTMLElement;
let logSelect: HTMLSelectElement;
let infoPanel: HTMLElement;
let controlsPanel: HTMLElement;
let turnInfo: HTMLElement;
let playersInfo: HTMLElement;
let actionCounter: HTMLElement;
let speedValue: HTMLElement;

// Three.js objects
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let renderer: THREE.WebGLRenderer;
let controls: OrbitControls;

// Game objects
let boardGroup: THREE.Group | null = null;
let buildingsManager: BuildingsManager | null = null;
let replayController: ReplayController | null = null;
let editorController: EditorController | null = null;

// Current game data for rebuilding board
let currentLogData: LogData | null = null;
let vertexMarkersGroup: THREE.Group | null = null;
let edgeMarkersGroup: THREE.Group | null = null;
let tileMeshes: THREE.Mesh[] = [];
let playController: PlayController | null = null;

/**
 * Initialize the Three.js scene.
 */
function initScene(): void {
  container = document.getElementById("canvas-container")!;
  loadingEl = document.getElementById("loading")!;

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  // Camera
  camera = new THREE.PerspectiveCamera(
    60,
    container.clientWidth / container.clientHeight,
    0.1,
    100
  );
  camera.position.set(0, 8, 8);
  camera.lookAt(0, 0, 0);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  // Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.minDistance = 3;
  controls.maxDistance = 20;
  controls.maxPolarAngle = Math.PI / 2.1;

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(5, 10, 5);
  directionalLight.castShadow = true;
  directionalLight.shadow.mapSize.width = 2048;
  directionalLight.shadow.mapSize.height = 2048;
  directionalLight.shadow.camera.near = 0.5;
  directionalLight.shadow.camera.far = 50;
  directionalLight.shadow.camera.left = -10;
  directionalLight.shadow.camera.right = 10;
  directionalLight.shadow.camera.top = 10;
  directionalLight.shadow.camera.bottom = -10;
  scene.add(directionalLight);

  // Handle window resize
  window.addEventListener("resize", onWindowResize);

  // Initialize editor controller
  editorController = new EditorController(container, camera, scene);
  editorController.setConfigChangeCallback(onConfigChange);

  // Start render loop
  animate();
}

/**
 * Handle window resize.
 */
function onWindowResize(): void {
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

/**
 * Animation loop.
 */
function animate(): void {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

/**
 * Initialize UI elements.
 */
function initUI(): void {
  logSelect = document.getElementById("log-select") as HTMLSelectElement;
  infoPanel = document.getElementById("info-panel")!;
  controlsPanel = document.getElementById("controls")!;
  turnInfo = document.getElementById("turn-info")!;
  playersInfo = document.getElementById("players-info")!;
  actionCounter = document.getElementById("action-counter")!;
  speedValue = document.getElementById("speed-value")!;

  // Playback controls
  document.getElementById("btn-reset")!.addEventListener("click", () => {
    replayController?.reset();
  });

  document.getElementById("btn-step-back")!.addEventListener("click", () => {
    replayController?.stepBackward();
  });

  document.getElementById("btn-play-pause")!.addEventListener("click", () => {
    replayController?.togglePlayPause();
    updatePlayPauseButton();
  });

  document.getElementById("btn-step-forward")!.addEventListener("click", () => {
    replayController?.stepForward();
  });

  // Speed slider
  const speedSlider = document.getElementById(
    "speed-slider"
  ) as HTMLInputElement;
  speedSlider.addEventListener("input", () => {
    const speed = parseFloat(speedSlider.value);
    replayController?.setPlaybackSpeed(speed);
    speedValue.textContent = `${speed}x`;
  });

  // Log selector
  logSelect.addEventListener("change", async () => {
    const filename = logSelect.value;
    if (filename) {
      await loadGame(filename);
    }
  });
}

/**
 * Update play/pause button state.
 */
function updatePlayPauseButton(): void {
  const btn = document.getElementById("btn-play-pause")!;
  const isPlaying = replayController?.getIsPlaying() ?? false;
  btn.textContent = isPlaying ? "⏸ Pause" : "▶ Play";
}

/**
 * Fetch available log files from server.
 */
async function fetchReplaySummaries(): Promise<ReplaySummary[]> {
  try {
    const response = await fetch("/api/replays");
    if (!response.ok) throw new Error("Failed to fetch replays");
    const data = await response.json();
    return data.replays;
  } catch (error) {
    console.error("Error fetching replay summaries:", error);
    return [];
  }
}

/**
 * Fetch a specific log file.
 */
async function fetchLog(filename: string): Promise<LogData> {
  const response = await fetch(`/api/replays/${filename}`);
  if (!response.ok) throw new Error("Failed to fetch replay");
  const data = await response.json();
  return toLogData(data.replay as ReplayRecord);
}

function toLogData(replay: ReplayRecord): LogData {
  return {
    seed: replay.summary.seed,
    num_players: replay.summary.num_players,
    tile_resources: replay.board.tile_resources,
    tile_numbers: replay.board.tile_numbers,
    port_types: replay.board.port_types,
    actions: replay.actions,
  };
}

/**
 * Populate the log file selector.
 */
async function populateLogSelector(): Promise<void> {
  const replays = await fetchReplaySummaries();

  // Clear existing options
  logSelect.innerHTML = '<option value="">-- Select a log file --</option>';

  if (replays.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No replays found";
    option.disabled = true;
    logSelect.appendChild(option);
  } else {
    for (const replay of replays) {
      const option = document.createElement("option");
      option.value = replay.id;
      option.textContent = `${replay.id} (${replay.action_count} actions)`;
      logSelect.appendChild(option);
    }
  }
}

/**
 * Load and display a game.
 */
async function loadGame(filename: string): Promise<void> {
  loadingEl.textContent = "Loading game...";
  loadingEl.style.display = "block";
  infoPanel.style.display = "none";
  controlsPanel.style.display = "none";
  // Leave play mode if active.
  document.getElementById("play-panel")!.style.display = "none";

  try {
    const logData = await fetchLog(filename);
    currentLogData = logData;

    // Build the board with current config
    rebuildBoard(logData, getConfig());

    // Create replay controller
    replayController = new ReplayController(logData);
    replayController.setBuildingsManager(buildingsManager!);
    replayController.setEventCallback(onReplayEvent);

    // Initialize display
    updateInfoPanel(replayController.getState());
    updateActionCounter();

    // Show UI
    loadingEl.style.display = "none";
    infoPanel.style.display = "block";
    controlsPanel.style.display = "flex";

    // Reset camera
    camera.position.set(0, 8, 8);
    controls.target.set(0, 0, 0);
    controls.update();
  } catch (error) {
    console.error("Error loading game:", error);
    loadingEl.textContent = "Error loading game. Check console for details.";
  }
}

/**
 * Rebuild the board with new config.
 */
function rebuildBoard(logData: LogData, config: BoardConfig): void {
  // Remove existing board
  if (boardGroup) {
    scene.remove(boardGroup);
    boardGroup = null;
  }

  // Create new board with config
  const boardResult = createBoard(
    logData.tile_resources,
    logData.tile_numbers,
    config
  );
  boardGroup = boardResult.group;
  tileMeshes = boardResult.tileMeshes;
  scene.add(boardGroup);

  // Create buildings manager
  buildingsManager = new BuildingsManager(
    boardResult.vertexPositions,
    boardResult.edges,
    boardResult.tilePositions
  );
  boardGroup.add(buildingsManager.getGroup());

  // Add debug markers if enabled
  if (DEBUG_SHOW_MARKERS) {
    vertexMarkersGroup = createVertexMarkers(boardResult.vertexPositions);
    edgeMarkersGroup = createEdgeMarkers(boardResult.edges);
    boardGroup.add(vertexMarkersGroup);
    boardGroup.add(edgeMarkersGroup);

    // Update editor with new markers
    if (editorController) {
      editorController.setMarkers(vertexMarkersGroup, edgeMarkersGroup);
    }
  }
}

/**
 * Handle config changes from the editor.
 */
function onConfigChange(config: BoardConfig): void {
  if (currentLogData) {
    // Store current replay state
    const actionIndex = replayController?.getActionIndex() || 0;

    // Rebuild board with new config
    rebuildBoard(currentLogData, config);

    // Recreate replay controller and restore state
    replayController = new ReplayController(currentLogData);
    replayController.setBuildingsManager(buildingsManager!);
    replayController.setEventCallback(onReplayEvent);
    replayController.jumpToAction(actionIndex);

    // Update display
    updateInfoPanel(replayController.getState());
    updateActionCounter();
  }
}

/**
 * Handle replay events.
 */
function onReplayEvent(event: ReplayEvent): void {
  updateInfoPanel(event.state);
  updateActionCounter();
  updatePlayPauseButton();
}

/**
 * Update the info panel with current game state.
 */
function updateInfoPanel(
  state: ReturnType<ReplayController["getState"]>
): void {
  const logData = replayController?.getLogData();
  if (!logData) return;

  // Turn info
  turnInfo.innerHTML = `
    <div>Turn: ${state.turn}</div>
    <div>Phase: ${state.phase}</div>
  `;

  // Player info
  let playersHtml = "";
  for (let i = 0; i < logData.num_players; i++) {
    const isActive = i === state.currentPlayer;
    const colorName = PLAYER_COLOR_NAMES[i];
    const vp = state.victoryPoints[i];

    playersHtml += `
      <div class="player-info ${isActive ? "active" : ""}">
        <div class="player-name player-${i}">
          Player ${i} (${colorName})
          ${state.winner === i ? " 👑" : ""}
        </div>
        <div class="player-stats">
          VP: ${vp}
          ${state.longestRoadPlayer === i ? " | 🛣️ Longest Road" : ""}
          ${state.largestArmyPlayer === i ? " | ⚔️ Largest Army" : ""}
        </div>
      </div>
    `;
  }
  playersInfo.innerHTML = playersHtml;
}

/**
 * Update action counter display.
 */
function updateActionCounter(): void {
  if (!replayController) return;
  const current = replayController.getActionIndex();
  const total = replayController.getTotalActions();
  actionCounter.textContent = `Action: ${current} / ${total}`;
}

/**
 * Start an interactive game against the live Rust engine.
 */
async function startPlayGame(): Promise<void> {
  const numPlayers = parseInt(
    (document.getElementById("play-num-players") as HTMLSelectElement).value,
    10
  );
  const seedRaw = (
    document.getElementById("play-seed") as HTMLInputElement
  ).value.trim();
  const seats: string[] = [];
  for (let i = 0; i < numPlayers; i++) {
    seats.push(
      (document.getElementById(`play-seat-${i}`) as HTMLSelectElement).value
    );
  }

  loadingEl.textContent = "Starting game on Rust engine...";
  loadingEl.style.display = "block";

  let view: PlayView;
  try {
    view = await createGame({
      numPlayers,
      seed: seedRaw ? parseInt(seedRaw, 10) : undefined,
      seats,
    });
  } catch (e) {
    loadingEl.textContent =
      "Could not reach the engine server. Run: cargo run -p catan-web";
    console.error(e);
    return;
  }

  // Leave replay mode entirely.
  replayController = null;
  currentLogData = null;
  infoPanel.style.display = "none";
  controlsPanel.style.display = "none";

  // Build the board for this game's tile layout.
  rebuildBoard(
    {
      seed: view.state.seed,
      num_players: view.state.num_players,
      tile_resources: view.state.tile_resources,
      tile_numbers: view.state.tile_numbers,
      port_types: view.state.port_types,
      actions: [],
    },
    getConfig()
  );

  if (!playController) {
    playController = new PlayController({
      container,
      camera,
      buildingsManager: buildingsManager!,
      vertexMarkers: vertexMarkersGroup,
      edgeMarkers: edgeMarkersGroup,
      tileMeshes,
      isEditorEnabled: () => editorController?.isEnabled() ?? false,
    });
  } else {
    playController.updateDeps({
      buildingsManager: buildingsManager!,
      vertexMarkers: vertexMarkersGroup,
      edgeMarkers: edgeMarkersGroup,
      tileMeshes,
    });
  }
  playController.attach(view);

  document.getElementById("play-panel")!.style.display = "flex";
  loadingEl.style.display = "none";

  camera.position.set(0, 8, 8);
  controls.target.set(0, 0, 0);
  controls.update();
}

/**
 * Show/hide seat selectors to match the player count.
 */
function syncSeatRows(): void {
  const numPlayers = parseInt(
    (document.getElementById("play-num-players") as HTMLSelectElement).value,
    10
  );
  for (let i = 0; i < 4; i++) {
    const row = document.getElementById(`play-seat-${i}`)!
      .parentElement as HTMLElement;
    row.style.display = i < numPlayers ? "flex" : "none";
  }
}

/**
 * Main initialization.
 */
async function main(): Promise<void> {
  // Load board config first
  await loadConfig();

  initScene();
  initUI();

  document
    .getElementById("btn-play-start")!
    .addEventListener("click", () => startPlayGame());
  document
    .getElementById("play-num-players")!
    .addEventListener("change", syncSeatRows);
  syncSeatRows();

  await populateLogSelector();

  // Hide loading, show selector
  loadingEl.style.display = "none";
}

// Start the application
main().catch(console.error);
