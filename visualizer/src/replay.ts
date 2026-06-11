/**
 * Replay controller for Catan game visualization.
 * Manages game state and action playback.
 */

import {
  LogData,
  Action,
  ActionType,
  ACTION_NAMES,
  RESOURCE_NAMES,
  ResourceType,
  NUM_VERTICES,
  NUM_EDGES,
} from "./types";
import { BuildingsManager } from "./buildings";

// Initial game state template
interface ReplayState {
  turn: number;
  currentPlayer: number;
  phase: "setup_forward" | "setup_backward" | "playing" | "finished";
  vertices: number[]; // -1 = empty, 0-3 = settlement, 4-7 = city
  edges: number[]; // -1 = empty, 0-3 = road by player
  robberTile: number;
  resources: number[][]; // [player][resource_type]
  victoryPoints: number[];
  longestRoadPlayer: number;
  largestArmyPlayer: number;
  winner: number;
}

export type ReplayEventCallback = (event: ReplayEvent) => void;

export interface ReplayEvent {
  type: "state_change" | "action_executed" | "game_end";
  action?: Action;
  actionIndex?: number;
  state: ReplayState;
}

/**
 * Replay controller manages stepping through game actions.
 */
export class ReplayController {
  private logData: LogData;
  private actionIndex: number = 0;
  private state: ReplayState;
  private buildingsManager: BuildingsManager | null = null;
  private eventCallback: ReplayEventCallback | null = null;

  // Playback state
  private isPlaying: boolean = false;
  private playbackSpeed: number = 1.0;
  private playbackInterval: number | null = null;

  constructor(logData: LogData) {
    this.logData = logData;
    this.state = this.createInitialState();
  }

  setBuildingsManager(manager: BuildingsManager): void {
    this.buildingsManager = manager;
  }

  setEventCallback(callback: ReplayEventCallback): void {
    this.eventCallback = callback;
  }

  private createInitialState(): ReplayState {
    const numPlayers = this.logData.num_players;

    // Find desert tile for initial robber position
    let robberTile = 0;
    for (let i = 0; i < this.logData.tile_resources.length; i++) {
      if (this.logData.tile_resources[i] === ResourceType.DESERT) {
        robberTile = i;
        break;
      }
    }

    return {
      turn: 0,
      currentPlayer: 0,
      phase: "setup_forward",
      vertices: new Array(NUM_VERTICES).fill(-1),
      edges: new Array(NUM_EDGES).fill(-1),
      robberTile,
      resources: Array.from({ length: numPlayers }, () => [0, 0, 0, 0, 0]),
      victoryPoints: new Array(numPlayers).fill(0),
      longestRoadPlayer: -1,
      largestArmyPlayer: -1,
      winner: -1,
    };
  }

  /**
   * Reset to initial state.
   */
  reset(): void {
    this.stop();
    this.actionIndex = 0;
    this.state = this.createInitialState();
    this.updateVisuals();
    this.emitEvent("state_change");
  }

  /**
   * Get current state.
   */
  getState(): ReplayState {
    return { ...this.state };
  }

  /**
   * Get current action index.
   */
  getActionIndex(): number {
    return this.actionIndex;
  }

  /**
   * Get total number of actions.
   */
  getTotalActions(): number {
    return this.logData.actions.length;
  }

  /**
   * Get the log data.
   */
  getLogData(): LogData {
    return this.logData;
  }

  /**
   * Check if at end of replay.
   */
  isAtEnd(): boolean {
    return this.actionIndex >= this.logData.actions.length;
  }

  /**
   * Check if at start of replay.
   */
  isAtStart(): boolean {
    return this.actionIndex === 0;
  }

  /**
   * Step forward one action.
   */
  stepForward(): Action | null {
    if (this.isAtEnd()) {
      this.stop();
      return null;
    }

    const action = this.logData.actions[this.actionIndex];
    this.executeAction(action);
    this.actionIndex++;

    this.emitEvent("action_executed", action);

    if (this.isAtEnd() || this.state.winner >= 0) {
      this.stop();
      this.emitEvent("game_end");
    }

    return action;
  }

  /**
   * Step backward one action (replay from start to actionIndex - 1).
   */
  stepBackward(): boolean {
    if (this.isAtStart()) {
      return false;
    }

    const targetIndex = this.actionIndex - 1;
    this.reset();

    while (this.actionIndex < targetIndex) {
      const action = this.logData.actions[this.actionIndex];
      this.executeAction(action);
      this.actionIndex++;
    }

    this.emitEvent("state_change");
    return true;
  }

  /**
   * Jump to a specific action index.
   */
  jumpToAction(targetIndex: number): void {
    if (targetIndex < 0 || targetIndex > this.logData.actions.length) {
      return;
    }

    this.stop();

    if (targetIndex < this.actionIndex) {
      this.reset();
    }

    while (this.actionIndex < targetIndex) {
      const action = this.logData.actions[this.actionIndex];
      this.executeAction(action);
      this.actionIndex++;
    }

    this.updateVisuals();
    this.emitEvent("state_change");
  }

  /**
   * Start auto-playback.
   */
  play(): void {
    if (this.isPlaying) return;
    if (this.isAtEnd()) {
      this.reset();
    }

    this.isPlaying = true;
    this.scheduleNextAction();
  }

  /**
   * Pause playback.
   */
  pause(): void {
    this.isPlaying = false;
    if (this.playbackInterval !== null) {
      clearTimeout(this.playbackInterval);
      this.playbackInterval = null;
    }
  }

  /**
   * Stop playback.
   */
  stop(): void {
    this.pause();
  }

  /**
   * Toggle play/pause.
   */
  togglePlayPause(): void {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  /**
   * Check if currently playing.
   */
  getIsPlaying(): boolean {
    return this.isPlaying;
  }

  /**
   * Set playback speed.
   */
  setPlaybackSpeed(speed: number): void {
    this.playbackSpeed = Math.max(0.1, Math.min(10, speed));
  }

  private scheduleNextAction(): void {
    if (!this.isPlaying || this.isAtEnd()) {
      this.stop();
      return;
    }

    const delay = 500 / this.playbackSpeed;
    this.playbackInterval = window.setTimeout(() => {
      this.stepForward();
      this.scheduleNextAction();
    }, delay);
  }

  /**
   * Execute an action and update state.
   */
  private executeAction(action: Action): void {
    const { player, payload } = action;
    const actionType = ActionType[action.type];

    switch (actionType) {
      case ActionType.PLACE_INITIAL_SETTLEMENT:
      case ActionType.BUILD_SETTLEMENT:
        if (payload && payload.length > 0) {
          const vertex = payload[0];
          this.state.vertices[vertex] = player;
          this.buildingsManager?.placeSettlement(vertex, player);
          this.updateVictoryPoints(player);
        }
        break;

      case ActionType.PLACE_INITIAL_ROAD:
      case ActionType.BUILD_ROAD:
        if (payload && payload.length > 0) {
          const edge = payload[0];
          this.state.edges[edge] = player;
          this.buildingsManager?.placeRoad(edge, player);
        }
        break;

      case ActionType.BUILD_CITY:
        if (payload && payload.length > 0) {
          const vertex = payload[0];
          this.state.vertices[vertex] = player + 4; // 4-7 = city
          this.buildingsManager?.placeCity(vertex, player);
          this.updateVictoryPoints(player);
        }
        break;

      case ActionType.MOVE_ROBBER:
        if (payload && payload.length > 0) {
          const tile = payload[0];
          this.state.robberTile = tile;
          this.buildingsManager?.moveRobber(tile);
        }
        break;

      case ActionType.END_TURN:
        this.state.turn++;
        this.state.currentPlayer =
          (this.state.currentPlayer + 1) % this.logData.num_players;
        break;

      case ActionType.ROLL_DICE:
        // Dice rolling doesn't change visual state
        break;

      // Other actions don't affect visuals directly
      default:
        break;
    }

    // Check for phase transitions
    this.checkPhaseTransition();
  }

  private checkPhaseTransition(): void {
    // Count settlements and roads per player for setup phase tracking
    const settlementCounts = new Array(this.logData.num_players).fill(0);
    const roadCounts = new Array(this.logData.num_players).fill(0);

    for (const val of this.state.vertices) {
      if (val >= 0 && val < 4) {
        settlementCounts[val]++;
      } else if (val >= 4) {
        settlementCounts[val - 4]++;
      }
    }

    for (const val of this.state.edges) {
      if (val >= 0) {
        roadCounts[val]++;
      }
    }

    // Check if all players have 2 settlements and 2 roads
    const allSetup =
      settlementCounts.every((c) => c >= 2) && roadCounts.every((c) => c >= 2);

    if (
      allSetup &&
      this.state.phase !== "playing" &&
      this.state.phase !== "finished"
    ) {
      this.state.phase = "playing";
    }
  }

  private updateVictoryPoints(player: number): void {
    let vp = 0;

    // Count settlements (1 VP each)
    for (const val of this.state.vertices) {
      if (val === player) vp++;
      if (val === player + 4) vp += 2; // Cities worth 2
    }

    // Longest road (2 VP)
    if (this.state.longestRoadPlayer === player) vp += 2;

    // Largest army (2 VP)
    if (this.state.largestArmyPlayer === player) vp += 2;

    this.state.victoryPoints[player] = vp;

    // Check for winner
    if (vp >= 10) {
      this.state.winner = player;
      this.state.phase = "finished";
    }
  }

  private updateVisuals(): void {
    this.buildingsManager?.updateFromState(
      this.state.vertices,
      this.state.edges,
      this.state.robberTile
    );
  }

  private emitEvent(type: ReplayEvent["type"], action?: Action): void {
    if (this.eventCallback) {
      this.eventCallback({
        type,
        action,
        actionIndex: this.actionIndex,
        state: this.getState(),
      });
    }
  }

  /**
   * Get a description of an action.
   */
  static describeAction(action: Action): string {
    const actionType = ActionType[action.type];
    const actionName = ACTION_NAMES[actionType] || action.type;
    let description = `Player ${action.player}: ${actionName}`;

    if (action.payload && action.payload.length > 0) {
      switch (actionType) {
        case ActionType.PLACE_INITIAL_SETTLEMENT:
        case ActionType.BUILD_SETTLEMENT:
          description += ` at vertex ${action.payload[0]}`;
          break;
        case ActionType.PLACE_INITIAL_ROAD:
        case ActionType.BUILD_ROAD:
          description += ` at edge ${action.payload[0]}`;
          break;
        case ActionType.BUILD_CITY:
          description += ` at vertex ${action.payload[0]}`;
          break;
        case ActionType.MOVE_ROBBER:
          description += ` to tile ${action.payload[0]}`;
          break;
        case ActionType.STEAL_RESOURCE:
          if (action.payload[0] >= 0) {
            description += ` from Player ${action.payload[0]}`;
          }
          break;
        case ActionType.TRADE_WITH_BANK:
          if (action.payload.length >= 2) {
            const give = RESOURCE_NAMES[action.payload[0] as ResourceType];
            const recv = RESOURCE_NAMES[action.payload[1] as ResourceType];
            description += ` (${give} → ${recv})`;
          }
          break;
      }
    }

    return description;
  }
}
