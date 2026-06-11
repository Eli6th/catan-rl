/**
 * Interactive play mode: drives the Rust engine (catan-web server) so a
 * human can play full games against bots, inspect the complete god-view
 * state, and try to break the rules engine. Every click is submitted to the
 * engine verbatim — including illegal ones — and the engine's verdict is
 * shown. The server also cross-checks the RL action codec against the
 * engine on every request; any mismatch shows up as a red banner here.
 */

import * as THREE from "three";
import { BuildingsManager } from "./buildings";
import { PLAYER_COLOR_NAMES } from "./types";

const RES_NAMES = ["Wheat", "Sheep", "Wood", "Brick", "Stone"];
const DEV_NAMES = ["Knight", "VP", "RoadBuild", "YearOfPlenty", "Monopoly"];

interface ServerAction {
  type: string;
  player: number;
  [key: string]: unknown;
}

interface ValidAction {
  codec_id: number;
  label: string;
  action: ServerAction;
}

export interface PlayView {
  id: number;
  game_phase: string;
  turn_phase: string;
  current_player: number;
  seats: string[];
  roads_to_place: number;
  pending_discards: [number, number][];
  trade_offer: {
    proposer: number;
    give_amount: number;
    give_name: string;
    recv_name: string;
  } | null;
  trade_accepts: boolean[];
  trades_proposed_this_turn: number;
  state: {
    seed: number;
    num_players: number;
    turn: number;
    current_player: number;
    winner: number;
    tile_resources: number[];
    tile_numbers: number[];
    port_types: number[];
    vertices: number[];
    edges: number[];
    resources: number[][];
    bank: number[];
    victory_points: number[];
    robber_tile: number;
    dice_roll: number;
    has_rolled: boolean;
    longest_road_player: number;
    longest_road_length: number;
    largest_army_player: number;
    largest_army_size: number;
    dev_cards: number[][];
    knights_played: number[];
    dev_deck_remaining: number[];
    dev_deck_total_remaining: number;
    settlements_built: number[];
    cities_built: number[];
    roads_built: number[];
    road_lengths: number[];
    port_any: boolean[];
    port_resource: boolean[][];
  };
  valid_actions: ValidAction[];
  codec_check: { ok: boolean; errors: string[] };
  history_len: number;
  history_tail: ServerAction[];
  ok?: boolean;
  submitted?: ServerAction;
  bot_actions?: { player: number; action: ServerAction }[];
  bot_error?: string;
}

export interface PlayDeps {
  container: HTMLElement;
  camera: THREE.Camera;
  buildingsManager: BuildingsManager;
  vertexMarkers: THREE.Group | null;
  edgeMarkers: THREE.Group | null;
  tileMeshes: THREE.Mesh[];
  isEditorEnabled: () => boolean;
}

export async function createGame(opts: {
  numPlayers: number;
  seed?: number;
  seats: string[];
}): Promise<PlayView> {
  const body: Record<string, unknown> = {
    num_players: opts.numPlayers,
    seats: opts.seats,
  };
  if (opts.seed !== undefined) body.seed = opts.seed;
  const resp = await fetch("/rustapi/games", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`new game failed: ${await resp.text()}`);
  return resp.json();
}

export function labelAction(a: ServerAction): string {
  const r = (k: string) => RES_NAMES[a[k] as number] ?? "?";
  switch (a.type) {
    case "PlaceInitialSettlement":
      return `places settlement @ v${a.vertex}`;
    case "PlaceInitialRoad":
      return `places road @ e${a.edge}`;
    case "RollDice":
      return "rolls dice";
    case "BuildRoad":
      return `builds road @ e${a.edge}`;
    case "BuildSettlement":
      return `builds settlement @ v${a.vertex}`;
    case "BuildCity":
      return `builds city @ v${a.vertex}`;
    case "BuyDevCard":
      return "buys dev card";
    case "PlayKnight":
      return "plays Knight";
    case "PlayRoadBuilding":
      return "plays Road Building";
    case "PlayYearOfPlenty":
      return `plays Year of Plenty (${r("r1")}+${r("r2")})`;
    case "PlayMonopoly":
      return `plays Monopoly (${r("resource")})`;
    case "MoveRobber":
      return `moves robber → tile ${a.tile}`;
    case "StealResource":
      return (a.victim as number) < 0
        ? "steals from nobody"
        : `steals from P${a.victim}`;
    case "DiscardResource":
      return `discards ${r("resource")}`;
    case "TradeWithBank":
      return `bank trade ${r("give")} → ${r("recv")}`;
    case "ProposeTrade":
      return `offers ${a.give_amount} ${r("give")} for 1 ${r("recv")}`;
    case "RespondTrade":
      return a.accept ? "accepts trade" : "rejects trade";
    case "ConfirmTrade":
      return (a.partner as number) < 0
        ? "cancels trade"
        : `confirms trade with P${a.partner}`;
    case "EndTurn":
      return "ends turn";
    default:
      return a.type;
  }
}

function phaseHint(view: PlayView): string {
  if (view.game_phase === "SetupForward" || view.game_phase === "SetupBackward") {
    return "Setup: click a green vertex (settlement), then a green edge (road).";
  }
  if (view.game_phase === "Finished") {
    return `Game over — Player ${view.state.winner} wins!`;
  }
  switch (view.turn_phase) {
    case "MustRoll":
    case "PreRoll":
      return "Roll the dice (or play a Knight first).";
    case "RobberDiscard":
      return "Over 7 cards on a 7 — discard down to half.";
    case "RobberMove":
      return "Click a highlighted tile to move the robber.";
    case "RobberSteal":
      return "Pick a victim to steal from.";
    case "RoadBuilding":
      return `Place ${view.roads_to_place} free road(s) — click a green edge.`;
    case "TradeResponse":
      return "Respond to the open trade offer.";
    case "TradeChoose":
      return "Pick an accepting partner, or cancel.";
    default:
      return "Build, trade, play dev cards — or End Turn.";
  }
}

export class PlayController {
  private deps: PlayDeps;
  private view: PlayView | null = null;
  private raycaster = new THREE.Raycaster();
  private mouse = new THREE.Vector2();
  private downPos: { x: number; y: number } | null = null;
  private log: string[] = [];
  private busy = false;

  // Shared marker materials: dim for inert positions, green for legal ones.
  private dimVertexMat = new THREE.MeshStandardMaterial({
    color: 0xaaaa55,
    transparent: true,
    opacity: 0.18,
  });
  private validVertexMat = new THREE.MeshStandardMaterial({
    color: 0x22ff66,
    emissive: 0x116633,
  });
  private dimEdgeMat = new THREE.MeshStandardMaterial({
    color: 0x55aaaa,
    transparent: true,
    opacity: 0.18,
  });
  private validEdgeMat = new THREE.MeshStandardMaterial({
    color: 0x22ff66,
    emissive: 0x116633,
  });

  constructor(deps: PlayDeps) {
    this.deps = deps;
    deps.container.addEventListener("mousedown", (e) => {
      this.downPos = { x: e.clientX, y: e.clientY };
    });
    deps.container.addEventListener("mouseup", (e) => this.onMouseUp(e));
    this.bindTools();
  }

  attach(view: PlayView): void {
    this.log = [];
    this.applyView(view);
    this.maybeAutoBots();
  }

  /** Re-point at fresh scene objects after the board is rebuilt. */
  updateDeps(
    deps: Pick<
      PlayDeps,
      "buildingsManager" | "vertexMarkers" | "edgeMarkers" | "tileMeshes"
    >
  ): void {
    this.deps = { ...this.deps, ...deps };
  }

  getView(): PlayView | null {
    return this.view;
  }

  // ---------- server calls ----------

  private async post(path: string, body?: unknown): Promise<PlayView> {
    const resp = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    });
    if (!resp.ok) throw new Error(await resp.text());
    return resp.json();
  }

  async submitAction(action: ServerAction): Promise<void> {
    if (!this.view || this.busy) return;
    this.busy = true;
    try {
      const v = await this.post(`/rustapi/games/${this.view.id}/action`, {
        action,
      });
      this.recordResult(v);
      this.applyView(v);
      this.maybeAutoBots();
    } catch (e) {
      this.pushLog(`⚠ request failed: ${e}`);
      this.renderPanels();
    } finally {
      this.busy = false;
    }
  }

  async submitCodecId(id: number): Promise<void> {
    if (!this.view || this.busy) return;
    this.busy = true;
    try {
      const v = await this.post(`/rustapi/games/${this.view.id}/action-id`, {
        id,
      });
      this.recordResult(v, `codec ${id}: `);
      this.applyView(v);
      this.maybeAutoBots();
    } catch (e) {
      this.pushLog(`⚠ codec ${id} request failed: ${e}`);
      this.renderPanels();
    } finally {
      this.busy = false;
    }
  }

  async botStep(): Promise<void> {
    if (!this.view || this.busy) return;
    this.busy = true;
    try {
      const v = await this.post(`/rustapi/games/${this.view.id}/bot-step`);
      if (v.bot_error) this.pushLog(`🚨 BOT/ENGINE BUG: ${v.bot_error}`);
      for (const b of v.bot_actions ?? []) {
        this.pushLog(`🤖 P${b.player} ${labelAction(b.action)}`);
      }
      this.applyView(v);
    } catch (e) {
      this.pushLog(`⚠ bot step failed: ${e}`);
      this.renderPanels();
    } finally {
      this.busy = false;
    }
  }

  private recordResult(v: PlayView, prefix = ""): void {
    const what = v.submitted ? labelAction(v.submitted) : "?";
    const actor = v.submitted ? `P${v.submitted.player}` : "";
    if (v.ok) {
      this.pushLog(`✅ ${prefix}${actor} ${what}`);
    } else {
      this.pushLog(`⛔ ${prefix}engine REJECTED: ${actor} ${what}`);
    }
  }

  private maybeAutoBots(): void {
    const v = this.view;
    if (!v || v.game_phase === "Finished") return;
    if (v.seats[v.current_player] !== "human") {
      setTimeout(() => this.botStep(), 350);
    }
  }

  // ---------- board clicks ----------

  private onMouseUp(e: MouseEvent): void {
    if (!this.view || this.deps.isEditorEnabled()) return;
    if (!this.downPos) return;
    const dx = e.clientX - this.downPos.x;
    const dy = e.clientY - this.downPos.y;
    this.downPos = null;
    if (Math.hypot(dx, dy) > 5) return; // it was a camera drag
    if (e.button !== 0) return;

    const rect = this.deps.container.getBoundingClientRect();
    this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.mouse, this.deps.camera as THREE.PerspectiveCamera);

    const targets: THREE.Object3D[] = [];
    if (this.deps.vertexMarkers) targets.push(...this.deps.vertexMarkers.children);
    if (this.deps.edgeMarkers) targets.push(...this.deps.edgeMarkers.children);
    targets.push(...this.deps.tileMeshes);

    const hits = this.raycaster.intersectObjects(targets, false);
    if (hits.length === 0) return;
    const ud = hits[0].object.userData;

    if (ud.vertexIdx !== undefined) {
      this.clickVertex(ud.vertexIdx as number);
    } else if (ud.edgeIdx !== undefined) {
      this.clickEdge(ud.edgeIdx as number);
    } else if (ud.tileIdx !== undefined) {
      this.clickTile(ud.tileIdx as number);
    }
  }

  private clickVertex(vertex: number): void {
    const v = this.view!;
    // Prefer the engine's own legal action at this vertex; otherwise build
    // a best-guess action and let the engine judge it (break-testing).
    const match = v.valid_actions.find(
      (a) =>
        a.action.vertex === vertex &&
        ["PlaceInitialSettlement", "BuildSettlement", "BuildCity"].includes(
          a.action.type
        )
    );
    if (match) {
      this.submitAction(match.action);
      return;
    }
    const player = v.current_player;
    const inSetup = v.game_phase.startsWith("Setup");
    const owned = v.state.vertices[vertex];
    let type: string;
    if (inSetup) type = "PlaceInitialSettlement";
    else if (owned >= 0 && owned % 4 === player) type = "BuildCity";
    else type = "BuildSettlement";
    this.submitAction({ type, player, vertex });
  }

  private clickEdge(edge: number): void {
    const v = this.view!;
    const match = v.valid_actions.find(
      (a) =>
        a.action.edge === edge &&
        ["PlaceInitialRoad", "BuildRoad"].includes(a.action.type)
    );
    if (match) {
      this.submitAction(match.action);
      return;
    }
    const type = v.game_phase.startsWith("Setup")
      ? "PlaceInitialRoad"
      : "BuildRoad";
    this.submitAction({ type, player: v.current_player, edge });
  }

  private clickTile(tile: number): void {
    const v = this.view!;
    this.submitAction({ type: "MoveRobber", player: v.current_player, tile });
  }

  // ---------- rendering ----------

  private applyView(view: PlayView): void {
    this.view = view;
    this.deps.buildingsManager.updateFromState(
      view.state.vertices,
      view.state.edges,
      view.state.robber_tile
    );
    this.updateMarkers();
    this.renderPanels();
  }

  private updateMarkers(): void {
    const v = this.view!;
    const validVerts = new Set<number>();
    const validEdges = new Set<number>();
    const validTiles = new Set<number>();
    for (const a of v.valid_actions) {
      if (a.action.vertex !== undefined) validVerts.add(a.action.vertex as number);
      if (a.action.edge !== undefined) validEdges.add(a.action.edge as number);
      if (a.action.type === "MoveRobber") validTiles.add(a.action.tile as number);
    }

    if (this.deps.vertexMarkers) {
      for (const child of this.deps.vertexMarkers.children) {
        const mesh = child as THREE.Mesh;
        const idx = mesh.userData.vertexIdx as number;
        mesh.material = validVerts.has(idx) ? this.validVertexMat : this.dimVertexMat;
      }
    }
    if (this.deps.edgeMarkers) {
      for (const child of this.deps.edgeMarkers.children) {
        const mesh = child as THREE.Mesh;
        const idx = mesh.userData.edgeIdx as number;
        mesh.material = validEdges.has(idx) ? this.validEdgeMat : this.dimEdgeMat;
      }
    }
    for (const mesh of this.deps.tileMeshes) {
      const mat = mesh.material as THREE.MeshStandardMaterial;
      const idx = mesh.userData.tileIdx as number;
      mat.emissive.setHex(validTiles.has(idx) ? 0x335533 : 0x000000);
    }
  }

  private pushLog(line: string): void {
    this.log.push(line);
    if (this.log.length > 200) this.log.splice(0, this.log.length - 200);
  }

  private el(id: string): HTMLElement {
    return document.getElementById(id)!;
  }

  private renderPanels(): void {
    const v = this.view;
    if (!v) return;

    // Codec banner
    const banner = this.el("play-codec-banner");
    if (v.codec_check.ok) {
      banner.style.display = "none";
    } else {
      banner.style.display = "block";
      banner.textContent =
        "🚨 CODEC/ENGINE MISMATCH:\n" + v.codec_check.errors.join("\n");
    }

    // Status
    const actor = v.current_player;
    const seatKind = v.seats[actor];
    const dice = v.state.has_rolled ? ` 🎲 ${v.state.dice_roll}` : "";
    this.el("play-status").innerHTML = `
      <span class="player-${actor}">P${actor} (${PLAYER_COLOR_NAMES[actor]}, ${seatKind})</span>
      — ${v.game_phase} / ${v.turn_phase}${dice}<br>
      <span class="play-hint">${phaseHint(v)}</span>`;

    // Trade offer
    const tradeEl = this.el("play-trade");
    if (v.trade_offer) {
      const t = v.trade_offer;
      tradeEl.style.display = "block";
      tradeEl.textContent = `Open offer: P${t.proposer} gives ${t.give_amount} ${t.give_name} for 1 ${t.recv_name} (accepts: ${v.trade_accepts
        .map((a, i) => (a ? `P${i}` : null))
        .filter(Boolean)
        .join(", ") || "none yet"})`;
    } else {
      tradeEl.style.display = "none";
    }

    // Players god view
    let html = "";
    for (let p = 0; p < v.state.num_players; p++) {
      const res = v.state.resources[p];
      const dev = v.state.dev_cards[p];
      const devStr = dev
        .map((c, i) => (c > 0 ? `${DEV_NAMES[i]}×${c}` : null))
        .filter(Boolean)
        .join(" ");
      const ports =
        (v.state.port_any[p] ? ["3:1"] : [])
          .concat(
            v.state.port_resource[p]
              .map((has, i) => (has ? `2:1 ${RES_NAMES[i]}` : null))
              .filter(Boolean) as string[]
          )
          .join(", ") || "—";
      const badges =
        (v.state.longest_road_player === p ? " 🛣️" : "") +
        (v.state.largest_army_player === p ? " ⚔️" : "") +
        (v.state.winner === p ? " 👑" : "");
      html += `
        <div class="player-info ${p === actor ? "active" : ""}">
          <div class="player-name player-${p}">P${p} ${PLAYER_COLOR_NAMES[p]} (${v.seats[p]}) — ${v.state.victory_points[p]} VP${badges}</div>
          <div class="player-stats">
            ${res.map((c, i) => `${RES_NAMES[i].slice(0, 2)}:${c}`).join(" ")}
            (Σ${res.reduce((a, b) => a + b, 0)})<br>
            Dev: ${devStr || "—"} | Knights: ${v.state.knights_played[p]} | Road len: ${v.state.road_lengths[p]}<br>
            Built ${v.state.settlements_built[p]}s ${v.state.cities_built[p]}c ${v.state.roads_built[p]}r | Ports: ${ports}
          </div>
        </div>`;
    }
    this.el("play-players").innerHTML = html;

    // Bank + deck
    this.el("play-bank").innerHTML =
      `Bank: ${v.state.bank.map((c, i) => `${RES_NAMES[i].slice(0, 2)}:${c}`).join(" ")}<br>` +
      `Dev deck (${v.state.dev_deck_total_remaining} left): ${v.state.dev_deck_remaining
        .map((c, i) => `${DEV_NAMES[i]}×${c}`)
        .join(" ")}<br>` +
      `Seed ${v.state.seed} · Turn ${v.state.turn} · ${v.history_len} actions`;

    this.renderActions();
    this.el("play-log").innerHTML = this.log
      .slice(-60)
      .map((l) => `<div>${l}</div>`)
      .join("");
    this.el("play-log").scrollTop = this.el("play-log").scrollHeight;
  }

  private renderActions(): void {
    const v = this.view!;
    const boardTypes = new Set([
      "PlaceInitialSettlement",
      "PlaceInitialRoad",
      "BuildSettlement",
      "BuildCity",
      "BuildRoad",
      "MoveRobber",
    ]);
    const tradeTypes = new Set(["TradeWithBank", "ProposeTrade"]);

    const board = v.valid_actions.filter((a) => boardTypes.has(a.action.type));
    const trades = v.valid_actions.filter((a) => tradeTypes.has(a.action.type));
    const rest = v.valid_actions.filter(
      (a) => !boardTypes.has(a.action.type) && !tradeTypes.has(a.action.type)
    );

    const container = this.el("play-actions");
    container.innerHTML = "";

    const addButtons = (parent: HTMLElement, actions: ValidAction[]) => {
      for (const a of actions) {
        const btn = document.createElement("button");
        btn.textContent = a.label;
        btn.title = `codec id ${a.codec_id}`;
        btn.addEventListener("click", () => this.submitAction(a.action));
        parent.appendChild(btn);
      }
    };

    addButtons(container, rest);

    if (board.length > 0) {
      const det = document.createElement("details");
      det.innerHTML = `<summary>Board placements (${board.length}) — or just click the board</summary>`;
      addButtons(det, board);
      container.appendChild(det);
    }
    if (trades.length > 0) {
      const det = document.createElement("details");
      det.innerHTML = `<summary>Trades (${trades.length})</summary>`;
      addButtons(det, trades);
      container.appendChild(det);
    }
  }

  private bindTools(): void {
    this.el("btn-play-bots").addEventListener("click", () => this.botStep());
    this.el("btn-play-codec").addEventListener("click", () => {
      const input = this.el("play-codec-id") as HTMLInputElement;
      const id = parseInt(input.value, 10);
      if (!Number.isNaN(id)) this.submitCodecId(id);
    });
    this.el("btn-play-json").addEventListener("click", () => {
      const ta = this.el("play-json") as HTMLTextAreaElement;
      try {
        const action = JSON.parse(ta.value);
        this.submitAction(action);
      } catch (e) {
        this.pushLog(`⚠ bad JSON: ${e}`);
        this.renderPanels();
      }
    });
  }
}
