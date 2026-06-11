/**
 * Editor module for the Catan 3D visualizer.
 * Provides interactive editing of vertex and edge positions.
 */

import * as THREE from "three";
import {
  getConfig,
  updateConfig,
  setVertexOverride,
  setEdgeOverride,
  getVertexOverride,
  getEdgeOverride,
  resetConfig,
  downloadConfig,
  copyConfigToClipboard,
  BoardConfig,
} from "./config";
import { TILE_TOP_HEIGHT } from "./board";

export type ConfigChangeCallback = (config: BoardConfig) => void;

interface DragState {
  active: boolean;
  type: "vertex" | "edge" | null;
  index: number;
  startPosition: THREE.Vector3;
  startMouse: THREE.Vector2;
}

/**
 * Editor controller for interactive position editing.
 */
export class EditorController {
  private enabled: boolean = false;
  private container: HTMLElement;
  private camera: THREE.Camera;

  private raycaster: THREE.Raycaster;
  private mouse: THREE.Vector2;

  private vertexMarkers: THREE.Group | null = null;
  private edgeMarkers: THREE.Group | null = null;

  private selectedObject: THREE.Mesh | null = null;
  private highlightMaterial: THREE.MeshStandardMaterial;
  private originalMaterials: Map<THREE.Mesh, THREE.Material> = new Map();

  private dragState: DragState = {
    active: false,
    type: null,
    index: -1,
    startPosition: new THREE.Vector3(),
    startMouse: new THREE.Vector2(),
  };

  private plane: THREE.Plane;
  private intersection: THREE.Vector3;

  private configChangeCallback: ConfigChangeCallback | null = null;

  // UI elements
  private panel: HTMLElement;
  private toggleBtn: HTMLElement;
  private selectedInfo: HTMLElement;

  constructor(
    container: HTMLElement,
    camera: THREE.Camera,
    _scene: THREE.Scene
  ) {
    this.container = container;
    this.camera = camera;

    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    // Horizontal plane for dragging at tile height
    this.plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -TILE_TOP_HEIGHT - 0.1);
    this.intersection = new THREE.Vector3();

    // Highlight material for selected objects
    this.highlightMaterial = new THREE.MeshStandardMaterial({
      color: 0xff00ff,
      emissive: 0x660066,
    });

    // Get UI elements
    this.panel = document.getElementById("debug-panel")!;
    this.toggleBtn = document.getElementById("debug-toggle")!;
    this.selectedInfo = document.getElementById("selected-info")!;

    this.setupEventListeners();
    this.setupUIControls();
  }

  /**
   * Set callback for config changes.
   */
  setConfigChangeCallback(callback: ConfigChangeCallback): void {
    this.configChangeCallback = callback;
  }

  /**
   * Set the vertex and edge marker groups for raycasting.
   */
  setMarkers(vertexMarkers: THREE.Group, edgeMarkers: THREE.Group): void {
    this.vertexMarkers = vertexMarkers;
    this.edgeMarkers = edgeMarkers;
  }

  /**
   * Check if editor is enabled.
   */
  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Toggle editor mode.
   */
  toggle(): void {
    this.enabled = !this.enabled;
    this.panel.classList.toggle("visible", this.enabled);

    if (!this.enabled) {
      this.deselectObject();
    }

    // Sync UI with current config
    if (this.enabled) {
      this.syncUIWithConfig();
    }
  }

  /**
   * Enable editor mode.
   */
  enable(): void {
    if (!this.enabled) {
      this.toggle();
    }
  }

  /**
   * Disable editor mode.
   */
  disable(): void {
    if (this.enabled) {
      this.toggle();
    }
  }

  private setupEventListeners(): void {
    // Mouse events for raycasting and dragging
    this.container.addEventListener("mousedown", this.onMouseDown.bind(this));
    this.container.addEventListener("mousemove", this.onMouseMove.bind(this));
    this.container.addEventListener("mouseup", this.onMouseUp.bind(this));

    // Keyboard shortcut
    document.addEventListener("keydown", (e) => {
      if (e.key === "d" || e.key === "D") {
        // Don't toggle if user is typing in an input
        if ((e.target as HTMLElement).tagName !== "INPUT") {
          this.toggle();
        }
      }

      // Escape to deselect
      if (e.key === "Escape" && this.enabled) {
        this.deselectObject();
      }
    });
  }

  private setupUIControls(): void {
    // Toggle button
    this.toggleBtn.addEventListener("click", () => this.toggle());

    // Close button
    document.getElementById("debug-close")!.addEventListener("click", () => {
      this.disable();
    });

    // Sliders
    this.setupSlider("slider-gap-horiz", "val-gap-horiz", "tileGapHoriz");
    this.setupSlider("slider-gap-vert", "val-gap-vert", "tileGapVert");
    this.setupSlider("slider-vertex-scale", "val-vertex-scale", "vertexRadiusScale");
    this.setupSlider("slider-edge-scale", "val-edge-scale", "edgeLengthScale");

    // Action buttons
    document.getElementById("btn-export-config")!.addEventListener("click", () => {
      downloadConfig();
    });

    document.getElementById("btn-copy-config")!.addEventListener("click", async () => {
      await copyConfigToClipboard();
      // Brief visual feedback
      const btn = document.getElementById("btn-copy-config")!;
      const original = btn.textContent;
      btn.textContent = "✓ Copied!";
      setTimeout(() => {
        btn.textContent = original;
      }, 1500);
    });

    document.getElementById("btn-reset-config")!.addEventListener("click", () => {
      resetConfig();
      this.syncUIWithConfig();
      this.notifyConfigChange();
    });
  }

  private setupSlider(
    sliderId: string,
    valueId: string,
    configKey: keyof BoardConfig
  ): void {
    const slider = document.getElementById(sliderId) as HTMLInputElement;
    const valueSpan = document.getElementById(valueId)!;

    slider.addEventListener("input", () => {
      const value = parseFloat(slider.value);
      valueSpan.textContent = value.toFixed(2);
      updateConfig({ [configKey]: value });
      this.notifyConfigChange();
    });
  }

  private syncUIWithConfig(): void {
    const config = getConfig();

    this.setSliderValue("slider-gap-horiz", "val-gap-horiz", config.tileGapHoriz);
    this.setSliderValue("slider-gap-vert", "val-gap-vert", config.tileGapVert);
    this.setSliderValue("slider-vertex-scale", "val-vertex-scale", config.vertexRadiusScale);
    this.setSliderValue("slider-edge-scale", "val-edge-scale", config.edgeLengthScale);
  }

  private setSliderValue(sliderId: string, valueId: string, value: number): void {
    const slider = document.getElementById(sliderId) as HTMLInputElement;
    const valueSpan = document.getElementById(valueId)!;
    slider.value = value.toString();
    valueSpan.textContent = value.toFixed(2);
  }

  private notifyConfigChange(): void {
    if (this.configChangeCallback) {
      this.configChangeCallback(getConfig());
    }
  }

  private updateMouse(event: MouseEvent): void {
    const rect = this.container.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  private onMouseDown(event: MouseEvent): void {
    if (!this.enabled) return;
    if (event.button !== 0) return; // Left click only

    this.updateMouse(event);
    this.raycaster.setFromCamera(this.mouse, this.camera);

    // Check for intersections with markers
    const objects: THREE.Object3D[] = [];
    if (this.vertexMarkers) {
      objects.push(...this.vertexMarkers.children);
    }
    if (this.edgeMarkers) {
      objects.push(...this.edgeMarkers.children);
    }

    const intersects = this.raycaster.intersectObjects(objects);

    if (intersects.length > 0) {
      const hit = intersects[0].object as THREE.Mesh;

      // Select the object
      this.selectObject(hit);

      // Start dragging
      this.dragState.active = true;
      this.dragState.startPosition.copy(hit.position);
      this.dragState.startMouse.copy(this.mouse);

      // Determine type and index
      if (hit.userData.vertexIdx !== undefined) {
        this.dragState.type = "vertex";
        this.dragState.index = hit.userData.vertexIdx;
      } else if (hit.userData.edgeIdx !== undefined) {
        this.dragState.type = "edge";
        this.dragState.index = hit.userData.edgeIdx;
      }
    }
  }

  private onMouseMove(event: MouseEvent): void {
    if (!this.enabled || !this.dragState.active || !this.selectedObject) return;

    this.updateMouse(event);
    this.raycaster.setFromCamera(this.mouse, this.camera);

    // Find intersection with horizontal plane
    if (this.raycaster.ray.intersectPlane(this.plane, this.intersection)) {
      // Calculate offset from original position
      const dx = this.intersection.x - this.dragState.startPosition.x;
      const dz = this.intersection.z - this.dragState.startPosition.z;

      // Move the marker
      this.selectedObject.position.x = this.dragState.startPosition.x + dx;
      this.selectedObject.position.z = this.dragState.startPosition.z + dz;

      // Update info display
      this.updateSelectedInfo();
    }
  }

  private onMouseUp(_event: MouseEvent): void {
    if (!this.enabled || !this.dragState.active) return;

    if (this.selectedObject && this.dragState.type !== null) {
      // Calculate final offset from original computed position
      const dx = this.selectedObject.position.x - this.dragState.startPosition.x;
      const dz = this.selectedObject.position.z - this.dragState.startPosition.z;

      // Only save if there was actual movement
      if (Math.abs(dx) > 0.001 || Math.abs(dz) > 0.001) {
        if (this.dragState.type === "vertex") {
          // Get existing override and add to it
          const existing = getVertexOverride(this.dragState.index);
          const newDx = (existing?.dx || 0) + dx;
          const newDz = (existing?.dz || 0) + dz;
          setVertexOverride(this.dragState.index, newDx, newDz);
        } else if (this.dragState.type === "edge") {
          const existing = getEdgeOverride(this.dragState.index);
          const newDx = (existing?.dx || 0) + dx;
          const newDz = (existing?.dz || 0) + dz;
          setEdgeOverride(this.dragState.index, newDx, newDz, existing?.rotation || 0);
        }

        this.notifyConfigChange();
      }
    }

    this.dragState.active = false;
  }

  private selectObject(mesh: THREE.Mesh): void {
    // Deselect previous
    this.deselectObject();

    // Store original material and apply highlight
    this.selectedObject = mesh;
    this.originalMaterials.set(mesh, mesh.material as THREE.Material);
    mesh.material = this.highlightMaterial;

    this.updateSelectedInfo();
  }

  private deselectObject(): void {
    if (this.selectedObject) {
      const original = this.originalMaterials.get(this.selectedObject);
      if (original) {
        this.selectedObject.material = original;
        this.originalMaterials.delete(this.selectedObject);
      }
      this.selectedObject = null;
    }

    this.selectedInfo.innerHTML = `
      Click a <strong>vertex</strong> (yellow) or <strong>edge</strong> (cyan) marker to select it, then drag to adjust position.
    `;
  }

  private updateSelectedInfo(): void {
    if (!this.selectedObject) return;

    const pos = this.selectedObject.position;

    if (this.selectedObject.userData.vertexIdx !== undefined) {
      const idx = this.selectedObject.userData.vertexIdx;
      const override = getVertexOverride(idx);
      const dx = this.selectedObject.position.x - this.dragState.startPosition.x + (override?.dx || 0);
      const dz = this.selectedObject.position.z - this.dragState.startPosition.z + (override?.dz || 0);

      this.selectedInfo.innerHTML = `
        <strong>Vertex ${idx}</strong><br>
        Position: (${pos.x.toFixed(2)}, ${pos.z.toFixed(2)})<br>
        Offset: dx=${dx.toFixed(3)}, dz=${dz.toFixed(3)}
      `;
    } else if (this.selectedObject.userData.edgeIdx !== undefined) {
      const idx = this.selectedObject.userData.edgeIdx;
      const vertices = this.selectedObject.userData.vertices;
      const override = getEdgeOverride(idx);
      const dx = this.selectedObject.position.x - this.dragState.startPosition.x + (override?.dx || 0);
      const dz = this.selectedObject.position.z - this.dragState.startPosition.z + (override?.dz || 0);

      this.selectedInfo.innerHTML = `
        <strong>Edge ${idx}</strong> (${vertices[0]}-${vertices[1]})<br>
        Position: (${pos.x.toFixed(2)}, ${pos.z.toFixed(2)})<br>
        Offset: dx=${dx.toFixed(3)}, dz=${dz.toFixed(3)}
      `;
    }
  }
}

