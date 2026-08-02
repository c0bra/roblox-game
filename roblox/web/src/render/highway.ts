import type { ChartNote, Lane } from "../data/level";

const laneColors = ["#55d8ff", "#ffe08a", "#bf83ff"] as const;
const travelTime = 2.25;

export class HighwayRenderer {
  private readonly context: CanvasRenderingContext2D;
  private width = 0;
  private height = 0;
  private pixelRatio = 1;

  constructor(private readonly canvas: HTMLCanvasElement) {
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas 2D is unavailable");
    this.context = context;
    this.resize();
  }

  resize(): void {
    const rect = this.canvas.getBoundingClientRect();
    this.pixelRatio = Math.min(window.devicePixelRatio, 2);
    this.width = Math.max(1, rect.width);
    this.height = Math.max(1, rect.height);
    this.canvas.width = Math.round(this.width * this.pixelRatio);
    this.canvas.height = Math.round(this.height * this.pixelRatio);
    this.context.setTransform(this.pixelRatio, 0, 0, this.pixelRatio, 0, 0);
  }

  draw(
    notes: readonly ChartNote[],
    judged: ReadonlySet<number>,
    held: ReadonlySet<number>,
    songTime: number,
  ): void {
    const context = this.context;
    context.clearRect(0, 0, this.width, this.height);
    const top = this.height * 0.04;
    const strike = this.height * 0.9;
    const center = this.width / 2;
    const topWidth = this.width * 0.22;
    const bottomWidth = this.width * 0.92;
    context.fillStyle = "rgba(4, 6, 14, .58)";
    context.beginPath();
    context.moveTo(center - topWidth / 2, top);
    context.lineTo(center + topWidth / 2, top);
    context.lineTo(center + bottomWidth / 2, this.height);
    context.lineTo(center - bottomWidth / 2, this.height);
    context.fill();
    this.drawLaneLines(top, strike, topWidth, bottomWidth);
    this.drawStrikeLine(strike);
    for (const [index, note] of notes.entries()) {
      const isHeld = held.has(index);
      if (judged.has(index) && !isHeld) continue;
      const until = note.time - songTime;
      if (until > travelTime || until < -note.duration - 0.18) continue;
      const progress = Math.min(1, 1 - until / travelTime);
      if (note.duration > 0) {
        const endProgress = Math.min(
          1,
          1 - (note.time + note.duration - songTime) / travelTime,
        );
        this.drawSustain(
          note.lane,
          progress,
          endProgress,
          isHeld,
          top,
          strike,
          topWidth,
          bottomWidth,
        );
      }
      this.drawNote(note.lane, progress, top, strike, topWidth, bottomWidth);
    }
  }

  private drawLaneLines(
    top: number,
    strike: number,
    topWidth: number,
    bottomWidth: number,
  ): void {
    const context = this.context;
    context.strokeStyle = "rgba(163, 211, 255, .22)";
    context.lineWidth = 1;
    for (let lane = 0; lane <= 3; lane += 1) {
      const fraction = lane / 3 - 0.5;
      context.beginPath();
      context.moveTo(this.width / 2 + fraction * topWidth, top);
      context.lineTo(this.width / 2 + fraction * bottomWidth, strike);
      context.stroke();
    }
  }

  private drawStrikeLine(y: number): void {
    const context = this.context;
    context.shadowColor = "#7ce8ff";
    context.shadowBlur = 15;
    context.strokeStyle = "#dffbff";
    context.lineWidth = 3;
    context.beginPath();
    context.moveTo(this.width * 0.04, y);
    context.lineTo(this.width * 0.96, y);
    context.stroke();
    context.shadowBlur = 0;
  }

  private drawNote(
    lane: Lane,
    progress: number,
    top: number,
    strike: number,
    topWidth: number,
    bottomWidth: number,
  ): void {
    const point = this.notePoint(
      lane,
      progress,
      top,
      strike,
      topWidth,
      bottomWidth,
    );
    const { x, y, laneWidth } = point;
    const radius = Math.max(8, laneWidth * 0.23);
    const context = this.context;
    context.fillStyle = laneColors[lane];
    context.shadowColor = laneColors[lane];
    context.shadowBlur = 14;
    context.beginPath();
    if (lane === 0) {
      context.arc(x, y, radius * 0.72, 0, Math.PI * 2);
    } else if (lane === 1) {
      context.moveTo(x, y - radius * 0.78);
      context.lineTo(x + radius, y);
      context.lineTo(x, y + radius * 0.78);
      context.lineTo(x - radius, y);
      context.closePath();
    } else {
      context.moveTo(x, y - radius * 0.78);
      context.lineTo(x + radius, y + radius * 0.68);
      context.lineTo(x - radius, y + radius * 0.68);
      context.closePath();
    }
    context.fill();
    context.fillStyle = "rgba(255,255,255,.72)";
    if (lane === 0) context.fillRect(x - radius * 0.4, y - 1, radius * 0.8, 2);
    context.shadowBlur = 0;
  }

  private drawSustain(
    lane: Lane,
    headProgress: number,
    endProgress: number,
    held: boolean,
    top: number,
    strike: number,
    topWidth: number,
    bottomWidth: number,
  ): void {
    const head = this.notePoint(
      lane,
      headProgress,
      top,
      strike,
      topWidth,
      bottomWidth,
    );
    const end = this.notePoint(
      lane,
      endProgress,
      top,
      strike,
      topWidth,
      bottomWidth,
    );
    const context = this.context;
    context.save();
    context.globalAlpha = held ? 0.92 : 0.52;
    context.fillStyle = laneColors[lane];
    context.shadowColor = laneColors[lane];
    context.shadowBlur = held ? 18 : 8;
    context.beginPath();
    context.moveTo(end.x - end.laneWidth * 0.12, end.y);
    context.lineTo(end.x + end.laneWidth * 0.12, end.y);
    context.lineTo(head.x + head.laneWidth * 0.16, head.y);
    context.lineTo(head.x - head.laneWidth * 0.16, head.y);
    context.closePath();
    context.fill();
    context.restore();
  }

  private notePoint(
    lane: Lane,
    progress: number,
    top: number,
    strike: number,
    topWidth: number,
    bottomWidth: number,
  ): { readonly x: number; readonly y: number; readonly laneWidth: number } {
    const normalized = Math.max(0, Math.min(1, progress));
    const eased = normalized * normalized;
    const y = top + (strike - top) * eased;
    const roadWidth = topWidth + (bottomWidth - topWidth) * eased;
    const laneWidth = roadWidth / 3;
    return {
      x: this.width / 2 + (lane - 1) * laneWidth,
      y,
      laneWidth,
    };
  }
}
