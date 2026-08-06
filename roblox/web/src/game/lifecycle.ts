export interface RuntimeScheduler {
  setTimeout(callback: () => void, delayMilliseconds: number): number;
  clearTimeout(id: number): void;
  setInterval(callback: () => void, delayMilliseconds: number): number;
  clearInterval(id: number): void;
  requestAnimationFrame(callback: FrameRequestCallback): number;
  cancelAnimationFrame(id: number): void;
}

export interface GameModeController {
  mount(): Promise<void>;
  dispose(): void;
}

export const browserScheduler: RuntimeScheduler = {
  setTimeout: (callback, delayMilliseconds) =>
    window.setTimeout(callback, delayMilliseconds),
  clearTimeout: (id) => window.clearTimeout(id),
  setInterval: (callback, delayMilliseconds) =>
    window.setInterval(callback, delayMilliseconds),
  clearInterval: (id) => window.clearInterval(id),
  requestAnimationFrame: (callback) => window.requestAnimationFrame(callback),
  cancelAnimationFrame: (id) => window.cancelAnimationFrame(id),
};

export class LifecycleScope {
  private disposers: Array<() => void> = [];
  private disposed = false;

  constructor(
    private readonly scheduler: RuntimeScheduler = browserScheduler,
  ) {}

  own(dispose: () => void): void {
    if (this.disposed) throw new Error("Lifecycle scope is disposed");
    this.disposers.push(dispose);
  }

  listen(
    target: EventTarget,
    type: string,
    listener: EventListenerOrEventListenerObject,
    options?: AddEventListenerOptions | boolean,
  ): void {
    target.addEventListener(type, listener, options);
    this.own(() => target.removeEventListener(type, listener, options));
  }

  timeout(callback: () => void, delayMilliseconds: number): number {
    const id = this.scheduler.setTimeout(callback, delayMilliseconds);
    this.own(() => this.scheduler.clearTimeout(id));
    return id;
  }

  interval(callback: () => void, delayMilliseconds: number): number {
    const id = this.scheduler.setInterval(callback, delayMilliseconds);
    this.own(() => this.scheduler.clearInterval(id));
    return id;
  }

  frame(callback: FrameRequestCallback): number {
    const id = this.scheduler.requestAnimationFrame(callback);
    this.own(() => this.scheduler.cancelAnimationFrame(id));
    return id;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const dispose of this.disposers.reverse()) dispose();
    this.disposers = [];
  }
}
