import { describe, expect, test } from "bun:test";
import { LifecycleScope, type RuntimeScheduler } from "../src/game/lifecycle";

class FakeScheduler implements RuntimeScheduler {
  readonly timeouts = new Map<number, () => void>();
  readonly intervals = new Map<number, () => void>();
  readonly frames = new Map<number, FrameRequestCallback>();
  private nextId = 1;

  setTimeout(callback: () => void): number {
    return this.save(this.timeouts, callback);
  }

  clearTimeout(id: number): void {
    this.timeouts.delete(id);
  }

  setInterval(callback: () => void): number {
    return this.save(this.intervals, callback);
  }

  clearInterval(id: number): void {
    this.intervals.delete(id);
  }

  requestAnimationFrame(callback: FrameRequestCallback): number {
    return this.save(this.frames, callback);
  }

  cancelAnimationFrame(id: number): void {
    this.frames.delete(id);
  }

  private save<T>(entries: Map<number, T>, value: T): number {
    const id = this.nextId;
    this.nextId += 1;
    entries.set(id, value);
    return id;
  }
}

describe("lifecycle scope", () => {
  test("Given listeners, timers, frames, and resources, when disposed, then every owned resource is released once", () => {
    const scheduler = new FakeScheduler();
    const scope = new LifecycleScope(scheduler);
    const target = new EventTarget();
    let eventCount = 0;
    let disposalCount = 0;
    scope.listen(target, "arena", () => {
      eventCount += 1;
    });
    scope.timeout(() => undefined, 50);
    scope.interval(() => undefined, 50);
    scope.frame(() => undefined);
    scope.own(() => {
      disposalCount += 1;
    });
    target.dispatchEvent(new Event("arena"));

    scope.dispose();
    scope.dispose();
    target.dispatchEvent(new Event("arena"));

    expect(eventCount).toBe(1);
    expect(disposalCount).toBe(1);
    expect(scheduler.timeouts.size).toBe(0);
    expect(scheduler.intervals.size).toBe(0);
    expect(scheduler.frames.size).toBe(0);
  });

  test("Given a disposed scope, when new ownership is attempted, then it is rejected", () => {
    const scope = new LifecycleScope(new FakeScheduler());
    scope.dispose();

    expect(() => scope.own(() => undefined)).toThrow("disposed");
  });
});
