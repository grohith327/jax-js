// Make sure .ref move semantics are working correctly, and that arrays are
// freed at the right time.

import { grad, numpy as np } from "@jax-js/jax";
import { expect, suite, test } from "vitest";

suite("refcount through grad", () => {
  test("add and sum", () => {
    const f = (x: np.Array) => x.ref.add(x).sum();
    const df = grad(f);

    const x = np.array([1, 2, 3, 4]);
    expect(df(x).js()).toEqual([2, 2, 2, 2]);
    expect(() => x.dispose()).toThrowError(ReferenceError);
    expect(() => df(x).js()).toThrowError(ReferenceError);
  });

  // TODO(2025-11-13): Why is this test failing?
  // Also, check out grad-grad-jit test and update README
  test.skip("multiply and sum", () => {
    const f = (x: np.Array) => x.ref.mul(x).sum();
    const df = grad(f);

    const x = np.array([1, 2, 3, 4]);
    expect(df(x).js()).toEqual([2, 4, 6, 8]);
    expect(() => x.dispose()).toThrowError(ReferenceError);
    expect(() => df(x).js()).toThrowError(ReferenceError);
  });
});
