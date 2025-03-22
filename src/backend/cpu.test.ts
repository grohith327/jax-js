import { expect, test } from "vitest";
import { accessorAlu, getBackend } from "../backend";
import { ShapeTracker } from "../shape";
import { AluExp, DType } from "../alu";

test("can run cpu operations", async ({ skip }) => {
  const backend = await getBackend("cpu");
  if (!backend) {
    // Not all environments support WebGPU, especially in CI.
    return skip();
  }

  const shape = ShapeTracker.fromShape([3]);
  const a = backend.malloc(3 * 4, new Float32Array([1, 2, 3]).buffer);
  const b = backend.malloc(3 * 4, new Float32Array([4, 5, 6]).buffer);
  const c = backend.malloc(3 * 4);

  try {
    const gidx = AluExp.special(DType.Int32, "gidx", 3);
    const arg1 = accessorAlu(0, shape, gidx);
    const arg2 = accessorAlu(1, shape, gidx);

    await backend.execute(AluExp.mul(arg1, arg2), [a, b], [c]);

    const buf = await backend.read(c);
    expect(new Float32Array(buf)).toEqual(new Float32Array([4, 10, 18]));

    await backend.execute(AluExp.add(arg1, arg2), [a, b], [c]);
    const buf2 = await backend.read(c);
    expect(new Float32Array(buf2)).toEqual(new Float32Array([5, 7, 9]));
  } finally {
    backend.decRef(a);
    backend.decRef(b);
    backend.decRef(c);
  }
});
