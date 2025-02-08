import { expect, test } from "vitest";
import { jvp, numpy as np } from "jax-js";

// test("has webgpu", async () => {
//   const adapter = await navigator.gpu?.requestAdapter();
//   const device = await adapter?.requestDevice();
//   if (!adapter || !device) {
//     throw new Error("No adapter or device");
//   }
//   console.log(device.adapterInfo.architecture);
//   console.log(device.adapterInfo.vendor);
//   console.log(adapter.limits.maxVertexBufferArrayStride);
// });

/** Take the derivative of a simple function. */
function deriv(f: (x: np.Array) => np.Array): (x: np.ArrayLike) => np.Array {
  return (x) => {
    const [_y, dy] = jvp(f, [x], [1.0]);
    return dy;
  };
}

test("can create array", async () => {
  const x = 3.0;

  const result = jvp(
    (x: { a: np.Array; b: np.Array }) => x.a.add(x.b),
    [{ a: 1, b: 2 }],
    [{ a: 1, b: 0 }]
  );
  console.log(result[0].js());
  console.log(result[1].js());

  console.log(np.sin(x).js());
  console.log(deriv(np.sin)(x).js());
  console.log(deriv(deriv(np.sin))(x).js());
  console.log(deriv(deriv(deriv(np.sin)))(x).js());
});
