// Mirrors the `jax.lax` module in JAX.
//
// Unlike in JAX, this does not actually underpin `jax.numpy` as a more "core"
// set of operations, as they both build open the same foundations.

import { Array } from "./frontend/array";
import { conv } from "./frontend/core";

export function convGeneralDilated(lhs: Array, rhs: Array): Array {
  // XXX: Pass down padding and the other parameters.
  return conv(lhs, rhs) as Array;
}
