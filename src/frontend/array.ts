import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops";
import "@tensorflow/tfjs-core/dist/register_all_gradients";
import "@tensorflow/tfjs-backend-cpu";

import { DType } from "../alu";
import {
  AbstractValue,
  getAval,
  newMain,
  Primitive,
  ShapedArray,
  Trace,
  Tracer,
  TracerValue,
} from "./core";

tf.setBackend("cpu"); // TODO: support multiple devices, move arrays between devices

type ImplRule = (tracers: Array[], params: any) => Array[];

/**
 * Equivalent to `jnp.Array` from JAX, a tensor type.
 *
 * Not to be confused with the JavaScript "Array" constructor. Avoid importing
 * this into your code's namespace if you're already using the JavaScript
 * "Array" type by name.
 */
export class Array extends Tracer {
  readonly dtype: DType;

  constructor(readonly data: tf.Tensor) {
    super(baseArrayTrace);
    if (Object.values(DType).includes(data.dtype as any)) {
      this.dtype = data.dtype as DType;
    } else {
      throw new TypeError(`Unsupported dtype: ${data.dtype}`);
    }
  }

  get aval(): AbstractValue {
    return new ShapedArray(this.data.shape, this.dtype);
  }

  /** Return a simple string representation of the array's dimensions. */
  toString(): string {
    return `Array:${this.dtype}[${this.data.shape.join(",")}]`;
  }

  /** Convert this array into a JavaScript object (blocking). */
  js() {
    return this.data.arraySync();
  }

  /** Convert this array into a JavaScript object, asynchronously. */
  async jsAsync() {
    return await this.data.array();
  }

  // These need to be defined so they have access to the private properties.
  // static _implRules: Record<Primitive, ImplRule> = {
  //   [Primitive.Add]([x, y]) {
  //     x.#hello;
  //   },
  // };
}

/** If x is a value, lift it into an array, otherwise leave it be. */
export function pureArray(x: TracerValue): Tracer {
  if (x instanceof Tracer) {
    return x;
  } else {
    return new Array(tf.scalar(x));
  }
}

class EvalTrace extends Trace {
  // No boxing in Tracers needed.
  pure = (x: TracerValue) => pureArray(x);
  lift = (x: Tracer) => x;

  processPrimitive(
    primitive: Primitive,
    tracers: Array[],
    params: Record<string, any>,
  ): Tracer[] {
    return implRules[primitive](tracers, params);
  }
}

// Special bottom of the stack: must be level 0.
const baseArrayTrace = new EvalTrace(newMain(EvalTrace, null));

const implRules: Record<Primitive, ImplRule> = {
  [Primitive.Add]([x, y]) {
    return [new Array(tf.add(x.data, y.data))];
  },
  [Primitive.Mul]([x, y]) {
    return [new Array(tf.mul(x.data, y.data))];
  },
  [Primitive.Neg]([x]) {
    return [new Array(tf.neg(x.data))];
  },
  [Primitive.Sin]([x]) {
    return [new Array(tf.sin(x.data))];
  },
  [Primitive.Cos]([x]) {
    return [new Array(tf.cos(x.data))];
  },
  [Primitive.ReduceSum]([x], { axis }: { axis: number[] }) {
    return [new Array(tf.sum(x.data, axis))];
  },
  [Primitive.Greater]([x, y]) {
    return [new Array(tf.greater(x.data, y.data))];
  },
  [Primitive.Less]([x, y]) {
    return [new Array(tf.less(x.data, y.data))];
  },
  [Primitive.Transpose]([x], { perm }: { perm?: number[] }) {
    return [new Array(tf.transpose(x.data, perm))];
  },
  [Primitive.Broadcast](
    [x],
    { shape, axes }: { shape: number[]; axes: number[] },
  ) {
    let data = x.data;
    for (const axis of axes.toSorted()) {
      data = tf.expandDims(data, axis);
    }
    return [new Array(tf.broadcastTo(data, shape))];
  },
};

export function zerosLike(val: TracerValue): Array {
  const aval = getAval(val);
  return zeros(aval.shape, aval.dtype);
}

export function zeros(shape: number[], dtype: DType): Array {
  return new Array(tf.zeros(shape, dtype));
}
